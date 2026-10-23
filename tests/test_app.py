import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from immo.export import commune_table, ring_centroid, simplify_geojson
from immo.predict import LOTS_DEFAUT, build_input, estimate

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "app" / "artifacts"


def communes():
    df = pd.DataFrame({"nom": ["Aa", "Bb"], "latitude": [44.8, 44.9], "longitude": [-0.6, -0.5],
                       "log_population": [8.0, 6.0], "log_densite": [6.0, 3.0]},
                      index=pd.Index(["33001", "33002"], name="code_commune"))
    return df


class Stub:
    """Faux modèle : retourne toujours 3 000 €/m² (permet de tester la mise en forme sans le vrai modèle)."""
    def predict(self, X):
        assert "log_population" in X.columns            # les variables de commune sont bien ajoutées
        return np.full(len(X), 3000.0)


FACTORS = {"all": [0.7, 1.3], "Appartement": [0.8, 1.2], "Maison": [0.6, 1.4]}
REF = pd.Timestamp("2025-12-31")


def test_build_input_defaults_and_commune_point():
    row = build_input(communes(), "33001", "Appartement", 60, 3, 500.0, True, REF)
    assert row.loc[0, "nombre_lots"] == LOTS_DEFAUT["Appartement"]
    assert np.isnan(row.loc[0, "surface_terrain"])                  # le terrain ne compte que pour une maison
    assert row.loc[0, "latitude"] == 44.8 and row.loc[0, "has_dependance"]
    house = build_input(communes(), "33002", "Maison", 100, 4, 400.0, False, REF)
    assert house.loc[0, "surface_terrain"] == 400.0 and house.loc[0, "nombre_lots"] == 0


def test_build_input_rejects_bad_values():
    with pytest.raises(ValueError):
        build_input(communes(), "33001", "Château", 60, 3, None, False, REF)
    with pytest.raises(KeyError):
        build_input(communes(), "99999", "Maison", 60, 3, None, False, REF)


def test_estimate_applies_type_specific_factors_and_total():
    a = estimate(Stub(), communes(), FACTORS, "33001", "Appartement", 50, 2, None, False, REF)
    m = estimate(Stub(), communes(), FACTORS, "33001", "Maison", 50, 2, 300.0, False, REF)
    assert (a["prix_m2_bas"], a["prix_m2"], a["prix_m2_haut"]) == (2400.0, 3000.0, 3600.0)
    assert (m["prix_m2_bas"], m["prix_m2_haut"]) == (1800.0, 4200.0)         # maison : fourchette plus large
    assert a["prix_total"] == 150_000.0 and a["total_bas"] == 120_000.0 and a["total_haut"] == 180_000.0
    assert a["prix_m2_bas"] < a["prix_m2"] < a["prix_m2_haut"]


# --- export ---
def test_ring_centroid_of_square():
    lon, lat = ring_centroid(np.array([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]], float))
    assert (lon, lat) == pytest.approx((1.0, 1.0))


def test_simplify_geojson_rounds_and_keeps_only_needed_properties():
    ring = [[0.123456, 0.0], [0.123457, 0.0], [1.0, 0.0], [1.0, 1.0], [0.123456, 0.0]]
    gj = {"features": [{"type": "Feature", "properties": {"code": "A", "nom": "Aa", "population": 5,
                                                          "codesPostaux": ["1"], "surface": 9},
                        "geometry": {"type": "Polygon", "coordinates": [ring]}}]}
    out = simplify_geojson(gj)
    f = out["features"][0]
    assert set(f["properties"]) == {"code", "nom", "population"}
    pts = f["geometry"]["coordinates"][0]
    assert pts[0] == [0.1235, 0.0] and len(pts) < len(ring)             # arrondi + point dupliqué retiré


def test_commune_table_uses_sales_point_then_polygon_centroid():
    gj = {"features": [
        {"properties": {"code": "A", "nom": "Aa", "codesPostaux": ["33000"]},
         "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}},
        {"properties": {"code": "B", "nom": "Bb", "codesPostaux": []},
         "geometry": {"type": "Polygon", "coordinates": [[[10, 10], [12, 10], [12, 12], [10, 12], [10, 10]]]}}]}
    base = pd.DataFrame({"population": [100, 200], "surface_km2": [1.0, 1.0], "densite": [100.0, 200.0],
                         "log_population": [1.0, 2.0], "log_densite": [1.0, 2.0]},
                        index=pd.Index(["A", "B"], name="code_commune"))
    sales = pd.DataFrame({"code_commune": ["A", "A"], "latitude": [44.0, 46.0], "longitude": [-1.0, -3.0],
                          "prix_m2": [1.0, 2.0]})
    import immo.export as ex
    orig = ex.load_communes
    ex.load_communes = lambda cfg: base
    try:
        t = commune_table(sales, gj, {})
    finally:
        ex.load_communes = orig
    assert (t.loc["A", "latitude"], t.loc["A", "longitude"], t.loc["A", "n_ventes"]) == (45.0, -2.0, 2)
    assert (t.loc["B", "latitude"], t.loc["B", "longitude"], t.loc["B", "n_ventes"]) == (11.0, 11.0, 0)


# --- l'application complète (nécessite `python -m immo.export`) ---
@pytest.fixture(scope="module")
def app():
    if not (ARTIFACTS / "model.joblib").exists():
        pytest.skip("Lancer `python -m immo.export`")
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()


@pytest.mark.network
def test_app_runs_and_shows_a_range(app):
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "€/m²" in text and " – " in text                          # fourchette affichée
    assert len(app.tabs) == 3
    texts = [e.value.lower() for e in list(app.markdown) + list(app.caption)]
    assert any("version 1" in t or "v1" in t for t in texts)


@pytest.mark.network
def test_app_house_shows_terrain_field_and_wider_range(app):
    labels_apt = [n.label for n in app.number_input]
    assert "Surface du terrain (m²)" not in labels_apt
    apt = " ".join(m.value for m in app.markdown if "–" in m.value and "€/m²" in m.value and m.value.startswith("###"))
    app.radio[0].set_value("Maison").run()
    assert not app.exception
    assert "Surface du terrain (m²)" in [n.label for n in app.number_input]
    house = " ".join(m.value for m in app.markdown if "–" in m.value and "€/m²" in m.value and m.value.startswith("###"))
    assert apt != house
