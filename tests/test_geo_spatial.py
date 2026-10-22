import numpy as np
import pandas as pd
import pytest

from immo.config import PROJECT_ROOT, load_config
from immo.features import GEO_INPUT, GEO_OUTPUT_EXTRA, POPULATION, CommuneGeo, add_features
from immo.geo import (BORDEAUX, communes_path, dist_bordeaux_km, dist_littoral_km, load_communes,
                      to_km)
from immo.maps import MIN_VENTES_ERREUR, choropleth_deck, error_table, price_table, rings
from immo.models import build_hgb, build_ridge
from immo.neighbors import NEIGHBOR_INPUT, NeighborPrice
from immo.spatial import block_groups, commune_groups, spatial_cv
from test_models import synthetic

CFG = load_config()


# --- distances ---
def test_distance_to_bordeaux_is_zero_at_center_and_symmetric():
    assert dist_bordeaux_km(*BORDEAUX) == pytest.approx(0.0)
    d1 = dist_bordeaux_km(BORDEAUX[0] + 0.1, BORDEAUX[1])
    assert d1 == pytest.approx(11.1, abs=0.1)                       # 0,1° de latitude ~ 11,1 km
    assert dist_bordeaux_km(BORDEAUX[0] - 0.1, BORDEAUX[1]) == pytest.approx(d1)


def test_distance_to_coast_known_places():
    assert dist_littoral_km(44.8378, -0.5792) == pytest.approx(51, abs=6)     # Bordeaux ~ 50 km
    assert dist_littoral_km(45.00, -1.19) < 3                                 # Lacanau-Océan en bord de mer
    assert dist_littoral_km(44.915, -0.243) > 70                              # Libourne, à l'intérieur
    assert dist_littoral_km(44.66, -1.17) < 12                                # Arcachon


def test_distances_vectorized_and_nan_safe():
    lat, lon = np.array([44.8, np.nan]), np.array([-0.6, -0.6])
    d = dist_littoral_km(lat, lon)
    assert d.shape == (2,) and np.isfinite(d[0]) and np.isnan(d[1])
    x, y = to_km(*BORDEAUX)
    assert x == 0 and y == 0


def test_commune_geo_extra_adds_distances():
    train = pd.DataFrame({"code_commune": ["A", "B"], "latitude": [44.8, 45.0], "longitude": [-0.6, -1.19]})
    out = CommuneGeo(extra=True).fit(train[GEO_INPUT]).transform(train[GEO_INPUT])
    assert out.shape == (2, len(GEO_OUTPUT_EXTRA))
    assert out[1, -1] < out[0, -1]                                  # B (Lacanau) plus près du littoral que A


# --- prix des voisins : uniquement des ventes antérieures ---
def neighbor_frame(days, lons, maison, y):
    X = pd.DataFrame({"code_commune": "A", "latitude": 44.8, "longitude": lons,
                      "jour_index": np.asarray(days, float), "is_maison": np.asarray(maison, float)})
    return X[NEIGHBOR_INPUT], np.asarray(y, float)


def test_neighbor_price_uses_only_strictly_past_sales():
    # 6 ventes de maison de plus en plus récentes, toutes au même endroit
    X, y = neighbor_frame(days=[10, 20, 30, 40, 50, 60], lons=-0.6, maison=1, y=[1, 2, 3, 4, 5, 6])
    nb = NeighborPrice(k=10, min_neighbors=3).fit(X, y)
    out = nb.transform(X)[:, 0]
    assert np.isnan(out[:3]).all()                                  # moins de 3 ventes antérieures
    assert out[3] == pytest.approx(2.0)                             # médiane de {1,2,3}
    assert out[5] == pytest.approx(3.0)                             # médiane de {1,2,3,4,5}


def test_neighbor_price_ignores_own_and_future_targets():
    days = [10, 20, 30, 40, 50, 60]
    X, y = neighbor_frame(days, -0.6, 1, [1, 2, 3, 4, 5, 6])
    base = NeighborPrice(min_neighbors=3).fit(X, y).transform(X)
    y2 = y.copy()
    y2[2] = 1000.0                                                  # on modifie la vente du jour 30
    changed = NeighborPrice(min_neighbors=3).fit(X, y2).transform(X)
    earlier_or_same = [0, 1, 2]                                     # la vente modifiée et celles d'avant
    assert np.allclose(base[earlier_or_same], changed[earlier_or_same], equal_nan=True)
    assert changed[4, 0] != base[4, 0]                              # les ventes ultérieures, elles, la voient
    assert changed[5, 0] != base[5, 0]


def test_neighbor_price_same_day_is_not_a_neighbor_and_types_are_separated():
    X, y = neighbor_frame(days=[10, 10, 10, 20, 20, 20, 30], lons=-0.6,
                          maison=[1, 1, 1, 0, 0, 0, 1], y=[1, 2, 3, 100, 100, 100, 5])
    out = NeighborPrice(min_neighbors=3).fit(X, y).transform(X)[:, 0]
    assert np.isnan(out[:3]).all()                                  # même jour : pas de voisin
    assert out[6] == pytest.approx(2.0)                             # seules les maisons : médiane de {1,2,3}
    new_apt, _ = neighbor_frame([40], -0.6, 0, [0])
    assert NeighborPrice(min_neighbors=3).fit(X, y).transform(new_apt)[0, 0] == pytest.approx(100.0)


def test_neighbor_price_prefers_nearest_sales():
    X, y = neighbor_frame(days=[1, 2, 3, 4, 5], lons=[-0.600, -0.601, -0.602, -0.9, -1.0],
                          maison=1, y=[10, 10, 10, 99, 99])
    new, _ = neighbor_frame([100], -0.6005, 1, [0])
    out = NeighborPrice(k=3, min_neighbors=3).fit(X, y).transform(new)
    assert out[0, 0] == pytest.approx(10.0)                         # les 3 plus proches, pas les lointaines


# --- modèles avec variables géographiques ---
def fake_communes():
    pop = {"A": 500, "B": 5000, "C": 50000, "D": 250000}
    df = pd.DataFrame({"population": pop}).rename_axis("code_commune")
    df["surface_km2"] = 10.0
    df["log_population"] = np.log1p(df["population"])
    df["log_densite"] = np.log1p(df["population"] / df["surface_km2"])
    return df


def test_add_features_with_communes_adds_population_columns():
    X = add_features(synthetic(30), fake_communes())
    assert set(POPULATION) <= set(X.columns) and X[POPULATION].notna().all().all()
    assert not set(POPULATION) & set(add_features(synthetic(30)).columns)


@pytest.mark.parametrize("builder", [lambda: build_ridge(alpha=10, geo=True),
                                     lambda: build_hgb(geo=True, max_iter=40)])
def test_geo_models_fit_predict_and_use_extra_features(builder):
    train, test = synthetic(800, 0), synthetic(200, 1, years=(2024,))
    c = fake_communes()
    model = builder().fit(add_features(train, c), train["prix_m2"])
    pred = model.predict(add_features(test, c))
    assert (pred > 0).all() and np.isfinite(pred).all()
    assert np.mean(np.abs(pred - test["prix_m2"])) < 0.25 * test["prix_m2"].mean()
    names = model.regressor_.named_steps["prep"].get_feature_names_out()
    assert {"dist_bordeaux_km", "dist_littoral_km", "knn_median_log", "log_population"} <= set(names)


# --- validation spatiale ---
def test_commune_and_block_groups():
    df = synthetic(300)
    assert set(commune_groups(df)) == {"A", "B", "C", "D"}
    b = block_groups(df, 5)
    assert len(b) == len(df)
    assert len(set(block_groups(df, 100))) < len(set(b))            # blocs plus grands -> moins de groupes


def test_spatial_cv_never_shares_a_group_and_baseline_falls_back_for_unseen_communes():
    df = synthetic(900, 0)
    out = spatial_cv(df, commune_groups(df), {"hgb": {"max_iter": 30}}, CFG, fake_communes(),
                     n_splits=4, models=["hgb"])
    assert set(out) == {"baseline", "hgb"}
    assert out["hgb"]["metrics"]["all"]["n"] == len(df)             # chaque vente évaluée exactement une fois
    assert len(out["hgb"]["mae_par_pli"]) == 4
    # une commune jamais vue : la baseline retombe sur la médiane du type, donc elle est moins bonne que le modèle
    assert out["hgb"]["metrics"]["all"]["mae"] < out["baseline"]["metrics"]["all"]["mae"]


# --- cartes ---
def test_price_and_error_tables_hide_small_communes():
    df = pd.DataFrame({"code_commune": ["A"] * 12 + ["B"] * 3, "prix_m2": [1000.0] * 12 + [2000.0] * 3})
    assert price_table(df)["prix_median"].to_dict()["A"] == 1000.0
    assert np.isnan(price_table(df).loc["B", "prix_median"])
    err = error_table(df, np.r_[np.full(12, 1100.0), np.full(3, 2000.0)])
    assert err.loc["A", "mae"] == pytest.approx(100.0)
    assert err.loc["A", "biais_pct"] == pytest.approx(10.0)
    assert 3 < MIN_VENTES_ERREUR and np.isnan(err.loc["B", "mae"])


def tiny_geojson():
    sq = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"code": "A", "nom": "Aa"}, "geometry": {"type": "Polygon", "coordinates": [sq]}},
        {"type": "Feature", "properties": {"code": "B", "nom": "Bb"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[sq], [sq]]}},
    ]}


def test_rings_and_choropleth_deck_build():
    gj = tiny_geojson()
    assert [len(rings(f)) for f in gj["features"]] == [1, 2]
    from matplotlib import colors
    values = pd.Series({"A": 1000.0, "B": np.nan})
    deck = choropleth_deck(gj, values, "YlOrRd", colors.Normalize(0, 2000), "Prix",
                           pd.DataFrame({"n": [5, 7]}, index=["A", "B"]))
    assert "GeoJsonLayer" in deck.to_json()
    assert gj["features"][0]["properties"].get("color") is None     # le geojson d'origine n'est pas modifié


# --- vrais fichiers ---
@pytest.mark.network
def test_real_communes_table():
    if not communes_path(CFG).exists():
        pytest.skip("Lancer `python -m immo.geo`")
    c = load_communes(CFG)
    assert len(c) > 500 and c.index.is_unique
    assert c.loc["33063", "population"] > 200_000                    # Bordeaux
    assert c["densite"].between(1, 20_000).all()
