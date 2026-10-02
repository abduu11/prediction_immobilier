import pandas as pd
import pytest

from immo.clean import OUTPUT_COLUMNS, apply_price_bounds, clean_dvf, fit_price_bounds
from immo.config import PROJECT_ROOT, load_config
from immo.download import raw_path
from immo.loader import USECOLS, load_raw

CFG = load_config()


def row(mut, type_local="Appartement", surf=50.0, pieces=2, valeur=200000.0, nature="Vente",
        disp="000001", parcelle="P1", lot="1", culture=None, terrain=None, lat=44.8, lon=-0.6):
    base = dict.fromkeys(USECOLS)
    base.update(
        id_mutation=mut, date_mutation=pd.Timestamp("2024-05-01"), numero_disposition=disp,
        nature_mutation=nature, valeur_fonciere=valeur, code_commune="33063",
        nom_commune="Bordeaux", code_postal="33000", code_departement="33",
        id_parcelle=parcelle, lot1_numero=lot, nombre_lots=1, type_local=type_local,
        surface_reelle_bati=surf, nombre_pieces_principales=pieces,
        code_nature_culture=culture, surface_terrain=terrain, latitude=lat, longitude=lon,
    )
    return base


def make(rows):
    df = pd.DataFrame(rows)
    df["nature_mutation"] = df["nature_mutation"].astype("category")
    df["type_local"] = df["type_local"].astype("category")
    return df


def clean(rows):
    return clean_dvf(make(rows), CFG)


def test_simple_apartment_with_dependance_is_kept():
    out, _ = clean([row("A"), row("A", type_local="Dépendance", surf=0.0, pieces=0, lot=None)])
    assert list(out["id_mutation"]) == ["A"]
    assert out.loc[0, "has_dependance"]
    assert out.loc[0, "prix_m2"] == pytest.approx(4000.0)


def test_two_apartments_in_one_mutation_are_excluded():
    out, _ = clean([row("B", lot="1"), row("B", lot="2")])
    assert out.empty


def test_same_house_repeated_per_culture_is_merged_and_land_summed():
    out, _ = clean([
        row("C", "Maison", surf=100, pieces=4, lot=None, culture="S", terrain=600.0),
        row("C", "Maison", surf=100, pieces=4, lot=None, culture="J", terrain=400.0),
    ])
    assert len(out) == 1
    assert out.loc[0, "surface_terrain"] == 1000.0


def test_land_shared_with_dependance_row_not_double_counted():
    out, _ = clean([
        row("L", "Maison", surf=100, pieces=4, lot=None, culture="S", terrain=600.0),
        row("L", "Dépendance", surf=0.0, pieces=0, lot=None, culture="S", terrain=600.0),
    ])
    assert out.loc[0, "surface_terrain"] == 600.0


def test_land_only_rows_add_land():
    out, _ = clean([
        row("T", "Maison", surf=100, pieces=4, lot=None, parcelle="P1", culture="S", terrain=500.0),
        row("T", None, surf=None, pieces=None, lot=None, parcelle="P2", culture="T", terrain=300.0),
    ])
    assert out.loc[0, "surface_terrain"] == 800.0


@pytest.mark.parametrize("kwargs", [
    dict(nature="Vente en l'état futur d'achèvement"),
    dict(nature="Echange"),
    dict(valeur=None),
    dict(valeur=5000.0),            # prix symbolique
    dict(surf=5.0),                 # surface trop petite
    dict(pieces=0),
    dict(valeur=2_000_000.0),       # 40 000 €/m² : hors garde-fous
])
def test_row_filters(kwargs):
    out, _ = clean([row("X", **kwargs)])
    assert out.empty


def test_commercial_local_in_mutation_excludes_it():
    out, _ = clean([row("E", "Maison", lot=None),
                    row("E", "Local industriel. commercial ou assimilé", lot=None)])
    assert out.empty


def test_land_only_and_multi_disposition_are_excluded():
    out, _ = clean([
        row("F", None, surf=None, pieces=None, lot=None),
        row("G", disp="000001"), row("G", disp="000002", lot="2"),
    ])
    assert out.empty


def test_house_with_huge_land_is_excluded():
    out, _ = clean([row("H", "Maison", surf=100, pieces=4, lot=None, culture="S", terrain=50000.0)])
    assert out.empty


def test_missing_coordinates_are_flagged_not_dropped():
    out, _ = clean([row("M", lat=None, lon=None)])
    assert len(out) == 1 and out.loc[0, "coords_missing"]


def test_output_contract_and_funnel():
    rows = [row("A"), row("B", lot="1"), row("B", lot="2"), row("D", nature="Echange")]
    out, funnel = clean(rows)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out["id_mutation"].is_unique
    assert funnel["mutations"].is_monotonic_decreasing
    assert funnel["mutations"].iloc[-1] == len(out) == 1
    assert funnel["etape"].iloc[0] == "Lignes brutes"


def test_price_bounds_use_train_only():
    train = pd.DataFrame({"type_local": ["Maison"] * 100, "prix_m2": range(1000, 1100)})
    test = pd.DataFrame({"type_local": ["Maison"] * 3, "prix_m2": [500.0, 1050.0, 90000.0]})
    bounds = fit_price_bounds(train, (0.01, 0.99))
    kept = apply_price_bounds(test, bounds)
    assert list(kept["prix_m2"]) == [1050.0]
    # les bornes ne changent pas si on change le jeu de test
    assert fit_price_bounds(train, (0.01, 0.99)) == bounds


# --- Tests sur les vrais fichiers ---
@pytest.fixture(scope="module")
def real_clean():
    d = CFG["data"]
    for y in d["years"]:
        if not raw_path(PROJECT_ROOT / d["raw_dir"], y, str(d["departement"])).exists():
            pytest.skip("Fichiers DVF absents : lancer `python -m immo.download`")
    return clean_dvf(load_raw(CFG), CFG)


@pytest.mark.network
def test_real_one_row_per_sale_and_bounds(real_clean):
    out, funnel = real_clean
    c = CFG["clean"]
    assert out["id_mutation"].is_unique
    assert set(out["type_local"]) == {"Appartement", "Maison"}
    assert out[["valeur_fonciere", "surface_reelle_bati", "prix_m2",
                "nombre_pieces_principales", "date_mutation"]].notna().all().all()
    assert out["prix_m2"].between(*c["prix_m2_bounds"]).all()
    assert out["date_mutation"].between("2021-01-01", "2025-12-31").all()
    assert (out["prix_m2"] - out["valeur_fonciere"] / out["surface_reelle_bati"]).abs().max() < 1e-6


@pytest.mark.network
def test_real_funnel_and_retention(real_clean):
    out, funnel = real_clean
    assert funnel["mutations"].is_monotonic_decreasing
    assert funnel["mutations"].iloc[-1] == len(out)
    assert 50_000 < len(out) < 125_513   # ventes appart/maison simples de Gironde
