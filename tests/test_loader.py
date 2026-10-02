import gzip

import pandas as pd
import pytest

from immo.config import PROJECT_ROOT, load_config
from immo.download import raw_path
from immo.loader import EXPECTED_COLUMNS, USECOLS, load_raw, read_dvf_file


def _write_mini_csv(path):
    header = ",".join(EXPECTED_COLUMNS)
    row = {c: "" for c in EXPECTED_COLUMNS}
    row.update(
        id_mutation="2024-1", date_mutation="2024-03-05", nature_mutation="Vente",
        valeur_fonciere="200000.0", code_commune="33063", nom_commune="Bordeaux",
        code_departement="33", type_local="Appartement", surface_reelle_bati="50.0",
        nombre_pieces_principales="2", longitude="-0.57", latitude="44.83", nombre_lots="1",
    )
    line = ",".join(row[c] for c in EXPECTED_COLUMNS)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(header + "\n" + line + "\n")


def test_usecols_subset_of_expected():
    assert set(USECOLS) <= set(EXPECTED_COLUMNS)


def test_read_dvf_file_types(tmp_path):
    p = tmp_path / "mini.csv.gz"
    _write_mini_csv(p)
    df = read_dvf_file(p)
    assert list(df.columns) == USECOLS
    assert pd.api.types.is_datetime64_any_dtype(df["date_mutation"])
    assert df.loc[0, "code_commune"] == "33063"          # code INSEE conservé en texte
    assert df.loc[0, "valeur_fonciere"] == 200000.0
    assert df.loc[0, "nombre_pieces_principales"] == 2


# --- Tests sur les vrais fichiers (nécessitent `python -m immo.download`) ---
@pytest.fixture(scope="module")
def raw_df():
    cfg = load_config()
    dep = str(cfg["data"]["departement"])
    for y in cfg["data"]["years"]:
        if not raw_path(PROJECT_ROOT / cfg["data"]["raw_dir"], y, dep).exists():
            pytest.skip("Fichiers DVF absents : lancer `python -m immo.download`")
    return load_raw(cfg)


@pytest.mark.network
def test_real_schema_matches_expected(raw_df):
    cfg = load_config()["data"]
    path = raw_path(PROJECT_ROOT / cfg["raw_dir"], cfg["years"][0], str(cfg["departement"]))
    cols = pd.read_csv(path, nrows=0, compression="gzip").columns.tolist()
    assert cols == EXPECTED_COLUMNS


@pytest.mark.network
def test_real_department_and_dates(raw_df):
    cfg = load_config()["data"]
    assert set(raw_df["code_departement"].unique()) == {str(cfg["departement"])}
    assert raw_df["date_mutation"].min() >= pd.Timestamp("2021-01-01")
    assert raw_df["date_mutation"].max() <= pd.Timestamp("2025-12-31")


@pytest.mark.network
def test_real_expected_categories(raw_df):
    assert "Vente" in set(raw_df["nature_mutation"].dropna().unique())
    assert {"Appartement", "Maison"} <= set(raw_df["type_local"].dropna().unique())


@pytest.mark.network
def test_real_coordinates_in_gironde_bbox(raw_df):
    geo = raw_df.dropna(subset=["latitude", "longitude"])
    assert len(geo) / len(raw_df) > 0.95
    assert geo["latitude"].between(44.1, 45.7).all()
    assert geo["longitude"].between(-1.6, 0.4).all()
