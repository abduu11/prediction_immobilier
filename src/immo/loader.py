"""Lecture typée des fichiers DVF géolocalisés (colonnes utiles, types économes en mémoire)."""
from pathlib import Path

import pandas as pd

from immo.config import PROJECT_ROOT, load_config
from immo.download import raw_path

# Schéma complet du fichier geo-DVF (vérifié sur dvf_2024_33.csv.gz)
EXPECTED_COLUMNS = [
    "id_mutation", "date_mutation", "numero_disposition", "nature_mutation",
    "valeur_fonciere", "adresse_numero", "adresse_suffixe", "adresse_nom_voie",
    "adresse_code_voie", "code_postal", "code_commune", "nom_commune",
    "code_departement", "ancien_code_commune", "ancien_nom_commune",
    "id_parcelle", "ancien_id_parcelle", "numero_volume",
    "lot1_numero", "lot1_surface_carrez", "lot2_numero", "lot2_surface_carrez",
    "lot3_numero", "lot3_surface_carrez", "lot4_numero", "lot4_surface_carrez",
    "lot5_numero", "lot5_surface_carrez", "nombre_lots", "code_type_local",
    "type_local", "surface_reelle_bati", "nombre_pieces_principales",
    "code_nature_culture", "nature_culture", "code_nature_culture_speciale",
    "nature_culture_speciale", "surface_terrain", "longitude", "latitude",
]

# Colonnes conservées pour la suite du pipeline (sprint 2+)
USECOLS = [
    "id_mutation", "date_mutation", "numero_disposition", "nature_mutation",
    "valeur_fonciere", "adresse_numero", "adresse_nom_voie", "code_postal",
    "code_commune", "nom_commune", "code_departement", "id_parcelle",
    "lot1_numero", "nombre_lots", "type_local", "surface_reelle_bati",
    "nombre_pieces_principales", "code_nature_culture",
    "code_nature_culture_speciale", "surface_terrain", "longitude", "latitude",
]

DTYPES = {
    "id_mutation": "string",
    "numero_disposition": "string",
    "lot1_numero": "string",
    "code_nature_culture": "string",
    "code_nature_culture_speciale": "string",
    "nature_mutation": "category",
    "valeur_fonciere": "float64",
    "adresse_numero": "string",
    "adresse_nom_voie": "string",
    "code_postal": "string",
    "code_commune": "string",
    "nom_commune": "category",
    "code_departement": "string",
    "id_parcelle": "string",
    "nombre_lots": "Int16",
    "type_local": "category",
    "surface_reelle_bati": "float32",
    "nombre_pieces_principales": "Int16",
    "surface_terrain": "float32",
    "longitude": "float64",
    "latitude": "float64",
}


def read_dvf_file(path: Path) -> pd.DataFrame:
    """Lit un fichier DVF .csv.gz avec colonnes utiles et types explicites."""
    return pd.read_csv(
        path,
        usecols=USECOLS,
        dtype=DTYPES,
        parse_dates=["date_mutation"],
        compression="gzip",
    )


def load_raw(config: dict | None = None) -> pd.DataFrame:
    """Concatène les fichiers annuels du département configuré."""
    cfg = (config or load_config())["data"]
    raw_dir = PROJECT_ROOT / cfg["raw_dir"]
    dep = str(cfg["departement"])
    frames = [read_dvf_file(raw_path(raw_dir, y, dep)) for y in cfg["years"]]
    return pd.concat(frames, ignore_index=True)
