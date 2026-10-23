"""Estimation d'un bien avec fourchette de prix (utilisé par l'application Streamlit).

L'utilisateur ne donne pas de coordonnées : on utilise le point représentatif de la commune
(moyenne des ventes connues, sinon centroïde de son contour). L'estimation est donnée au niveau
de marché de la dernière vente connue (`date_reference`), pas à la date du jour.
"""
import pandas as pd

from immo.features import add_features
from immo.intervals import OVERALL

LOTS_DEFAUT = {"Appartement": 1, "Maison": 0}      # valeurs les plus fréquentes dans les données
TYPES = ["Appartement", "Maison"]


def build_input(communes: pd.DataFrame, code_commune: str, type_local: str, surface: float,
                pieces: int, terrain: float | None, dependance: bool,
                date_reference: pd.Timestamp) -> pd.DataFrame:
    if type_local not in TYPES:
        raise ValueError(f"type_local doit être l'un de {TYPES}")
    if code_commune not in communes.index:
        raise KeyError(f"commune inconnue : {code_commune}")
    c = communes.loc[code_commune]
    return pd.DataFrame({
        "date_mutation": [pd.Timestamp(date_reference)],
        "code_commune": [code_commune],
        "type_local": [type_local],
        "surface_reelle_bati": [float(surface)],
        "nombre_pieces_principales": [int(pieces)],
        "surface_terrain": [float(terrain) if (type_local == "Maison" and terrain) else float("nan")],
        "has_dependance": [bool(dependance)],
        "nombre_lots": [LOTS_DEFAUT[type_local]],
        "latitude": [float(c["latitude"])],
        "longitude": [float(c["longitude"])],
    })


def estimate(model, communes: pd.DataFrame, factors: dict, code_commune: str, type_local: str,
             surface: float, pieces: int, terrain: float | None, dependance: bool,
             date_reference: pd.Timestamp) -> dict:
    """Retourne l'estimation centrale et la fourchette, en €/m² et en prix total."""
    row = build_input(communes, code_commune, type_local, surface, pieces, terrain, dependance,
                      date_reference)
    pred = float(model.predict(add_features(row, communes))[0])
    lo_f, hi_f = factors.get(type_local, factors[OVERALL])
    return {
        "prix_m2": pred, "prix_m2_bas": pred * lo_f, "prix_m2_haut": pred * hi_f,
        "prix_total": pred * surface, "total_bas": pred * lo_f * surface,
        "total_haut": pred * hi_f * surface,
    }
