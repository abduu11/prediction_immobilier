"""Variables explicatives.

`add_features` est sans état (calcul ligne à ligne, rien n'est appris) : elle peut donc être appelée
sur n'importe quel jeu sans risque de fuite. Tout ce qui s'apprend (centroïdes de communes,
encodage cible, normalisation) vit dans le `Pipeline` de `models.py`, ajusté sur le train seul.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

START_YEAR = 2021          # origine de l'index mensuel
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 79.0      # à ~44,8° N

NUMERIC = ["log_surface", "surface", "pieces", "surface_par_piece", "is_maison",
           "log_terrain", "has_dependance", "nombre_lots"]
TIME = ["mois_index"]
CATEGORICAL = ["code_commune", "commune_type"]
GEO_INPUT = ["code_commune", "latitude", "longitude"]
GEO_OUTPUT = ["latitude", "longitude", "geo_u", "geo_v"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    is_maison = (df["type_local"] == "Maison").astype(float)
    surface = df["surface_reelle_bati"].astype(float)
    pieces = df["nombre_pieces_principales"].astype(float)
    date = df["date_mutation"]
    X = pd.DataFrame(index=df.index)
    X["log_surface"] = np.log(surface)
    X["surface"] = surface
    X["pieces"] = pieces
    X["surface_par_piece"] = surface / pieces
    X["is_maison"] = is_maison
    X["log_terrain"] = np.log1p(df["surface_terrain"].astype(float).fillna(0)) * is_maison
    X["has_dependance"] = df["has_dependance"].astype(float)
    X["nombre_lots"] = df["nombre_lots"].astype(float).fillna(0)
    X["mois_index"] = ((date.dt.year - START_YEAR) * 12 + date.dt.month).astype(float)
    X["latitude"] = df["latitude"].astype(float)
    X["longitude"] = df["longitude"].astype(float)
    X["code_commune"] = df["code_commune"].astype(str).astype(object)
    X["commune_type"] = (df["code_commune"].astype(str) + "_" + df["type_local"].astype(str)).astype(object)
    return X


class CommuneGeo(BaseEstimator, TransformerMixin):
    """Complète les coordonnées manquantes par le centroïde de la commune (appris sur le train),
    puis ajoute deux axes tournés à 45° (en km) qui aident les arbres à tracer des frontières obliques.
    Entrée : colonnes [code_commune, latitude, longitude]. Sortie : GEO_OUTPUT.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=GEO_INPUT)
        geo = X.assign(latitude=X["latitude"].astype(float), longitude=X["longitude"].astype(float))
        self.centroids_ = geo.groupby("code_commune")[["latitude", "longitude"]].mean()
        self.global_ = geo[["latitude", "longitude"]].median()
        self.lat0_, self.lon0_ = self.global_["latitude"], self.global_["longitude"]
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=GEO_INPUT)
        lat = X["latitude"].astype(float)
        lon = X["longitude"].astype(float)
        cent = self.centroids_.reindex(X["code_commune"])
        lat = lat.fillna(pd.Series(cent["latitude"].to_numpy(), index=X.index)).fillna(self.global_["latitude"])
        lon = lon.fillna(pd.Series(cent["longitude"].to_numpy(), index=X.index)).fillna(self.global_["longitude"])
        x = (lon - self.lon0_) * KM_PER_DEG_LON
        y = (lat - self.lat0_) * KM_PER_DEG_LAT
        return np.column_stack([lat, lon, (x + y) / np.sqrt(2), (y - x) / np.sqrt(2)])

    def get_feature_names_out(self, input_features=None):
        return np.array(GEO_OUTPUT)
