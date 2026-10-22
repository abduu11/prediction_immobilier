"""Variables explicatives.

`add_features` est sans état (calcul ligne à ligne, rien n'est appris) : elle peut donc être appelée
sur n'importe quel jeu sans risque de fuite. Tout ce qui s'apprend (centroïdes de communes, prix des
voisins, encodage cible, normalisation) vit dans le `Pipeline` de `models.py`, ajusté sur le train seul.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from immo.geo import KM_PER_DEG_LAT, KM_PER_DEG_LON, dist_bordeaux_km, dist_littoral_km

START_DATE = pd.Timestamp("2021-01-01")   # origine des index temporels

NUMERIC = ["log_surface", "surface", "pieces", "surface_par_piece", "is_maison",
           "log_terrain", "has_dependance", "nombre_lots"]
TIME = ["mois_index"]
POPULATION = ["log_population", "log_densite"]     # nécessitent la table des communes (geo.py)
CATEGORICAL = ["code_commune", "commune_type"]
GEO_INPUT = ["code_commune", "latitude", "longitude"]
GEO_OUTPUT = ["latitude", "longitude", "geo_u", "geo_v"]
GEO_OUTPUT_EXTRA = GEO_OUTPUT + ["dist_bordeaux_km", "dist_littoral_km"]


def add_features(df: pd.DataFrame, communes: pd.DataFrame | None = None) -> pd.DataFrame:
    """`communes` : table de geo.load_communes() ; si fournie, ajoute population et densité."""
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
    X["mois_index"] = ((date.dt.year - START_DATE.year) * 12 + date.dt.month).astype(float)
    X["jour_index"] = (date - START_DATE).dt.days.astype(float)
    X["latitude"] = df["latitude"].astype(float)
    X["longitude"] = df["longitude"].astype(float)
    X["code_commune"] = df["code_commune"].astype(str).astype(object)
    X["commune_type"] = (df["code_commune"].astype(str) + "_" + df["type_local"].astype(str)).astype(object)
    if communes is not None:
        for col in POPULATION:
            X[col] = X["code_commune"].map(communes[col]).astype(float)
    return X


class CommuneGeo(BaseEstimator, TransformerMixin):
    """Complète les coordonnées manquantes par le centroïde de la commune (appris sur le train),
    puis ajoute deux axes tournés à 45° (en km) qui aident les arbres à tracer des frontières obliques.
    Avec `extra=True`, ajoute aussi la distance au centre de Bordeaux et au littoral.
    Entrée : colonnes GEO_INPUT. Sortie : GEO_OUTPUT (ou GEO_OUTPUT_EXTRA).
    """

    def __init__(self, extra: bool = False):
        self.extra = extra

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
        cols = [lat, lon, (x + y) / np.sqrt(2), (y - x) / np.sqrt(2)]
        if self.extra:
            cols += [dist_bordeaux_km(lat, lon), dist_littoral_km(lat, lon)]
        return np.column_stack(cols)

    def get_feature_names_out(self, input_features=None):
        return np.array(GEO_OUTPUT_EXTRA if self.extra else GEO_OUTPUT)
