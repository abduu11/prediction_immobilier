"""Pipelines scikit-learn : régression linéaire régularisée (Ridge) et arbres (HistGradientBoosting).

Tout ce qui est appris (centroïdes, prix des voisins, encodage cible par commune, normalisation) est
dans le Pipeline : à chaque `fit`, seules les données d'ajustement sont vues. La cible est modélisée
en log(prix/m²) ; les prédictions sont ramenées en €/m² (médiane conditionnelle, adaptée à la MAE).

`geo=True` ajoute les variables géographiques dérivées (distance à Bordeaux et au littoral,
population/densité de la commune, prix des ventes voisines antérieures). Les données d'entrée doivent
alors venir de `add_features(df, communes)`.
"""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, TargetEncoder

from immo.features import CATEGORICAL, GEO_INPUT, NUMERIC, POPULATION, TIME, CommuneGeo
from immo.neighbors import NEIGHBOR_INPUT, NeighborPrice

SEED = 0


def _preprocessor(scale: bool, use_time: bool, geo: bool) -> ColumnTransformer:
    def tail():
        # imputation (médiane) puis normalisation pour Ridge ; HGB gère les NaN et n'a pas besoin d'échelle
        steps = [("impute", SimpleImputer(strategy="median"))]
        return steps + ([("scale", StandardScaler())] if scale else [])

    te = TargetEncoder(target_type="continuous", smooth="auto", cv=5, shuffle=True, random_state=SEED)
    parts = [
        ("geo", Pipeline([("centroid", CommuneGeo(extra=geo))] + tail()), GEO_INPUT),
        ("commune", Pipeline([("te", te)] + tail()), CATEGORICAL),
        ("num", Pipeline(tail()), NUMERIC + (TIME if use_time else []) + (POPULATION if geo else [])),
    ]
    if geo:
        parts.append(("knn", Pipeline([("knn", NeighborPrice())] + tail()), NEIGHBOR_INPUT))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def build_ridge(alpha: float = 1.0, poly: bool = True, use_time: bool = False, geo: bool = False):
    steps = [("prep", _preprocessor(scale=True, use_time=use_time, geo=geo))]
    if poly:   # interactions d'ordre 2 (ex. surface x type x localisation), maîtrisées par la régularisation
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
        steps.append(("scale2", StandardScaler()))
    steps.append(("model", Ridge(alpha=alpha)))
    return TransformedTargetRegressor(Pipeline(steps), func=np.log, inverse_func=np.exp)


def build_hgb(use_time: bool = True, geo: bool = False, **params):
    defaults = dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=50,
                    l2_regularization=1.0, early_stopping=True, n_iter_no_change=20, random_state=SEED)
    model = HistGradientBoostingRegressor(**{**defaults, **params})
    pipe = Pipeline([("prep", _preprocessor(scale=False, use_time=use_time, geo=geo)), ("model", model)])
    return TransformedTargetRegressor(pipe, func=np.log, inverse_func=np.exp)
