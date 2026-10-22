"""Prix des ventes voisines, calculé uniquement à partir de ventes ANTÉRIEURES du jeu d'ajustement.

Pour une vente à la date d, la variable est la médiane du log(prix/m²) des k ventes de même type
(maison / appartement) les plus proches géographiquement parmi celles du jeu d'ajustement dont la
date est strictement antérieure à d. Conséquences :
- une vente d'entraînement ne voit jamais elle-même, ni le futur, ni une vente du même jour ;
- une vente de validation/test ne voit que le passé de l'entraînement ;
- les mêmes règles s'appliquent dans `fit_transform` et `transform` (un seul chemin de calcul).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, TransformerMixin

from immo.features import GEO_INPUT, CommuneGeo
from immo.geo import to_km

NEIGHBOR_INPUT = GEO_INPUT + ["jour_index", "is_maison"]
NEIGHBOR_OUTPUT = ["knn_median_log", "knn_dist_km"]


class NeighborPrice(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 10, n_candidates: int = 150, min_neighbors: int = 3):
        self.k = k
        self.n_candidates = n_candidates
        self.min_neighbors = min_neighbors

    def _prepare(self, X):
        X = pd.DataFrame(X, columns=NEIGHBOR_INPUT)
        geo = self.geo_.transform(X[GEO_INPUT])
        x, y = to_km(geo[:, 0], geo[:, 1])
        return (np.column_stack([x, y]), X["jour_index"].to_numpy(float),
                X["is_maison"].to_numpy(float))

    def fit(self, X, y):
        X = pd.DataFrame(X, columns=NEIGHBOR_INPUT)
        self.geo_ = CommuneGeo().fit(X[GEO_INPUT])
        pts, days, maison = self._prepare(X)
        y = np.asarray(y, dtype=float)
        self.groups_ = {}
        for t in (0.0, 1.0):
            m = maison == t
            if m.any():
                self.groups_[t] = (cKDTree(pts[m]), days[m], y[m])
        return self

    def transform(self, X):
        pts, days, maison = self._prepare(X)
        out = np.full((len(pts), 2), np.nan)
        for t, (tree, days_t, y_t) in self.groups_.items():
            rows = np.flatnonzero(maison == t)
            if rows.size == 0:
                continue
            kk = min(self.n_candidates, len(days_t))
            dist, idx = tree.query(pts[rows], k=kk)
            dist, idx = dist.reshape(len(rows), kk), idx.reshape(len(rows), kk)
            valid = days_t[idx] < days[rows][:, None]            # strictement dans le passé
            selected = valid & (np.cumsum(valid, axis=1) <= self.k)
            enough = selected.sum(axis=1) >= self.min_neighbors
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(np.where(selected, y_t[idx], np.nan), axis=1)
                dmean = np.nanmean(np.where(selected, dist, np.nan), axis=1)
            out[rows, 0] = np.where(enough, med, np.nan)
            out[rows, 1] = np.where(enough, dmean, np.nan)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(NEIGHBOR_OUTPUT)
