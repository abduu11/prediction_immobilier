"""Baseline : médiane du prix/m² par (commune, type de bien), repli sur la médiane du type."""
import numpy as np
import pandas as pd

COMMUNE = "code_commune"
TYPE = "type_local"


class MedianBaseline:
    def __init__(self, min_ventes: int = 5):
        self.min_ventes = min_ventes

    def fit(self, train: pd.DataFrame) -> "MedianBaseline":
        stats = (train.groupby([COMMUNE, TYPE])["prix_m2"].agg(["median", "size"])
                 .reset_index())
        stats = stats[stats["size"] >= self.min_ventes]
        self.commune_median_ = stats[[COMMUNE, TYPE, "median"]].rename(
            columns={"median": "_pred"})
        self.type_median_ = train.groupby(TYPE)["prix_m2"].median()
        return self

    def _commune_pred(self, df: pd.DataFrame) -> np.ndarray:
        merged = df[[COMMUNE, TYPE]].merge(self.commune_median_, on=[COMMUNE, TYPE], how="left")
        return merged["_pred"].to_numpy(dtype=float)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Ne lit jamais la cible : seules les colonnes commune et type sont utilisées."""
        pred = self._commune_pred(df)
        fallback = df[TYPE].map(self.type_median_).to_numpy(dtype=float)
        return np.where(np.isnan(pred), fallback, pred)

    def fallback_rate(self, df: pd.DataFrame) -> float:
        """Part des ventes sans médiane communale suffisante dans le train."""
        return float(np.isnan(self._commune_pred(df)).mean())
