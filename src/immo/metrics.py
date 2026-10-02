"""Métriques de régression sur le prix au m²."""
import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),            # €/m²
        "mape": float(np.mean(ape) * 100),             # %
        "median_ape": float(np.median(ape) * 100),     # %
        "bias": float(np.mean(err)),                   # €/m² (>0 : surestimation)
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def metrics_by_group(df: pd.DataFrame, y_pred, group: str, target: str = "prix_m2") -> dict:
    """Métriques globales et ventilées par modalité de `group` (ex. type_local)."""
    y_pred = pd.Series(np.asarray(y_pred, dtype=float), index=df.index)
    out = {"all": regression_metrics(df[target], y_pred)}
    for key, idx in df.groupby(group).groups.items():
        out[str(key)] = regression_metrics(df.loc[idx, target], y_pred.loc[idx])
    return out
