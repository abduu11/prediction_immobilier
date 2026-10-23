"""Fourchette de prix calibrée sur les erreurs observées.

Principe : sur un jeu de calibration (la validation, jamais vue à l'ajustement du modèle évalué), on mesure
le rapport r = prix réel / prix prédit. Ses quantiles (par type de bien) donnent des facteurs multiplicatifs :

    fourchette = [prédit x q_bas , prédit x q_haut]    avec q_bas, q_haut = quantiles (1-c)/2 et 1-(1-c)/2 de r

Avec c = 0,8 on vise que ~80 % des ventes tombent dans la fourchette. La couverture réelle est mesurée sur
un jeu jamais utilisé pour la calibration (le test) : c'est elle qu'on publie, pas la cible.
Les maisons, moins bien prédites, obtiennent naturellement une fourchette plus large que les appartements.
"""
import numpy as np
import pandas as pd

OVERALL = "all"


def fit_ratio_factors(y_true, y_pred, types, coverage: float = 0.8) -> dict:
    """Facteurs multiplicatifs (bas, haut) par type de bien, plus un jeu global de repli."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    types = np.asarray(types)
    ratio = y_true / y_pred
    alpha = (1 - coverage) / 2

    def q(r):
        return [float(np.quantile(r, alpha)), float(np.quantile(r, 1 - alpha))]

    factors = {OVERALL: q(ratio)}
    for t in np.unique(types):
        factors[str(t)] = q(ratio[types == t])
    return factors


def apply_factors(y_pred, types, factors: dict):
    """Retourne (bas, haut) en €/m² ; un type inconnu utilise le jeu global."""
    y_pred = np.asarray(y_pred, float)
    lo = np.array([factors.get(str(t), factors[OVERALL])[0] for t in types])
    hi = np.array([factors.get(str(t), factors[OVERALL])[1] for t in types])
    return y_pred * lo, y_pred * hi


def coverage_report(y_true, y_pred, types, factors: dict, target: float) -> dict:
    """Couverture réelle et largeur relative médiane, globalement et par type."""
    y_true, y_pred, types = np.asarray(y_true, float), np.asarray(y_pred, float), np.asarray(types)
    lo, hi = apply_factors(y_pred, types, factors)
    inside = (y_true >= lo) & (y_true <= hi)
    width = (hi - lo) / y_pred

    def block(mask):
        return {"n": int(mask.sum()), "couverture": float(inside[mask].mean()),
                "largeur_relative_mediane": float(np.median(width[mask]))}

    out = {"cible": target, OVERALL: block(np.ones(len(y_true), bool))}
    for t in np.unique(types):
        out[str(t)] = block(types == t)
    return out


def error_distribution(y_true, y_pred, types, thresholds=(0.10, 0.20, 0.30, 0.50)) -> dict:
    """Part des ventes dont l'erreur relative est sous chaque seuil, globalement et par type."""
    y_true, y_pred, types = np.asarray(y_true, float), np.asarray(y_pred, float), np.asarray(types)
    rel = np.abs(y_pred - y_true) / y_true
    out = {}
    for name, mask in [(OVERALL, np.ones(len(rel), bool))] + [(str(t), types == t) for t in np.unique(types)]:
        out[name] = {f"moins_de_{int(th * 100)}_pct": float((rel[mask] < th).mean()) for th in thresholds}
    return out


def to_frame(factors: dict) -> pd.DataFrame:
    return pd.DataFrame(factors, index=["facteur_bas", "facteur_haut"]).T
