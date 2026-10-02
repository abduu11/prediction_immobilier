"""Évaluation des modèles sur le découpage temporel.

    python -m immo.evaluate           # ajuste sur le train, évalue sur la validation (2024)
    python -m immo.evaluate --test    # ÉVALUATION FINALE : ajuste sur train+validation, évalue sur le test (2025)

Le test ne doit être lancé qu'une fois, quand les choix de modélisation sont figés.
Les bornes de quantiles du prix/m² sont estimées sur les données d'ajustement uniquement et
servent à filtrer ces données ; les jeux d'évaluation ne sont jamais tronqués.
"""
import argparse
import json

import pandas as pd

from immo.baseline import MedianBaseline
from immo.clean import apply_price_bounds, fit_price_bounds
from immo.config import PROJECT_ROOT, load_config
from immo.metrics import metrics_by_group
from immo.split import load_clean, temporal_split


def prepare_fit_set(fit_df: pd.DataFrame, cfg: dict):
    """Applique les bornes P1-P99 estimées sur `fit_df` lui-même (jamais sur l'évaluation)."""
    bounds = fit_price_bounds(fit_df, tuple(cfg["clean"]["quantile_bounds"]))
    return apply_price_bounds(fit_df, bounds), bounds


def evaluate_baseline(fit_df: pd.DataFrame, eval_df: pd.DataFrame, cfg: dict) -> dict:
    fit_trimmed, bounds = prepare_fit_set(fit_df, cfg)
    model = MedianBaseline(cfg["baseline"]["min_ventes_commune"]).fit(fit_trimmed)
    return {
        "n_fit": len(fit_trimmed),
        "n_fit_avant_bornes": len(fit_df),
        "bornes_prix_m2": {t: [round(lo, 1), round(hi, 1)] for t, (lo, hi) in bounds.items()},
        "taux_repli_type": round(model.fallback_rate(eval_df), 4),
        "metrics": metrics_by_group(eval_df, model.predict(eval_df), "type_local"),
    }


def run(final: bool = False, config: dict | None = None) -> dict:
    cfg = config or load_config()
    train, valid, test = temporal_split(load_clean(cfg), cfg)
    if final:
        fit_df, eval_df, name = pd.concat([train, valid]), test, "test"
    else:
        fit_df, eval_df, name = train, valid, "validation"

    report = {"evaluation": name, "baseline_mediane_commune": evaluate_baseline(fit_df, eval_df, cfg)}
    rep_dir = PROJECT_ROOT / cfg["clean"]["report_dir"]
    rep_dir.mkdir(parents=True, exist_ok=True)
    out = rep_dir / ("metrics_test.json" if final else "metrics.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    m = report["baseline_mediane_commune"]["metrics"]
    print(f"Évaluation sur {name} ({len(eval_df)} ventes), baseline médiane par commune :")
    for key, v in m.items():
        print(f"  {key:12s} n={v['n']:6d}  MAE={v['mae']:7.0f} €/m²  MAPE={v['mape']:5.1f} %  "
              f"MedAPE={v['median_ape']:5.1f} %  biais={v['bias']:+6.0f}  R²={v['r2']:.3f}")
    print(f"-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="évaluation finale sur le test (une seule fois)")
    run(final=ap.parse_args().test)
