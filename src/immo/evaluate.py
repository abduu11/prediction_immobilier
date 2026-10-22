"""Évaluation des modèles sur le découpage temporel.

    python -m immo.evaluate           # ajuste sur le train, évalue sur la validation (2024)
    python -m immo.evaluate --test    # ÉVALUATION FINALE : ajuste sur train+validation, évalue sur le test (2025)

Les hyperparamètres viennent de reports/best_params.json (produit par `python -m immo.train`).
Le test ne doit être lancé qu'une fois, quand les choix de modélisation sont figés.
Les bornes de quantiles du prix/m² sont estimées sur les données d'ajustement uniquement et
servent à filtrer ces données ; les jeux d'évaluation ne sont jamais tronqués.
"""
import argparse
import json

import joblib
import pandas as pd

from immo.baseline import MedianBaseline
from immo.clean import apply_price_bounds, fit_price_bounds
from immo.config import PROJECT_ROOT, load_config
from immo.features import add_features
from immo.metrics import metrics_by_group
from immo.models import build_hgb, build_ridge
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


def fit_models(fit_trimmed: pd.DataFrame, params: dict) -> dict:
    X, y = add_features(fit_trimmed), fit_trimmed["prix_m2"]
    return {
        "ridge": build_ridge(**params["ridge"]).fit(X, y),
        "hgb": build_hgb(**params["hgb"]).fit(X, y),
    }


def evaluate_models(fit_df: pd.DataFrame, eval_df: pd.DataFrame, cfg: dict,
                    params: dict | None = None, return_models: bool = False):
    report = {"baseline_mediane_commune": evaluate_baseline(fit_df, eval_df, cfg)}
    models = {}
    if params:
        fit_trimmed, _ = prepare_fit_set(fit_df, cfg)
        models = fit_models(fit_trimmed, params)
        X_eval = add_features(eval_df)
        for name, model in models.items():
            report[name] = {"params": params[name],
                            "metrics": metrics_by_group(eval_df, model.predict(X_eval), "type_local")}
    return (report, models) if return_models else report


def print_report(report: dict, name: str, n: int) -> None:
    base = report["baseline_mediane_commune"]["metrics"]
    print(f"Évaluation sur {name} ({n} ventes) :")
    for model, res in report.items():
        if not isinstance(res, dict) or "metrics" not in res:
            continue
        for key, v in res["metrics"].items():
            gain = ""
            if model != "baseline_mediane_commune":
                gain = f"  ({(v['mae'] / base[key]['mae'] - 1) * 100:+5.1f} % vs baseline)"
            print(f"  {model:26s} {key:12s} n={v['n']:6d}  MAE={v['mae']:6.0f} €/m²  MAPE={v['mape']:5.1f} %  "
                  f"MedAPE={v['median_ape']:5.1f} %  R²={v['r2']:.3f}{gain}")


def load_params(cfg: dict) -> dict | None:
    path = PROJECT_ROOT / cfg["clean"]["report_dir"] / "best_params.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run(final: bool = False, config: dict | None = None, save_models: bool = False) -> dict:
    cfg = config or load_config()
    train, valid, test = temporal_split(load_clean(cfg), cfg)
    if final:
        fit_df, eval_df, name = pd.concat([train, valid]), test, "test"
    else:
        fit_df, eval_df, name = train, valid, "validation"

    results, models = evaluate_models(fit_df, eval_df, cfg, load_params(cfg), return_models=True)
    report = {"evaluation": name, **results}
    if save_models:
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        for model_name, model in models.items():
            joblib.dump(model, models_dir / f"{model_name}_{name}.joblib")
    rep_dir = PROJECT_ROOT / cfg["clean"]["report_dir"]
    rep_dir.mkdir(parents=True, exist_ok=True)
    out = rep_dir / ("metrics_test.json" if final else "metrics.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report(report, name, len(eval_df))
    print(f"-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="évaluation finale sur le test (une seule fois)")
    run(final=ap.parse_args().test)
