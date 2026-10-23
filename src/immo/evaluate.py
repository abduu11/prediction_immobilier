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
from immo.geo import load_communes
from immo.intervals import coverage_report, error_distribution, fit_ratio_factors
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


# nom -> (constructeur, variables géographiques dérivées ?)
MODEL_SPECS = {
    "ridge": (build_ridge, False),
    "hgb": (build_hgb, False),
    "ridge_geo": (build_ridge, True),
    "hgb_geo": (build_hgb, True),
}


def model_inputs(df: pd.DataFrame, geo: bool, communes: pd.DataFrame | None) -> pd.DataFrame:
    return add_features(df, communes if geo else None)


def fit_models(fit_trimmed: pd.DataFrame, params: dict, communes: pd.DataFrame | None = None) -> dict:
    y = fit_trimmed["prix_m2"]
    models = {}
    for name, (builder, geo) in MODEL_SPECS.items():
        if name in params:
            kwargs = {**params[name], **({"geo": True} if geo else {})}
            models[name] = builder(**kwargs).fit(model_inputs(fit_trimmed, geo, communes), y)
    return models


def evaluate_models(fit_df: pd.DataFrame, eval_df: pd.DataFrame, cfg: dict,
                    params: dict | None = None, return_models: bool = False,
                    communes: pd.DataFrame | None = None):
    report = {"baseline_mediane_commune": evaluate_baseline(fit_df, eval_df, cfg)}
    models = {}
    if params:
        if communes is None and any(MODEL_SPECS[n][1] for n in params if n in MODEL_SPECS):
            communes = load_communes(cfg)
        fit_trimmed, _ = prepare_fit_set(fit_df, cfg)
        models = fit_models(fit_trimmed, params, communes)
        for name, model in models.items():
            pred = model.predict(model_inputs(eval_df, MODEL_SPECS[name][1], communes))
            report[name] = {"params": params[name],
                            "metrics": metrics_by_group(eval_df, pred, "type_local")}
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


def final_report(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, models: dict,
                 params: dict, cfg: dict) -> dict:
    """Fourchette de prix et répartition des erreurs du modèle retenu.

    Calibration : modèle ajusté sur le train seul, erreurs mesurées sur la validation.
    Mesure : modèle final (train+validation) sur le test, qui n'a servi ni à ajuster ni à calibrer.
    """
    name, target = cfg["final"]["model"], cfg["final"]["coverage"]
    geo = MODEL_SPECS[name][1]
    communes = load_communes(cfg) if geo else None
    calib = fit_models(prepare_fit_set(train, cfg)[0], {name: params[name]}, communes)[name]
    p_valid = calib.predict(model_inputs(valid, geo, communes))
    factors = fit_ratio_factors(valid["prix_m2"], p_valid, valid["type_local"], target)
    p_test = models[name].predict(model_inputs(test, geo, communes))
    return {
        "modele_final": name,
        "fourchette": {
            "calibration": "validation (modèle ajusté sur le train seul)",
            "facteurs_bas_haut": factors,
            "mesure_sur_test": coverage_report(test["prix_m2"], p_test, test["type_local"], factors, target),
        },
        "repartition_erreurs_test": error_distribution(test["prix_m2"], p_test, test["type_local"]),
    }


def run(final: bool = False, config: dict | None = None, save_models: bool = False,
        force: bool = False) -> dict:
    cfg = config or load_config()
    rep_dir = PROJECT_ROOT / cfg["clean"]["report_dir"]
    out = rep_dir / ("metrics_test.json" if final else "metrics.json")
    if final and out.exists() and not force:
        raise FileExistsError(f"{out} existe déjà : le test ne doit être évalué qu'une fois (--force pour passer outre)")

    train, valid, test = temporal_split(load_clean(cfg), cfg)
    if final:
        fit_df, eval_df, name = pd.concat([train, valid]), test, "test"
    else:
        fit_df, eval_df, name = train, valid, "validation"

    params = load_params(cfg)
    if final and not params:
        raise FileNotFoundError(f"{rep_dir / 'best_params.json'} absent : lancer `python -m immo.train`")
    results, models = evaluate_models(fit_df, eval_df, cfg, params, return_models=True)
    report = {"evaluation": name, **results}
    if final:
        report.update(final_report(train, valid, test, models, params, cfg))
    if save_models:
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        for model_name, model in models.items():
            joblib.dump(model, models_dir / f"{model_name}_{name}.joblib")
    rep_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if final:
        (rep_dir / "intervalles.json").write_text(
            json.dumps(report["fourchette"]["facteurs_bas_haut"], indent=2), encoding="utf-8")
    print_report(report, name, len(eval_df))
    if final:
        print_final(report)
    print(f"-> {out}")
    return report


def print_final(report: dict) -> None:
    f = report["fourchette"]
    print(f"\nFourchette ({report['modele_final']}), cible de couverture {f['mesure_sur_test']['cible']:.0%} :")
    for key, v in f["mesure_sur_test"].items():
        if isinstance(v, dict):
            fac = f["facteurs_bas_haut"][key]
            print(f"  {key:12s} facteurs x{fac[0]:.2f} .. x{fac[1]:.2f}  couverture test = {v['couverture']:.1%}  "
                  f"largeur relative médiane = {v['largeur_relative_mediane']:.0%}")
    print("Répartition des erreurs sur le test :")
    for key, v in report["repartition_erreurs_test"].items():
        print(f"  {key:12s} " + "  ".join(f"{k.replace('moins_de_', '<').replace('_pct', ' %')}: {x:.0%}" for k, x in v.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="évaluation finale sur le test (une seule fois)")
    ap.add_argument("--force", action="store_true", help="relancer le test malgré un rapport existant")
    a = ap.parse_args()
    run(final=a.test, save_models=a.test, force=a.force)
