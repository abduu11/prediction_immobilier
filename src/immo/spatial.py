"""Validation spatiale : le modèle sait-il prédire dans une zone qu'il n'a pas vue à l'entraînement ?

    python -m immo.spatial

Deux schémas de GroupKFold (5 plis), sur les ventes 2021-2024 (le test 2025 n'est pas utilisé) :
- "commune" : chaque commune est entièrement dans le pli d'évaluation ou entièrement dans l'entraînement ;
- "bloc"    : même principe avec des blocs carrés (config `spatial.block_km`), plus sévère car les
              communes voisines d'une zone évaluée sont aussi retirées de l'entraînement (au moins en partie).

Ce schéma ignore volontairement le temps : il isole l'effet de la géographie. Les bornes de quantiles du
prix/m² sont ré-estimées dans chaque pli sur ses seules données d'entraînement ; l'évaluation n'est jamais tronquée.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from immo import evaluate
from immo.baseline import MedianBaseline
from immo.config import PROJECT_ROOT, load_config
from immo.geo import load_communes, to_km
from immo.metrics import metrics_by_group, regression_metrics
from immo.split import load_clean, temporal_split

MODELS = ["hgb", "hgb_geo"]


def commune_groups(df: pd.DataFrame) -> np.ndarray:
    return df["code_commune"].astype(str).to_numpy()


def block_groups(df: pd.DataFrame, block_km: float) -> np.ndarray:
    """Carré de `block_km` km contenant la vente (coordonnée manquante -> centroïde de sa commune)."""
    lat, lon = df["latitude"].astype(float), df["longitude"].astype(float)
    cent = df.assign(lat=lat, lon=lon).groupby("code_commune")[["lat", "lon"]].transform("mean")
    lat, lon = lat.fillna(cent["lat"]), lon.fillna(cent["lon"])
    x, y = to_km(lat, lon)
    bx = np.floor(x / block_km).astype(int).astype(str)
    by = np.floor(y / block_km).astype(int).astype(str)
    return np.char.add(np.char.add(bx, "_"), by)


def spatial_cv(df: pd.DataFrame, groups: np.ndarray, params: dict, cfg: dict,
               communes: pd.DataFrame, n_splits: int = 5, models=MODELS) -> dict:
    df = df.reset_index(drop=True)
    preds = {name: np.full(len(df), np.nan) for name in ["baseline", *models]}
    fold_mae = {name: [] for name in preds}
    for tr, te in GroupKFold(n_splits=n_splits).split(df, groups=groups):
        assert not set(groups[tr]) & set(groups[te])          # aucune zone partagée
        fit_trimmed, _ = evaluate.prepare_fit_set(df.iloc[tr], cfg)
        held = df.iloc[te]
        preds["baseline"][te] = MedianBaseline(cfg["baseline"]["min_ventes_commune"]).fit(fit_trimmed).predict(held)
        fitted = evaluate.fit_models(fit_trimmed, {m: params[m] for m in models}, communes)
        for name, model in fitted.items():
            geo = evaluate.MODEL_SPECS[name][1]
            preds[name][te] = model.predict(evaluate.model_inputs(held, geo, communes))
        for name in preds:
            fold_mae[name].append(round(regression_metrics(held["prix_m2"], preds[name][te])["mae"], 1))
    return {name: {"metrics": metrics_by_group(df, p, "type_local"), "mae_par_pli": fold_mae[name]}
            for name, p in preds.items()}


def run(config: dict | None = None) -> dict:
    cfg = config or load_config()
    params = json.loads((PROJECT_ROOT / cfg["clean"]["report_dir"] / "best_params.json").read_text(encoding="utf-8"))
    train, valid, _ = temporal_split(load_clean(cfg), cfg)
    df = pd.concat([train, valid]).reset_index(drop=True)
    communes = load_communes(cfg)
    s = cfg["spatial"]

    report = {"donnees": f"ventes {min(cfg['split']['train_years'])}-{max(cfg['split']['valid_years'])}",
              "n_ventes": len(df), "n_plis": s["n_splits"]}
    schemes = {"commune": commune_groups(df), f"bloc_{s['block_km']}km": block_groups(df, s["block_km"])}
    for name, groups in schemes.items():
        print(f"Validation spatiale par {name} ({len(set(groups))} groupes) ...")
        report[name] = {"n_groupes": int(len(set(groups))),
                        **spatial_cv(df, groups, params, cfg, communes, s["n_splits"])}
        for model, res in report[name].items():
            if isinstance(res, dict) and "metrics" in res:
                m = res["metrics"]["all"]
                print(f"  {model:10s} MAE={m['mae']:6.0f} €/m²  MAPE={m['mape']:5.1f} %  "
                      f"MedAPE={m['median_ape']:5.1f} %  R²={m['r2']:.3f}")
    out = PROJECT_ROOT / cfg["clean"]["report_dir"] / "spatial_cv.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}")
    return report


if __name__ == "__main__":
    run()
