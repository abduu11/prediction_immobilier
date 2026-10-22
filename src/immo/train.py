"""Réglage des hyperparamètres (validation croisée temporelle sur le TRAIN) puis évaluation.

    python -m immo.train

- Le réglage n'utilise que le train 2021-2023 (folds : 2021 -> 2022, 2021-22 -> 2023).
- Critère : MAE en €/m² (sur l'échelle d'origine).
- Quatre modèles : Ridge et HGB, sans puis avec les variables géographiques dérivées (`_geo`).
- Les meilleurs paramètres sont écrits dans reports/best_params.json ; la validation 2024 sert
  ensuite à comparer les modèles. Le test 2025 n'est jamais utilisé ici.
"""
import json

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from immo import evaluate
from immo.config import PROJECT_ROOT, load_config
from immo.geo import load_communes
from immo.models import SEED, build_hgb, build_ridge
from immo.split import load_clean, temporal_split, year_splits

PREFIX = "regressor__model__"
RIDGE_GRID = {PREFIX + "alpha": [0.1, 1, 10, 100, 1000]}
HGB_DISTRIBUTIONS = {
    PREFIX + "learning_rate": [0.03, 0.05, 0.1],
    PREFIX + "max_leaf_nodes": [15, 31, 63, 127],
    PREFIX + "min_samples_leaf": [10, 20, 50, 100, 200],
    PREFIX + "l2_regularization": [0.0, 1.0, 10.0],
    PREFIX + "max_iter": [300, 600, 1000],
    PREFIX + "loss": ["squared_error", "absolute_error"],
}


def _clean(params: dict) -> dict:
    return {k.removeprefix(PREFIX): (v.item() if isinstance(v, np.generic) else v)
            for k, v in params.items()}


def tune(train_trimmed, communes, n_iter: int = 25) -> dict:
    y = train_trimmed["prix_m2"]
    cv = list(year_splits(train_trimmed["date_mutation"]))
    common = dict(scoring="neg_mean_absolute_error", cv=cv, n_jobs=2, refit=False)
    out, cv_mae = {}, {}
    for name, (builder, geo) in evaluate.MODEL_SPECS.items():
        X = evaluate.model_inputs(train_trimmed, geo, communes)
        if builder is build_ridge:
            base = build_ridge(poly=True, use_time=False, geo=geo)
            search = GridSearchCV(base, RIDGE_GRID, **common).fit(X, y)
            out[name] = {"poly": True, "use_time": False, **_clean(search.best_params_)}
        else:
            base = build_hgb(use_time=True, geo=geo)
            search = RandomizedSearchCV(base, HGB_DISTRIBUTIONS, n_iter=n_iter,
                                        random_state=SEED, **common).fit(X, y)
            out[name] = {"use_time": True, **_clean(search.best_params_)}
        cv_mae[name] = round(-search.best_score_, 1)
        print(f"  {name:10s} CV MAE = {cv_mae[name]}")
    out["cv_mae"] = cv_mae
    return out


def run(config: dict | None = None) -> dict:
    cfg = config or load_config()
    train, _, _ = temporal_split(load_clean(cfg), cfg)
    train_trimmed, _ = evaluate.prepare_fit_set(train, cfg)

    print("Réglage (validation croisée temporelle sur le train) :")
    params = tune(train_trimmed, load_communes(cfg))
    rep_dir = PROJECT_ROOT / cfg["clean"]["report_dir"]
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "best_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print("Meilleurs paramètres :", json.dumps(params, indent=2))

    return evaluate.run(final=False, config=cfg, save_models=True)


if __name__ == "__main__":
    run()
