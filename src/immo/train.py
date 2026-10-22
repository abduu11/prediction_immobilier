"""Réglage des hyperparamètres (validation croisée temporelle sur le TRAIN) puis évaluation.

    python -m immo.train

- Le réglage n'utilise que le train 2021-2023 (folds : 2021 -> 2022, 2021-22 -> 2023).
- Critère : MAE en €/m² (sur l'échelle d'origine).
- Les meilleurs paramètres sont écrits dans reports/best_params.json ; la validation 2024 sert
  ensuite à comparer les modèles. Le test 2025 n'est jamais utilisé ici.
"""
import json

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from immo import evaluate
from immo.config import PROJECT_ROOT, load_config
from immo.features import add_features
from immo.models import SEED, build_hgb, build_ridge
from immo.split import load_clean, temporal_split, year_splits

RIDGE_GRID = {"regressor__model__alpha": [0.1, 1, 10, 100, 1000]}
HGB_DISTRIBUTIONS = {
    "regressor__model__learning_rate": [0.03, 0.05, 0.1],
    "regressor__model__max_leaf_nodes": [15, 31, 63],
    "regressor__model__min_samples_leaf": [20, 50, 100, 200],
    "regressor__model__l2_regularization": [0.0, 1.0, 10.0],
    "regressor__model__max_iter": [300, 600],
    "regressor__model__loss": ["squared_error", "absolute_error"],
}


def _strip(params: dict) -> dict:
    return {k.removeprefix("regressor__model__"): v for k, v in params.items()}


def tune(train_trimmed, n_iter: int = 25) -> dict:
    X, y = add_features(train_trimmed), train_trimmed["prix_m2"]
    cv = list(year_splits(train_trimmed["date_mutation"]))
    common = dict(scoring="neg_mean_absolute_error", cv=cv, n_jobs=2, refit=False)

    ridge = GridSearchCV(build_ridge(poly=True, use_time=False), RIDGE_GRID, **common).fit(X, y)
    hgb = RandomizedSearchCV(build_hgb(use_time=True), HGB_DISTRIBUTIONS, n_iter=n_iter,
                             random_state=SEED, **common).fit(X, y)
    return {
        "ridge": {"poly": True, "use_time": False, **_strip(ridge.best_params_)},
        "hgb": {"use_time": True, **{k: (v.item() if isinstance(v, np.generic) else v)
                                     for k, v in _strip(hgb.best_params_).items()}},
        "cv_mae": {"ridge": round(-ridge.best_score_, 1), "hgb": round(-hgb.best_score_, 1)},
    }


def run(config: dict | None = None) -> dict:
    cfg = config or load_config()
    train, _, _ = temporal_split(load_clean(cfg), cfg)
    train_trimmed, _ = evaluate.prepare_fit_set(train, cfg)

    params = tune(train_trimmed)
    rep_dir = PROJECT_ROOT / cfg["clean"]["report_dir"]
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "best_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print("Meilleurs paramètres :", json.dumps(params, indent=2))

    return evaluate.run(final=False, config=cfg, save_models=True)


if __name__ == "__main__":
    run()
