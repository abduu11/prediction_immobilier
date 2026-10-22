import json

import numpy as np
import pandas as pd
import pytest

from immo.config import PROJECT_ROOT, load_config
from immo.evaluate import evaluate_models, prepare_fit_set
from immo.features import GEO_INPUT, GEO_OUTPUT, NUMERIC, CommuneGeo, add_features
from immo.models import build_hgb, build_ridge
from immo.split import load_clean, temporal_split, year_splits

CFG = load_config()


def synthetic(n=600, seed=0, communes=("A", "B", "C", "D"), years=(2021, 2022, 2023)):
    """Ventes fictives dont le prix/m² dépend de la commune, du type et de la surface."""
    rng = np.random.default_rng(seed)
    comm = rng.choice(communes, n)
    base = {"A": 2000, "B": 3000, "C": 4000, "D": 5000, "Z": 6000}
    maison = rng.random(n) < 0.5
    surface = rng.uniform(25, 150, n)
    prix_m2 = np.array([base[c] for c in comm]) * np.where(maison, 0.9, 1.1) * (1 - surface / 600)
    prix_m2 *= rng.lognormal(0, 0.05, n)
    dates = pd.to_datetime([f"{rng.choice(years)}-{rng.integers(1, 13):02d}-15" for _ in range(n)])
    return pd.DataFrame({
        "id_mutation": [f"s{seed}-{i}" for i in range(n)], "date_mutation": dates,
        "code_commune": comm, "type_local": np.where(maison, "Maison", "Appartement"),
        "surface_reelle_bati": surface, "nombre_pieces_principales": np.clip(surface // 25, 1, 8),
        "surface_terrain": np.where(maison, rng.uniform(100, 900, n), np.nan),
        "has_dependance": rng.random(n) < 0.5, "nombre_lots": 1,
        "latitude": 44.5 + rng.random(n) * 0.5, "longitude": -0.8 + rng.random(n) * 0.5,
        "valeur_fonciere": prix_m2 * surface, "prix_m2": prix_m2,
    }).sort_values("date_mutation").reset_index(drop=True)


# --- variables ---
def test_add_features_values_and_no_missing():
    df = synthetic(50)
    X = add_features(df)
    assert X[NUMERIC].notna().all().all()
    assert np.allclose(X["surface_par_piece"], df["surface_reelle_bati"] / df["nombre_pieces_principales"])
    assert (X.loc[df.type_local == "Appartement", "log_terrain"] == 0).all()
    assert (X["commune_type"] == df["code_commune"] + "_" + df["type_local"]).all()


def test_add_features_is_stateless():
    df = synthetic(50)
    pd.testing.assert_frame_equal(add_features(df), add_features(df.iloc[::-1]).loc[df.index])


def test_commune_geo_imputes_from_train_centroids_and_falls_back():
    train = pd.DataFrame({"code_commune": ["A", "A", "B"], "latitude": [44.0, 46.0, 45.0],
                          "longitude": [-1.0, -1.0, 0.0]})
    geo = CommuneGeo().fit(train[GEO_INPUT])
    new = pd.DataFrame({"code_commune": ["A", "Z", "A"], "latitude": [np.nan, np.nan, 44.2],
                        "longitude": [np.nan, np.nan, -0.9]})
    out = pd.DataFrame(geo.transform(new[GEO_INPUT]), columns=GEO_OUTPUT)
    assert out.loc[0, "latitude"] == pytest.approx(45.0)            # centroïde de A appris sur le train
    assert out.loc[1, "latitude"] == pytest.approx(45.0)            # commune inconnue -> médiane globale
    assert out.loc[2, "latitude"] == pytest.approx(44.2)            # coordonnée présente conservée
    assert out.notna().all().all()


# --- pipelines ---
@pytest.mark.parametrize("builder", [lambda: build_ridge(alpha=1.0), lambda: build_hgb(max_iter=50)])
def test_models_fit_predict_positive_and_learn_signal(builder):
    train, test = synthetic(800, 0), synthetic(300, 1)
    model = builder().fit(add_features(train), train["prix_m2"])
    pred = model.predict(add_features(test))
    assert (pred > 0).all() and np.isfinite(pred).all()
    assert np.mean(np.abs(pred - test["prix_m2"])) < 0.25 * test["prix_m2"].mean()


def test_unseen_commune_at_prediction_time_does_not_crash():
    train = synthetic(600, 0)
    test = synthetic(50, 1, communes=("Z",))
    for model in (build_ridge(), build_hgb(max_iter=30)):
        pred = model.fit(add_features(train), train["prix_m2"]).predict(add_features(test))
        assert np.isfinite(pred).all()


def test_target_encoder_is_fitted_on_training_data_only():
    train = synthetic(600, 0, communes=("A", "B"))
    other = synthetic(100, 1, communes=("Z",))
    model = build_hgb(max_iter=30).fit(add_features(train), train["prix_m2"])
    te = model.regressor_.named_steps["prep"].named_transformers_["commune"].named_steps["te"]
    assert set(te.categories_[0]) == {"A", "B"}          # "Z" n'a jamais été vu à l'ajustement
    model.predict(add_features(other))                    # la prédiction ne modifie pas l'encodeur
    assert set(te.categories_[0]) == {"A", "B"}


def test_prediction_never_uses_the_target_column():
    train, test = synthetic(500, 0), synthetic(100, 1)
    model = build_ridge().fit(add_features(train), train["prix_m2"])
    p1 = model.predict(add_features(test))
    p2 = model.predict(add_features(test.assign(prix_m2=1.0, valeur_fonciere=1.0)))
    assert np.array_equal(p1, p2)


def test_hgb_is_reproducible():
    train = synthetic(500, 0)
    X, y = add_features(train), train["prix_m2"]
    p1 = build_hgb(max_iter=60).fit(X, y).predict(X)
    p2 = build_hgb(max_iter=60).fit(X, y).predict(X)
    assert np.array_equal(p1, p2)


# --- validation croisée temporelle ---
def test_year_splits_expanding_and_never_train_on_future():
    df = synthetic(400)
    splits = list(year_splits(df["date_mutation"]))
    assert len(splits) == 2                                  # 2021->2022, 2021-22->2023
    for tr, te in splits:
        assert df["date_mutation"].iloc[tr].max() < df["date_mutation"].iloc[te].min()
        assert not set(tr) & set(te)
    assert len(splits[1][0]) > len(splits[0][0])


def test_evaluate_models_reports_all_models_on_untrimmed_eval():
    fit, ev = synthetic(500, 0), synthetic(120, 1, years=(2024,))
    params = {"ridge": {"alpha": 10.0}, "hgb": {"max_iter": 40}}
    rep = evaluate_models(fit, ev, CFG, params)
    assert {"baseline_mediane_commune", "ridge", "hgb"} <= set(rep)
    assert rep["hgb"]["metrics"]["all"]["n"] == len(ev)
    assert rep["ridge"]["metrics"]["all"]["n"] == len(ev)


# --- vrais fichiers ---
@pytest.mark.network
def test_real_models_beat_baseline_on_validation():
    clean = PROJECT_ROOT / CFG["data"]["processed_dir"] / "ventes_clean.parquet"
    params_path = PROJECT_ROOT / CFG["clean"]["report_dir"] / "best_params.json"
    if not clean.exists() or not params_path.exists():
        pytest.skip("Lancer `python -m immo.clean` puis `python -m immo.train`")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    train, valid, _ = temporal_split(load_clean(CFG), CFG)
    rep = evaluate_models(train, valid, CFG, params)
    base = rep["baseline_mediane_commune"]["metrics"]["all"]["mae"]
    assert rep["ridge"]["metrics"]["all"]["mae"] < base
    assert rep["hgb"]["metrics"]["all"]["mae"] < 0.90 * base      # >= 10 % mieux que la baseline
    assert rep["hgb"]["metrics"]["all"]["mae"] < rep["ridge"]["metrics"]["all"]["mae"]
