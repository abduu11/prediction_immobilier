import json

import numpy as np
import pandas as pd
import pytest

from immo.config import load_config
from immo.evaluate import final_report, fit_models, prepare_fit_set, run
from immo.intervals import (apply_factors, coverage_report, error_distribution, fit_ratio_factors,
                            to_frame)
from test_models import synthetic

CFG = load_config()


def noisy(n, sigma, seed, types=("Appartement", "Maison")):
    rng = np.random.default_rng(seed)
    pred = rng.uniform(1500, 5000, n)
    return pred * rng.lognormal(0, sigma, n), pred, rng.choice(types, n)


def test_factors_reach_target_coverage_on_fresh_data():
    y, p, t = noisy(20000, 0.2, 0)
    factors = fit_ratio_factors(y, p, t, coverage=0.8)
    y2, p2, t2 = noisy(20000, 0.2, 1)
    rep = coverage_report(y2, p2, t2, factors, 0.8)
    assert rep["all"]["couverture"] == pytest.approx(0.8, abs=0.02)
    assert factors["all"][0] < 1 < factors["all"][1]


def test_noisier_type_gets_wider_interval():
    y1, p1, _ = noisy(8000, 0.10, 0)
    y2, p2, _ = noisy(8000, 0.35, 1)
    y = np.r_[y1, y2]; p = np.r_[p1, p2]; t = np.r_[["Appartement"] * 8000, ["Maison"] * 8000]
    f = fit_ratio_factors(y, p, t, 0.8)
    assert (f["Maison"][1] - f["Maison"][0]) > 2 * (f["Appartement"][1] - f["Appartement"][0])
    rep = coverage_report(y, p, t, f, 0.8)
    assert rep["Maison"]["largeur_relative_mediane"] > rep["Appartement"]["largeur_relative_mediane"]
    assert rep["Appartement"]["couverture"] == pytest.approx(0.8, abs=0.03)   # couverture par type, pas seulement globale


def test_apply_factors_is_multiplicative_and_unknown_type_falls_back():
    f = {"all": [0.8, 1.3], "Maison": [0.7, 1.5]}
    lo, hi = apply_factors([1000.0, 1000.0, 1000.0], ["Maison", "Appartement", "Chateau"], f)
    assert list(lo) == [700.0, 800.0, 800.0] and list(hi) == [1500.0, 1300.0, 1300.0]


def test_error_distribution_known_values():
    d = error_distribution([100, 100, 100, 100], [105, 115, 125, 200], ["Maison"] * 4)
    assert d["all"]["moins_de_10_pct"] == 0.25 and d["all"]["moins_de_20_pct"] == 0.5
    assert d["all"]["moins_de_30_pct"] == 0.75 and d["all"]["moins_de_50_pct"] == 0.75


def test_to_frame_shape():
    assert list(to_frame({"all": [0.8, 1.2]}).columns) == ["facteur_bas", "facteur_haut"]


# --- évaluation finale ---
def fake_cfg():
    return {**CFG, "final": {"model": "hgb", "coverage": 0.8}}


def build_splits():
    train, valid, test = (synthetic(700, 0, years=(2021, 2022, 2023)), synthetic(300, 1, years=(2024,)),
                          synthetic(300, 2, years=(2025,)))
    params = {"hgb": {"max_iter": 40}}
    fit = pd.concat([train, valid])
    models = fit_models(prepare_fit_set(fit, CFG)[0], params)
    return train, valid, test, params, models


def test_final_report_structure_and_calibration_never_uses_test_targets():
    train, valid, test, params, models = build_splits()
    r1 = final_report(train, valid, test, models, params, fake_cfg())
    assert r1["modele_final"] == "hgb"
    cov = r1["fourchette"]["mesure_sur_test"]
    assert {"cible", "all", "Maison", "Appartement"} <= set(cov)
    assert 0.5 < cov["all"]["couverture"] <= 1.0
    # changer les prix du TEST ne modifie pas les facteurs (calibrés sur la validation uniquement)
    r2 = final_report(train, valid, test.assign(prix_m2=test["prix_m2"] * 3), models, params, fake_cfg())
    assert r1["fourchette"]["facteurs_bas_haut"] == r2["fourchette"]["facteurs_bas_haut"]
    assert r2["fourchette"]["mesure_sur_test"]["all"]["couverture"] < cov["all"]["couverture"]


def test_final_evaluation_refuses_to_run_twice(tmp_path):
    cfg = {**CFG, "clean": {**CFG["clean"], "report_dir": str(tmp_path)}}
    (tmp_path / "metrics_test.json").write_text(json.dumps({"evaluation": "test"}), encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(final=True, config=cfg)
    assert json.loads((tmp_path / "metrics_test.json").read_text(encoding="utf-8")) == {"evaluation": "test"}
