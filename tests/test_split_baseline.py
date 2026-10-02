import numpy as np
import pandas as pd
import pytest

from immo.baseline import MedianBaseline
from immo.clean import apply_price_bounds, fit_price_bounds
from immo.config import PROJECT_ROOT, load_config
from immo.evaluate import evaluate_baseline
from immo.metrics import metrics_by_group, regression_metrics
from immo.split import load_clean, temporal_split

CFG = load_config()


def sales(commune, type_local, prices, year=2022):
    return pd.DataFrame({
        "id_mutation": [f"{year}-{commune}-{type_local}-{i}" for i in range(len(prices))],
        "date_mutation": pd.Timestamp(f"{year}-06-01"),
        "code_commune": commune, "type_local": type_local, "prix_m2": np.array(prices, float),
    })


# --- métriques ---
def test_regression_metrics_known_values():
    m = regression_metrics([100, 200], [110, 180])
    assert m["mae"] == pytest.approx(15.0)
    assert m["mape"] == pytest.approx(10.0)
    assert m["bias"] == pytest.approx(-5.0)
    assert m["n"] == 2


def test_perfect_prediction():
    m = regression_metrics([100, 200, 300], [100, 200, 300])
    assert m["mae"] == 0 and m["mape"] == 0 and m["r2"] == pytest.approx(1.0)


def test_metrics_by_group_keys():
    df = pd.concat([sales("A", "Maison", [1000, 2000]), sales("A", "Appartement", [3000])],
                   ignore_index=True)
    out = metrics_by_group(df, df["prix_m2"].to_numpy(), "type_local")
    assert set(out) == {"all", "Maison", "Appartement"}
    assert out["Maison"]["n"] == 2


# --- découpage temporel ---
def test_temporal_split_is_strictly_ordered_and_disjoint():
    df = pd.concat([sales("A", "Maison", [1, 2], y) for y in range(2021, 2026)], ignore_index=True)
    train, valid, test = temporal_split(df, CFG)
    assert train["date_mutation"].max() < valid["date_mutation"].min()
    assert valid["date_mutation"].max() < test["date_mutation"].min()
    ids = [set(x["id_mutation"]) for x in (train, valid, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(train) + len(valid) + len(test) == len(df)


def test_temporal_split_rejects_overlapping_periods():
    bad = {"split": {"train_years": [2021, 2024], "valid_years": [2024], "test_years": [2025]}}
    with pytest.raises(ValueError):
        temporal_split(sales("A", "Maison", [1]), bad)


# --- baseline ---
@pytest.fixture
def train():
    return pd.concat([
        sales("A", "Maison", [1000, 2000, 3000, 4000, 5000]),   # médiane 3000, 5 ventes
        sales("B", "Maison", [9000, 9000]),                     # 2 ventes : < min_ventes
        sales("A", "Appartement", [4000] * 6),                  # médiane 4000
    ], ignore_index=True)


def test_baseline_commune_median_and_fallback(train):
    model = MedianBaseline(min_ventes=5).fit(train)
    new = pd.concat([sales("A", "Maison", [0]), sales("B", "Maison", [0]),
                     sales("Z", "Appartement", [0])], ignore_index=True)
    pred = model.predict(new)
    assert pred[0] == 3000                       # médiane communale
    assert pred[1] == train.loc[train.type_local == "Maison", "prix_m2"].median()  # repli (commune B trop petite)
    assert pred[2] == 4000                       # commune inconnue -> médiane du type
    assert model.fallback_rate(new) == pytest.approx(2 / 3)


def test_baseline_is_reproducible_and_ignores_target(train):
    new = sales("A", "Maison", [1, 2, 3])
    p1 = MedianBaseline().fit(train).predict(new)
    p2 = MedianBaseline().fit(train.sample(frac=1, random_state=0)).predict(new)
    assert np.array_equal(p1, p2)
    # la prédiction ne dépend pas de la cible du jeu évalué (pas de fuite)
    assert np.array_equal(p1, MedianBaseline().fit(train).predict(new.drop(columns="prix_m2")))


def test_evaluation_set_is_not_trimmed_by_price_bounds():
    fit = sales("A", "Maison", list(range(1000, 1100)))
    ev = sales("A", "Maison", [500.0, 1050.0, 90000.0], year=2024)
    out = evaluate_baseline(fit, ev, CFG)
    assert out["metrics"]["all"]["n"] == 3                  # aucun point d'évaluation retiré
    assert out["n_fit"] < out["n_fit_avant_bornes"]         # le train, lui, est tronqué
    bounds = fit_price_bounds(fit, tuple(CFG["clean"]["quantile_bounds"]))
    assert len(apply_price_bounds(fit, bounds)) == out["n_fit"]


# --- vrais fichiers ---
@pytest.fixture(scope="module")
def real():
    path = PROJECT_ROOT / CFG["data"]["processed_dir"] / "ventes_clean.parquet"
    if not path.exists():
        pytest.skip("Données nettoyées absentes : lancer `python -m immo.clean`")
    return temporal_split(load_clean(CFG), CFG)


@pytest.mark.network
def test_real_split_has_no_temporal_leakage(real):
    train, valid, test = real
    assert train["date_mutation"].max() < valid["date_mutation"].min()
    assert valid["date_mutation"].max() < test["date_mutation"].min()
    assert min(len(train), len(valid), len(test)) > 10_000


@pytest.mark.network
def test_real_commune_baseline_beats_global_type_median(real):
    train, valid, _ = real
    out = evaluate_baseline(train, valid, CFG)
    glob = valid["type_local"].map(train.groupby("type_local")["prix_m2"].median())
    glob_mae = regression_metrics(valid["prix_m2"], glob)["mae"]
    assert out["metrics"]["all"]["mae"] < glob_mae
