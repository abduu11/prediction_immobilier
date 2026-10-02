"""Chargement des ventes nettoyées et découpage temporel train / validation / test."""
import pandas as pd

from immo.config import PROJECT_ROOT, load_config


def load_clean(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    path = PROJECT_ROOT / cfg["data"]["processed_dir"] / "ventes_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} absent : lancer `python -m immo.clean`")
    return pd.read_parquet(path)


def temporal_split(df: pd.DataFrame, config: dict | None = None):
    """Découpe selon l'année de la date de mutation (config.yaml, section `split`).

    Aucune vente du test/validation n'est antérieure à une vente du train : le modèle
    ne voit jamais le futur.
    """
    s = (config or load_config())["split"]
    years = df["date_mutation"].dt.year
    train = df[years.isin(s["train_years"])]
    valid = df[years.isin(s["valid_years"])]
    test = df[years.isin(s["test_years"])]
    if not (max(s["train_years"]) < min(s["valid_years"])
            and max(s["valid_years"]) < min(s["test_years"])):
        raise ValueError("Les périodes train < validation < test doivent être strictement ordonnées")
    return train, valid, test
