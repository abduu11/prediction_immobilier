"""Chargement de la configuration du projet (config.yaml)."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
