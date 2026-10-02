import json
from pathlib import Path

import pytest

from immo.config import PROJECT_ROOT, load_config
from immo.download import build_url, raw_path, sha256_file


def test_build_url():
    url = build_url("https://files.data.gouv.fr/geo-dvf/latest/csv", 2024, "33")
    assert url == "https://files.data.gouv.fr/geo-dvf/latest/csv/2024/departements/33.csv.gz"


def test_raw_path():
    assert raw_path(Path("data/raw"), 2023, "33") == Path("data/raw/dvf_2023_33.csv.gz")


def test_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_config_years_in_covered_range():
    cfg = load_config()["data"]
    assert all(2021 <= y <= 2025 for y in cfg["years"])
    assert str(cfg["departement"]) not in {"57", "67", "68", "976"}  # hors périmètre DVF


@pytest.mark.network
def test_manifest_matches_files():
    cfg = load_config()["data"]
    manifest = json.loads((PROJECT_ROOT / cfg["manifest"]).read_text(encoding="utf-8"))
    for year in cfg["years"]:
        path = raw_path(PROJECT_ROOT / cfg["raw_dir"], year, str(cfg["departement"]))
        entry = manifest[path.name]
        assert entry["sha256"] == sha256_file(path)
        assert entry["size_bytes"] == path.stat().st_size
