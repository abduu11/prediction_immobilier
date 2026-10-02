"""Téléchargement idempotent des fichiers DVF géolocalisés (Etalab).

Source : https://files.data.gouv.fr/geo-dvf/latest/csv/{année}/departements/{dep}.csv.gz
Un manifeste (date, taille, SHA-256) est écrit pour figer la version utilisée :
les fichiers sont remplacés à chaque mise à jour semestrielle (avril / octobre).

Usage : python -m immo.download
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from immo.config import PROJECT_ROOT, load_config

CHUNK = 1024 * 1024


def build_url(base_url: str, year: int, dep: str) -> str:
    return f"{base_url}/{year}/departements/{dep}.csv.gz"


def raw_path(raw_dir: Path, year: int, dep: str) -> Path:
    return raw_dir / f"dvf_{year}_{dep}.csv.gz"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path, timeout: int = 60) -> None:
    """Télécharge `url` vers `dest` (écriture atomique via fichier .part)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(CHUNK):
                f.write(block)
    tmp.replace(dest)


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def run(config: dict | None = None, force: bool = False) -> dict:
    cfg = (config or load_config())["data"]
    raw_dir = PROJECT_ROOT / cfg["raw_dir"]
    manifest_path = PROJECT_ROOT / cfg["manifest"]
    dep = str(cfg["departement"])
    manifest = load_manifest(manifest_path)

    for year in cfg["years"]:
        dest = raw_path(raw_dir, year, dep)
        url = build_url(cfg["base_url"], year, dep)
        if dest.exists() and not force:
            print(f"[skip] {dest.name} déjà présent")
        else:
            print(f"[get ] {url}")
            download_file(url, dest)
            manifest[dest.name] = {
                "url": url,
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        entry = manifest.setdefault(dest.name, {"url": url})
        entry["size_bytes"] = dest.stat().st_size
        entry["sha256"] = sha256_file(dest)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifeste écrit : {manifest_path}")
    return manifest


if __name__ == "__main__":
    run()
