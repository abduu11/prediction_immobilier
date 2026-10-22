"""Géographie : communes (geo.api.gouv.fr), projection en km, distances au centre et au littoral.

Source des communes : https://geo.api.gouv.fr/departements/{dep}/communes (Licence Ouverte, Etalab).
`population` est un instantané récent : c'est une variable de contexte, pas une information sur la vente.
Usage : python -m immo.geo   (télécharge data/raw/communes_{dep}.geojson, idempotent)
"""
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from immo.config import PROJECT_ROOT, load_config
from immo.download import download_file, load_manifest, sha256_file

KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 79.0          # à ~44,8° N
BORDEAUX = (44.8378, -0.5792)  # (lat, lon) hôtel de ville de Bordeaux (préfecture de la Gironde)

# Trait de côte atlantique de la Gironde (lat, lon), du nord au sud. Tracé APPROXIMATIF (±3 km),
# suffisant pour une variable de distance ; ne pas utiliser pour de la cartographie fine.
COAST = [(45.575, -1.060), (45.500, -1.135), (45.375, -1.155), (45.225, -1.170),
         (45.100, -1.190), (45.000, -1.200), (44.880, -1.220), (44.730, -1.250),
         (44.630, -1.255), (44.590, -1.225), (44.470, -1.255)]

API_URL = "https://geo.api.gouv.fr/departements/{dep}/communes"
API_PARAMS = {"format": "geojson", "geometry": "contour",
              "fields": "nom,code,population,surface,codesPostaux"}


def to_km(lat, lon, origin=BORDEAUX):
    """Projection équirectangulaire locale (km) autour de `origin`."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    return (lon - origin[1]) * KM_PER_DEG_LON, (lat - origin[0]) * KM_PER_DEG_LAT


def dist_bordeaux_km(lat, lon):
    x, y = to_km(lat, lon)
    return np.hypot(x, y)


def _point_segment_km(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def dist_littoral_km(lat, lon):
    """Distance (km) au trait de côte approximatif ; NaN si les coordonnées sont NaN."""
    px, py = to_km(lat, lon)
    ax, ay = to_km([p[0] for p in COAST], [p[1] for p in COAST])
    d = np.full(np.shape(px), np.inf)
    for i in range(len(COAST) - 1):
        d = np.minimum(d, _point_segment_km(px, py, ax[i], ay[i], ax[i + 1], ay[i + 1]))
    return d


# --- communes ---
def communes_path(cfg: dict | None = None) -> Path:
    d = (cfg or load_config())["data"]
    return PROJECT_ROOT / d["raw_dir"] / f"communes_{d['departement']}.geojson"


def download_communes(cfg: dict | None = None, force: bool = False) -> Path:
    import requests

    cfg = cfg or load_config()
    dep = str(cfg["data"]["departement"])
    path = communes_path(cfg)
    if force or not path.exists():
        url = API_URL.format(dep=dep)
        print(f"[get ] {url}")
        r = requests.get(url, params=API_PARAMS, timeout=60)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(r.json(), ensure_ascii=False), encoding="utf-8")
        entry = {"url": r.url, "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    else:
        print(f"[skip] {path.name} déjà présent")
        entry = {}
    manifest_path = PROJECT_ROOT / cfg["data"]["manifest"]
    manifest = load_manifest(manifest_path)
    e = manifest.setdefault(path.name, {})
    e.update(entry)
    e["size_bytes"] = path.stat().st_size
    e["sha256"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


@lru_cache(maxsize=2)
def _load_geojson(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_geojson(cfg: dict | None = None) -> dict:
    path = communes_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} absent : lancer `python -m immo.geo`")
    return _load_geojson(str(path))


def load_communes(cfg: dict | None = None) -> pd.DataFrame:
    """Une ligne par commune : code, nom, population, surface_km2, densité, log_population, log_densite."""
    rows = [f["properties"] for f in load_geojson(cfg)["features"]]
    df = pd.DataFrame(rows).rename(columns={"code": "code_commune", "nom": "nom_commune_geo"})
    df["surface_km2"] = df["surface"] / 100.0                    # l'API donne des hectares
    df["densite"] = df["population"] / df["surface_km2"]
    df["log_population"] = np.log1p(df["population"])
    df["log_densite"] = np.log1p(df["densite"])
    return df.set_index("code_commune")[["nom_commune_geo", "population", "surface_km2", "densite",
                                          "log_population", "log_densite"]]


if __name__ == "__main__":
    print(download_communes())
