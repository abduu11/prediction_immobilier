"""Prépare les artefacts légers de l'application dans app/artifacts/.

    python -m immo.export

Le modèle déployé est ré-ajusté sur TOUTES les ventes nettoyées (2021-2025), avec les hyperparamètres de
reports/best_params.json et les bornes P1-P99 estimées sur ces mêmes données. Les métriques publiées
(reports/metrics_test.json) décrivent le modèle ajusté sur 2021-2024 ; celui-ci applique la même recette
avec une année de données en plus (sa précision exacte n'est donc pas mesurée).
Le fichier du modèle embarque les coordonnées et prix (log) des ventes d'ajustement, nécessaires à la variable
« prix des ventes voisines ». Ce sont des données publiques DVF ; l'application n'affiche aucune adresse.

L'application ne lit que app/artifacts/ : elle ne dépend ni de data/ ni de reports/ (non versionnés).
"""
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from immo import evaluate
from immo.config import PROJECT_ROOT, load_config
from immo.features import add_features
from immo.geo import load_communes, load_geojson
from immo.maps import price_table
from immo.models import build_hgb
from immo.predict import LOTS_DEFAUT
from immo.split import load_clean

ARTIFACTS = PROJECT_ROOT / "app" / "artifacts"
COORD_DECIMALS = 4          # ~11 m : suffisant pour les cartes, divise la taille du fichier


def ring_centroid(ring: np.ndarray) -> tuple[float, float]:
    """Centroïde (lon, lat) d'un anneau polygonal (formule de l'aire signée)."""
    x, y = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2
    if abs(a) < 1e-12:
        return float(x.mean()), float(y.mean())
    return float(((x + x1) * cross).sum() / (6 * a)), float(((y + y1) * cross).sum() / (6 * a))


def simplify_geojson(geojson: dict) -> dict:
    """Arrondit les coordonnées, retire les points consécutifs identiques et les propriétés inutiles."""
    def clean_ring(ring):
        pts = np.round(np.asarray(ring, float), COORD_DECIMALS)
        keep = np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]
        return pts[keep].tolist()

    feats = []
    for f in geojson["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        polys = [[clean_ring(r) for r in p] for p in polys]
        geom = ({"type": "Polygon", "coordinates": polys[0]} if g["type"] == "Polygon"
                else {"type": "MultiPolygon", "coordinates": polys})
        p = f["properties"]
        feats.append({"type": "Feature", "geometry": geom,
                      "properties": {"code": p["code"], "nom": p["nom"], "population": p.get("population")}})
    return {"type": "FeatureCollection", "features": feats}


def commune_table(clean: pd.DataFrame, geojson: dict, cfg: dict) -> pd.DataFrame:
    """Une ligne par commune de l'API : nom, code postal, population, point représentatif, nb de ventes."""
    table = load_communes(cfg).copy()
    props = {f["properties"]["code"]: f for f in geojson["features"]}
    table["nom"] = [props[c]["properties"]["nom"] for c in table.index]
    table["code_postal"] = [(props[c]["properties"].get("codesPostaux") or [""])[0] for c in table.index]
    sales = clean.groupby("code_commune").agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"),
                                              n_ventes=("prix_m2", "size"))
    table = table.join(sales)
    table["n_ventes"] = table["n_ventes"].fillna(0).astype(int)
    geom_c = {c: ring_centroid(np.asarray((f["geometry"]["coordinates"][0] if f["geometry"]["type"] == "Polygon"
                                           else max((p[0] for p in f["geometry"]["coordinates"]), key=len))))
              for c, f in props.items()}
    for code, (lon, lat) in geom_c.items():            # communes sans vente : centroïde du contour
        if code in table.index and pd.isna(table.loc[code, "latitude"]):
            table.loc[code, ["latitude", "longitude"]] = lat, lon
    return table[["nom", "code_postal", "population", "surface_km2", "densite", "log_population",
                  "log_densite", "latitude", "longitude", "n_ventes"]]


def run(config: dict | None = None) -> None:
    cfg = config or load_config()
    rep = PROJECT_ROOT / cfg["clean"]["report_dir"]
    params = evaluate.load_params(cfg)
    metrics = json.loads((rep / "metrics_test.json").read_text(encoding="utf-8"))
    factors = json.loads((rep / "intervalles.json").read_text(encoding="utf-8"))
    name = cfg["final"]["model"]

    clean = load_clean(cfg)
    geojson = load_geojson(cfg)
    communes = load_communes(cfg)
    fit_trimmed, bounds = evaluate.prepare_fit_set(clean, cfg)
    model = build_hgb(geo=True, **params[name]).fit(add_features(fit_trimmed, communes), fit_trimmed["prix_m2"])

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS / "model.joblib", compress=3)
    table = commune_table(clean, geojson, cfg)
    table.to_csv(ARTIFACTS / "communes.csv", index_label="code_commune", encoding="utf-8")
    types = (clean.groupby(["code_commune", "type_local"])["prix_m2"].agg(prix_median="median", n="size")
             .reset_index())
    types.loc[types["n"] < 5, "prix_median"] = float("nan")
    types.to_csv(ARTIFACTS / "prix_communes_type.csv", index=False, encoding="utf-8")
    price_table(clean).to_csv(ARTIFACTS / "prix_communes.csv", index_label="code_commune", encoding="utf-8")
    errors_path = rep / "erreurs_par_commune.csv"
    if errors_path.exists():      # produit par `python -m immo.maps`
        pd.read_csv(errors_path, dtype={"code_commune": str}).to_csv(
            ARTIFACTS / "erreurs_communes.csv", index=False, encoding="utf-8")
    (ARTIFACTS / "communes.geojson").write_text(
        json.dumps(simplify_geojson(geojson), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (ARTIFACTS / "intervalles.json").write_text(json.dumps(factors, indent=2), encoding="utf-8")

    hgb = metrics[name]["metrics"]
    meta = {
        "modele": name,
        "annees_entrainement": [int(clean["date_mutation"].dt.year.min()), int(clean["date_mutation"].dt.year.max())],
        "n_ventes_entrainement": int(len(fit_trimmed)),
        "date_reference": str(clean["date_mutation"].max().date()),
        "bornes_prix_m2_entrainement": {t: [round(lo), round(hi)] for t, (lo, hi) in bounds.items()},
        "lots_defaut": LOTS_DEFAUT,
        "bornes_saisie": {"surface": cfg["clean"]["surface_bounds"], "pieces": cfg["clean"]["pieces_bounds"],
                          "terrain_maison_max": cfg["clean"]["max_surface_terrain_maison"]},
        "evaluation_test_2025": {
            "note": "modèle ajusté sur 2021-2024, évalué une seule fois sur 2025",
            "mae_eur_m2": {k: round(v["mae"]) for k, v in hgb.items()},
            "mape_mediane_pct": {k: round(v["median_ape"], 1) for k, v in hgb.items()},
            "mape_pct": {k: round(v["mape"], 1) for k, v in hgb.items()},
            "mae_baseline_eur_m2": round(metrics["baseline_mediane_commune"]["metrics"]["all"]["mae"]),
            "couverture_fourchette": {k: round(v["couverture"], 3) for k, v in
                                      metrics["fourchette"]["mesure_sur_test"].items() if isinstance(v, dict)},
            "largeur_relative_mediane": {k: round(v["largeur_relative_mediane"], 2) for k, v in
                                         metrics["fourchette"]["mesure_sur_test"].items() if isinstance(v, dict)},
            "repartition_erreurs": metrics["repartition_erreurs_test"],
            "n_ventes_test": hgb["all"]["n"],
        },
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    for p in sorted(ARTIFACTS.iterdir()):
        print(f"  {p.name:26s} {p.stat().st_size / 1e6:6.2f} Mo")


if __name__ == "__main__":
    run()
