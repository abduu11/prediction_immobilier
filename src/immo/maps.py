"""Cartes : prix médian par commune et erreurs du modèle.

    python -m immo.maps

Produit reports/figures/carte_prix.png et carte_erreurs.png (versionnées) et des cartes interactives
pydeck dans reports/maps/*.html (non versionnées ; l'application Streamlit réutilise `choropleth_deck`).
Les erreurs sont celles du modèle `hgb_geo` sur la VALIDATION 2024 (le test 2025 reste réservé).
Les communes avec trop peu de ventes sont grisées (statistique non fiable).
"""
import copy
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.collections import PolyCollection

from immo import evaluate
from immo.config import PROJECT_ROOT, load_config
from immo.geo import load_communes, load_geojson
from immo.split import load_clean, temporal_split

GREY = (0.82, 0.82, 0.82, 1.0)
MIN_VENTES_PRIX = 5
MIN_VENTES_ERREUR = 10


# --- tables par commune ---
def price_table(df: pd.DataFrame, min_n: int = MIN_VENTES_PRIX) -> pd.DataFrame:
    """Prix/m² médian par commune (tous biens), nombre de ventes ; NaN si moins de `min_n` ventes."""
    g = df.groupby("code_commune")["prix_m2"].agg(prix_median="median", n="size")
    g.loc[g["n"] < min_n, "prix_median"] = np.nan
    return g


def error_table(eval_df: pd.DataFrame, pred, min_n: int = MIN_VENTES_ERREUR) -> pd.DataFrame:
    """Erreurs par commune : MAE (€/m²), erreur relative absolue médiane (%), biais relatif médian (%)."""
    d = pd.DataFrame({"code_commune": eval_df["code_commune"].to_numpy(),
                      "err": np.asarray(pred, float) - eval_df["prix_m2"].to_numpy(),
                      "rel": (np.asarray(pred, float) - eval_df["prix_m2"].to_numpy()) / eval_df["prix_m2"].to_numpy()})
    g = d.groupby("code_commune").agg(
        n=("err", "size"), mae=("err", lambda s: s.abs().mean()),
        ape_median=("rel", lambda s: s.abs().median() * 100), biais_pct=("rel", lambda s: s.median() * 100))
    g.loc[g["n"] < min_n, ["mae", "ape_median", "biais_pct"]] = np.nan
    return g


# --- géométrie ---
def rings(feature: dict):
    """Anneaux extérieurs (liste de tableaux (lon, lat)) d'un Polygon ou MultiPolygon."""
    geom = feature["geometry"]
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return [np.asarray(p[0]) for p in polys]


def colorize(values: pd.Series, cmap: str, norm) -> list:
    cm = plt.get_cmap(cmap)
    return [GREY if pd.isna(v) else cm(norm(v)) for v in values]


def draw_choropleth(ax, geojson: dict, values: pd.Series, cmap: str, norm, title: str):
    polys, cols = [], []
    colors_ = dict(zip(values.index, colorize(values, cmap, norm)))
    for f in geojson["features"]:
        c = colors_.get(f["properties"]["code"], GREY)
        for r in rings(f):
            polys.append(r)
            cols.append(c)
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors="white", linewidths=0.15))
    allpts = np.vstack(polys)
    ax.set_xlim(allpts[:, 0].min(), allpts[:, 0].max())
    ax.set_ylim(allpts[:, 1].min(), allpts[:, 1].max())
    ax.set_aspect(1 / np.cos(np.radians(44.8)))
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    return sm


def price_map(geojson: dict, table: pd.DataFrame, path) -> None:
    v = table["prix_median"]
    norm = colors.Normalize(vmin=v.quantile(0.05), vmax=v.quantile(0.95), clip=True)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=130)
    sm = draw_choropleth(ax, geojson, v, "YlOrRd", norm, "Prix médian au m² par commune (Gironde, 2021-2025)")
    fig.colorbar(sm, ax=ax, shrink=0.6, label="€/m²")
    fig.text(0.5, 0.04, "Gris : moins de 5 ventes. Source : DVF (DGFiP / Etalab), après nettoyage.",
             ha="center", fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def error_map(geojson: dict, table: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), dpi=130)
    bias_norm = colors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
    sm = draw_choropleth(axes[0], geojson, table["biais_pct"], "RdBu_r", bias_norm,
                         "Biais médian (prédit − réel) / réel")
    fig.colorbar(sm, ax=axes[0], shrink=0.6, label="%")
    ape_norm = colors.Normalize(vmin=5, vmax=35, clip=True)
    sm = draw_choropleth(axes[1], geojson, table["ape_median"], "viridis_r", ape_norm,
                         "Erreur relative absolue médiane")
    fig.colorbar(sm, ax=axes[1], shrink=0.6, label="%")
    fig.suptitle("Erreurs du modèle HGB (variables géographiques) sur la validation 2024", fontsize=12)
    fig.text(0.5, 0.04, "Biais : rouge = surestimation, bleu = sous-estimation. "
             "Gris : moins de 10 ventes de validation dans la commune.", ha="center", fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --- cartes interactives (pydeck) ---
def choropleth_deck(geojson: dict, values: pd.Series, cmap: str, norm, tooltip_label: str,
                    extra: pd.DataFrame | None = None):
    """Deck pydeck : communes colorées par `values` (index = code INSEE)."""
    import pydeck as pdk

    gj = copy.deepcopy(geojson)
    rgba = dict(zip(values.index, colorize(values, cmap, norm)))
    for f in gj["features"]:
        code = f["properties"]["code"]
        f["properties"]["color"] = [int(255 * c) for c in rgba.get(code, GREY)[:3]] + [190]
        v = values.get(code, np.nan)
        f["properties"]["valeur"] = "n.d." if pd.isna(v) else f"{v:,.0f}".replace(",", " ")
        if extra is not None and code in extra.index:
            f["properties"]["n_ventes"] = int(extra.loc[code, "n"])
    layer = pdk.Layer("GeoJsonLayer", gj, pickable=True, stroked=True, filled=True,
                      get_fill_color="properties.color", get_line_color=[255, 255, 255, 120],
                      line_width_min_pixels=0.3)
    view = pdk.ViewState(latitude=44.85, longitude=-0.55, zoom=8.2)
    tip = {"html": f"<b>{{nom}}</b><br/>{tooltip_label} : {{valeur}}<br/>Ventes : {{n_ventes}}"}
    return pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tip, map_style="light")


def run(config: dict | None = None) -> None:
    cfg = config or load_config()
    geojson = load_geojson(cfg)
    clean = load_clean(cfg)
    _, valid, _ = temporal_split(clean, cfg)

    fig_dir = PROJECT_ROOT / cfg["clean"]["report_dir"] / "figures"
    html_dir = PROJECT_ROOT / cfg["clean"]["report_dir"] / "maps"
    fig_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    prices = price_table(clean)
    price_map(geojson, prices, fig_dir / "carte_prix.png")

    model_path = PROJECT_ROOT / "models" / "hgb_geo_validation.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} absent : lancer `python -m immo.train`")
    model = joblib.load(model_path)
    pred = model.predict(evaluate.model_inputs(valid, True, load_communes(cfg)))
    errors = error_table(valid, pred)
    errors.to_csv(PROJECT_ROOT / cfg["clean"]["report_dir"] / "erreurs_par_commune.csv")
    error_map(geojson, errors, fig_dir / "carte_erreurs.png")

    p = prices["prix_median"]
    choropleth_deck(geojson, p, "YlOrRd", colors.Normalize(p.quantile(0.05), p.quantile(0.95), clip=True),
                    "Prix médian €/m²", prices).to_html(str(html_dir / "carte_prix.html"), open_browser=False)
    b = errors["biais_pct"]
    choropleth_deck(geojson, b, "RdBu_r", colors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30),
                    "Biais médian %", errors).to_html(str(html_dir / "carte_biais.html"), open_browser=False)
    print(f"Cartes -> {fig_dir} et {html_dir}")
    print(f"Communes avec erreur calculée : {int(errors['mae'].notna().sum())} / {len(errors)}")


if __name__ == "__main__":
    run()
