"""Application Streamlit : estimation du prix au m² en Gironde, avec fourchette (v1).

Lancer : streamlit run app/streamlit_app.py
L'application ne lit que app/artifacts/ (préparé par `python -m immo.export`).
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from matplotlib import colors

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT.parent / "src"))

from immo.maps import choropleth_deck  # noqa: E402
from immo.predict import TYPES, estimate  # noqa: E402

GITHUB = "https://github.com/abduu11/prediction_immobilier"
DEFAULT_COMMUNE = "33063"   # Bordeaux

st.set_page_config(page_title="Prix au m² en Gironde", page_icon=":material/home:", layout="wide")


# --- chargement (mis en cache) ---
@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS / "model.joblib")


@st.cache_data
def load_tables():
    read = lambda name, **kw: pd.read_csv(ARTIFACTS / name, dtype={"code_commune": str}, **kw)
    communes = read("communes.csv").set_index("code_commune")
    return {
        "communes": communes,
        "prix_type": read("prix_communes_type.csv"),
        "prix": read("prix_communes.csv").set_index("code_commune"),
        "erreurs": read("erreurs_communes.csv").set_index("code_commune"),
        "meta": json.loads((ARTIFACTS / "meta.json").read_text(encoding="utf-8")),
        "factors": json.loads((ARTIFACTS / "intervalles.json").read_text(encoding="utf-8")),
    }


@st.cache_data
def load_geojson():
    return json.loads((ARTIFACTS / "communes.geojson").read_text(encoding="utf-8"))


# --- mise en forme (espaces insécables à la française) ---
def num(x: float, step: int = 1) -> str:
    return f"{round(x / step) * step:,.0f}".replace(",", " ")


def eur_m2(x: float) -> str:
    return f"{num(x, 50)} €/m²"


def eur(x: float) -> str:
    return f"{num(x, 5000)} €"


def pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f} %".replace(".", ",")


def pct_value(x: float, digits: int = 1) -> str:
    """Pourcentage déjà exprimé en % (13.3 -> « 13,3 % »)."""
    return f"{x:.{digits}f} %".replace(".", ",")


def range_bar(low: float, central: float, high: float) -> str:
    pos = 50 if high <= low else min(max((central - low) / (high - low), 0), 1) * 100
    return f"""
<div style="position:relative;height:14px;border-radius:7px;margin:18px 0 6px;
  background:linear-gradient(90deg,rgba(46,134,193,.22),rgba(46,134,193,.65),rgba(46,134,193,.22))">
  <div style="position:absolute;left:{pos:.1f}%;top:-5px;width:4px;height:24px;background:#e67e22;border-radius:2px"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:.85rem;opacity:.8">
  <span>{eur_m2(low)}</span><span>{eur_m2(high)}</span></div>"""


# --- onglet 1 : estimer ---
def tab_estimate(t: dict, model) -> None:
    meta, communes = t["meta"], t["communes"]
    bounds = meta["bornes_saisie"]
    labels = {c: f"{r['nom']} ({r['code_postal']})" for c, r in communes.sort_values("nom").iterrows()}
    ev = meta["evaluation_test_2025"]

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("Le bien")
        code = st.selectbox("Commune", list(labels), index=list(labels).index(DEFAULT_COMMUNE),
                            format_func=labels.get, help="Tapez pour rechercher une commune de la Gironde.")
        type_local = st.radio("Type de bien", TYPES, horizontal=True)
        lo_s, hi_s = bounds["surface"][type_local]
        surface = st.number_input("Surface habitable (m²)", min_value=float(lo_s), max_value=float(hi_s),
                                  value=60.0 if type_local == "Appartement" else 110.0, step=1.0)
        pieces = st.number_input("Nombre de pièces principales", min_value=bounds["pieces"][0],
                                 max_value=bounds["pieces"][1], value=3 if type_local == "Appartement" else 5)
        terrain = None
        if type_local == "Maison":
            terrain = st.number_input("Surface du terrain (m²)", min_value=0.0,
                                      max_value=float(bounds["terrain_maison_max"]), value=500.0, step=50.0)
        dependance = st.checkbox("Avec dépendance (cave, garage, annexe…)", value=False)

    res = estimate(model, communes, t["factors"], code, type_local, surface, int(pieces), terrain,
                   dependance, pd.Timestamp(meta["date_reference"]))
    with right:
        st.subheader("Estimation")
        st.markdown(f"### {eur_m2(res['prix_m2_bas'])} – {eur_m2(res['prix_m2_haut'])}")
        st.caption(f"Estimation centrale : **{eur_m2(res['prix_m2'])}**")
        st.markdown(range_bar(res["prix_m2_bas"], res["prix_m2"], res["prix_m2_haut"]), unsafe_allow_html=True)
        st.info(f"Pour {num(surface)} m² : environ **{eur(res['total_bas'])} à {eur(res['total_haut'])}** "
                f"(central : {eur(res['prix_total'])}).")

        cover = ev["couverture_fourchette"].get(type_local)
        st.caption(
            f"La fourchette est calibrée pour contenir environ 80 % des ventes réelles : sur les ventes "
            f"de 2025 jamais vues par le modèle, {pct(cover) if cover else 'n.d.'} des "
            f"{'appartements' if type_local == 'Appartement' else 'maisons'} y sont tombés. Une vente sur cinq "
            f"sort donc de la fourchette.")

        info = communes.loc[code]
        if info["n_ventes"] == 0:
            st.warning("Aucune vente connue dans cette commune : l'estimation repose sur les communes voisines "
                       "et reste très approximative.")
        elif info["n_ventes"] < 10:
            st.warning(f"Seulement {int(info['n_ventes'])} ventes connues dans cette commune : "
                       "l'estimation est moins fiable.")
        ref = t["prix_type"]
        ref = ref[(ref["code_commune"] == code) & (ref["type_local"] == type_local)]
        if len(ref) and pd.notna(ref.iloc[0]["prix_median"]):
            st.markdown(f"**Repère dans la commune :** prix médian observé 2021-2025 pour ce type de bien : "
                        f"{eur_m2(ref.iloc[0]['prix_median'])} ({num(ref.iloc[0]['n'])} ventes). "
                        f"Population : {num(info['population'])} habitants.")
        else:
            st.markdown(f"Population : {num(info['population'])} habitants. Pas assez de ventes de ce type "
                        "pour afficher un prix médian communal.")

    st.divider()
    st.caption(f"Estimation indicative au niveau de marché de la dernière vente connue "
               f"({pd.Timestamp(meta['date_reference']).strftime('%d/%m/%Y')}), située au centre de la commune "
               "(aucune adresse n'est utilisée). Elle ne remplace pas l'avis d'un professionnel.")


# --- onglet 2 : cartes ---
def tab_maps(t: dict) -> None:
    choice = st.radio("Carte", ["Prix médian au m² (2021-2025)", "Biais du modèle (test 2025)",
                                "Erreur relative médiane du modèle (test 2025)"], horizontal=True)
    geojson = load_geojson()
    if choice.startswith("Prix"):
        df = t["prix"]
        v = df["prix_median"]
        deck = choropleth_deck(geojson, v, "YlOrRd", colors.Normalize(v.quantile(0.05), v.quantile(0.95), clip=True),
                               "Prix médian €/m²", df)
        st.caption(f"Couleur : du jaune ({num(v.quantile(0.05), 50)} €/m²) au rouge foncé "
                   f"({num(v.quantile(0.95), 50)} €/m² et plus). Gris : moins de 5 ventes.")
    elif choice.startswith("Biais"):
        df = t["erreurs"]
        deck = choropleth_deck(geojson, df["biais_pct"], "RdBu_r", colors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30),
                               "Biais médian %", df)
        st.caption("Rouge : le modèle surestime les prix ; bleu : il les sous-estime (échelle de −30 % à +30 %). "
                   "Gris : moins de 10 ventes de test dans la commune.")
    else:
        df = t["erreurs"]
        deck = choropleth_deck(geojson, df["ape_median"], "viridis_r", colors.Normalize(5, 35, clip=True),
                               "Erreur relative médiane %", df)
        st.caption("Jaune : erreur faible ; violet : erreur élevée (échelle de 5 % à 35 %). "
                   "Gris : moins de 10 ventes de test dans la commune.")
    st.pydeck_chart(deck, width="stretch")
    st.caption("Passez la souris sur une commune pour voir sa valeur. Les erreurs sont celles du modèle ajusté sur "
               "2021-2024 et évalué sur les ventes de 2025.")


# --- onglet 3 : méthodologie ---
def tab_method(t: dict) -> None:
    meta, ev = t["meta"], t["meta"]["evaluation_test_2025"]
    y0, y1 = meta["annees_entrainement"]
    rep = ev["repartition_erreurs"]["all"]
    gain = 1 - ev["mae_eur_m2"]["all"] / ev["mae_baseline_eur_m2"]
    st.markdown(f"""
### Version 1 : ce que fait l'outil, et ce qu'il ne fait pas
Cet outil estime le **prix au m²** d'un appartement ou d'une maison en Gironde à partir des ventes réelles
publiées par l'administration fiscale (**DVF**, {y0}-{y1}, {num(meta['n_ventes_entrainement'])} ventes utilisées
pour l'ajustement, après nettoyage et retrait des prix au m² extrêmes).
C'est un projet de portfolio : **une première version (v1)**, pas un outil d'expertise.

**Le modèle** est un `HistGradientBoostingRegressor` (scikit-learn) enrichi de variables géographiques
(distance à Bordeaux et au littoral, population et densité de la commune, prix des ventes voisines antérieures).
Le modèle en ligne est ajusté sur {y0}-{y1}. Les chiffres de précision ci-dessous ont été mesurés, une seule fois,
sur un modèle ajusté sur 2021-2024 et évalué sur **{num(ev['n_ventes_test'])} ventes de 2025** qu'il n'avait jamais vues ;
le modèle en ligne applique la même recette avec une année de données en plus (sa précision exacte n'est pas mesurée).

### Précision mesurée (test 2025)
| | Ensemble | Appartements | Maisons |
|---|---:|---:|---:|
| Erreur moyenne (MAE) | {num(ev['mae_eur_m2']['all'])} €/m² | {num(ev['mae_eur_m2']['Appartement'])} €/m² | {num(ev['mae_eur_m2']['Maison'])} €/m² |
| Erreur relative médiane | {pct_value(ev['mape_mediane_pct']['all'])} | {pct_value(ev['mape_mediane_pct']['Appartement'])} | {pct_value(ev['mape_mediane_pct']['Maison'])} |
| Couverture de la fourchette (cible 80 %) | {pct(ev['couverture_fourchette']['all'], 1)} | {pct(ev['couverture_fourchette']['Appartement'], 1)} | {pct(ev['couverture_fourchette']['Maison'], 1)} |
| Largeur relative médiane de la fourchette | {pct(ev['largeur_relative_mediane']['all'])} | {pct(ev['largeur_relative_mediane']['Appartement'])} | {pct(ev['largeur_relative_mediane']['Maison'])} |

Pour situer : prédire simplement la médiane par commune donne une erreur moyenne de {num(ev['mae_baseline_eur_m2'])} €/m²,
le modèle la réduit d'environ {pct(gain)}. Sur les ventes de 2025, l'écart entre le prix prédit et le prix réel est inférieur à
10 % pour {pct(rep['moins_de_10_pct'])} des ventes, à 20 % pour {pct(rep['moins_de_20_pct'])}, à 30 % pour
{pct(rep['moins_de_30_pct'])}, et supérieur à 50 % pour {pct(1 - rep['moins_de_50_pct'])}.

### Comment lire la fourchette
Elle est calculée à partir des erreurs réellement observées (rapport prix réel / prix prédit, par type de bien) pour
contenir environ 80 % des ventes. **Elle est large, notamment pour les maisons : c'est volontaire et honnête.**
Une vente sur cinq sort de la fourchette.

### Limites connues
- **Variables absentes des données publiques :** état du bien, étage, vue, exposition, DPE, qualité de la rue.
  Deux biens identiques dans la même rue peuvent différer de 30 % ; c'est la principale limite de précision.
- **Localisation à la commune :** le bien est placé au centre de la commune ; dans les grandes communes
  (Bordeaux, Mérignac…), le quartier n'est pas pris en compte.
- **Maisons moins bien prédites** que les appartements (le terrain, l'état et la situation comptent beaucoup).
- **Communes rurales :** peu de ventes, donc erreurs plus fortes et cartes d'erreurs grisées.
- **Marché :** les prix de 2025 sont plus bas qu'en 2022 ; un retournement brusque ne serait pas anticipé.
- **Littoral :** la distance à la côte vient d'un tracé approximatif (±3 km).
- Les ventes exclues du modèle : ventes sur plan, transactions regroupant plusieurs logements, locaux commerciaux.

### Données et transparence
Source : DVF (DGFiP), géolocalisées par Etalab, Licence Ouverte 2.0 ; contours et populations : geo.api.gouv.fr.
Le fichier du modèle contient les coordonnées et prix (en logarithme) des ventes d'ajustement, nécessaires à la
variable « prix des voisins » ; ce sont des données publiques et l'application n'affiche aucune adresse.
Code, tests et protocole complet : [{GITHUB}]({GITHUB}).
""")


def main() -> None:
    st.title(":material/home: Prix au m² en Gironde")
    st.caption("Version 1 · estimation statistique à partir des ventes réelles DVF 2021-2025 · "
               "l'estimation est donnée avec une fourchette")
    if not (ARTIFACTS / "model.joblib").exists():
        st.error("Artefacts introuvables : lancer `python -m immo.export`.")
        st.stop()
    t, model = load_tables(), load_model()
    tab1, tab2, tab3 = st.tabs(["Estimer un bien", "Cartes", "Méthodologie et limites"])
    with tab1:
        tab_estimate(t, model)
    with tab2:
        tab_maps(t)
    with tab3:
        tab_method(t)


main()
