"""Nettoyage DVF : ventes simples d'un appartement ou d'une maison -> 1 ligne = 1 vente.

Règles (voir README) :
1. ventes uniquement (pas de VEFA, échange, adjudication, expropriation, terrain à bâtir) ;
2. une mutation = une seule disposition (sinon le prix n'est pas attribuable) ;
3. aucun local industriel/commercial dans la mutation ;
4. exactement UN local principal (appartement ou maison) : les lignes répétées d'un même
   local (une par nature de culture) sont fusionnées, les mutations à plusieurs logements
   sont écartées car le prix n'est pas ventilable ;
5. les dépendances sont tolérées (indicateur `has_dependance`) ;
6. surface de terrain = somme des morceaux de terrain DISTINCTS de la mutation ;
7. valeurs manquantes et aberrantes (bornes absolues de config.yaml).

Les bornes par quantiles (P1-P99 du prix/m²) dépendent des données : elles sont fournies
par `fit_price_bounds` / `apply_price_bounds` et doivent être ajustées sur le TRAIN uniquement.

Usage : python -m immo.clean
"""
import pandas as pd

from immo.config import PROJECT_ROOT, load_config
from immo.loader import load_raw

MAIN_TYPES = ["Appartement", "Maison"]
COMMERCIAL = "Local industriel. commercial ou assimilé"
DEPENDANCE = "Dépendance"
# Un même local apparaît sur plusieurs lignes (une par nature de culture) :
# ces colonnes l'identifient. `lot1_numero` distingue deux appartements identiques.
LOCAL_KEY = ["id_mutation", "type_local", "surface_reelle_bati",
             "nombre_pieces_principales", "lot1_numero"]
LAND_KEY = ["id_mutation", "id_parcelle", "code_nature_culture",
            "code_nature_culture_speciale", "surface_terrain"]

OUTPUT_COLUMNS = [
    "id_mutation", "date_mutation", "annee", "valeur_fonciere", "prix_m2",
    "type_local", "surface_reelle_bati", "nombre_pieces_principales",
    "surface_terrain", "has_dependance", "nombre_lots", "code_commune",
    "nom_commune", "code_postal", "latitude", "longitude", "coords_missing",
]


class Funnel:
    """Compte lignes et mutations restantes après chaque étape."""

    def __init__(self):
        self.rows = []

    def log(self, step: str, df: pd.DataFrame) -> None:
        self.rows.append({"etape": step, "lignes": len(df),
                          "mutations": df["id_mutation"].nunique()})

    def to_frame(self) -> pd.DataFrame:
        out = pd.DataFrame(self.rows)
        out["mutations_perdues"] = (-out["mutations"].diff()).fillna(0).astype(int)
        return out


def _keep_ids(df: pd.DataFrame, ids) -> pd.DataFrame:
    return df[df["id_mutation"].isin(ids)]


def clean_dvf(df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (ventes nettoyées, tableau en entonnoir)."""
    c = (cfg or load_config())["clean"]
    f = Funnel()
    f.log("Lignes brutes", df)

    df = df[df["nature_mutation"] == "Vente"]
    f.log("Nature = Vente", df)

    n_disp = df.groupby("id_mutation")["numero_disposition"].nunique()
    df = _keep_ids(df, n_disp.index[n_disp == 1])
    f.log("Une seule disposition", df)

    has_commercial = df.loc[df["type_local"] == COMMERCIAL, "id_mutation"].unique()
    df = df[~df["id_mutation"].isin(has_commercial)]
    f.log("Sans local industriel/commercial", df)

    main = df[df["type_local"].isin(MAIN_TYPES)]
    df = _keep_ids(df, main["id_mutation"].unique())
    f.log("Au moins un appartement ou une maison", df)

    locaux = main[main["id_mutation"].isin(df["id_mutation"])].drop_duplicates(LOCAL_KEY)
    n_locaux = locaux.groupby("id_mutation").size()
    unique_ids = n_locaux.index[n_locaux == 1]
    df = _keep_ids(df, unique_ids)
    locaux = locaux[locaux["id_mutation"].isin(unique_ids)]
    f.log("Exactement un local principal", df)

    # --- 1 ligne = 1 vente ---
    terrain = (df.dropna(subset=["surface_terrain"]).drop_duplicates(LAND_KEY)
               .groupby("id_mutation")["surface_terrain"].sum().rename("surface_terrain"))
    dep = df.loc[df["type_local"] == DEPENDANCE, "id_mutation"].unique()
    sales = locaux.drop(columns="surface_terrain").merge(
        terrain, left_on="id_mutation", right_index=True, how="left")
    sales["has_dependance"] = sales["id_mutation"].isin(dep)
    sales["type_local"] = sales["type_local"].astype("string")
    f.log("Une ligne par vente", sales)

    sales = sales[sales["valeur_fonciere"].notna() & (sales["valeur_fonciere"] > 0)]
    f.log("Prix renseigné", sales)

    surf = sales["surface_reelle_bati"]
    ok = pd.Series(False, index=sales.index)
    for t, (lo, hi) in c["surface_bounds"].items():
        ok |= (sales["type_local"] == t) & surf.between(lo, hi)
    sales = sales[ok]
    f.log("Surface bâtie dans les bornes", sales)

    lo, hi = c["pieces_bounds"]
    sales = sales[sales["nombre_pieces_principales"].between(lo, hi)]
    f.log("Nombre de pièces dans les bornes", sales)

    sales = sales[sales["valeur_fonciere"] >= c["min_valeur_fonciere"]]
    f.log("Prix minimal", sales)

    too_much_land = (sales["type_local"] == "Maison") & (
        sales["surface_terrain"] > c["max_surface_terrain_maison"])
    sales = sales[~too_much_land]
    f.log("Terrain maison plafonné", sales)

    sales = sales.assign(prix_m2=sales["valeur_fonciere"] / sales["surface_reelle_bati"])
    lo, hi = c["prix_m2_bounds"]
    sales = sales[sales["prix_m2"].between(lo, hi)]
    f.log("Prix/m² dans les garde-fous absolus", sales)

    sales = sales.assign(
        annee=sales["date_mutation"].dt.year,
        coords_missing=sales["latitude"].isna() | sales["longitude"].isna(),
    )
    sales["nombre_pieces_principales"] = sales["nombre_pieces_principales"].astype("int16")
    sales["nombre_lots"] = sales["nombre_lots"].astype("Int16")
    out = (sales[OUTPUT_COLUMNS].sort_values(["date_mutation", "id_mutation"])
           .reset_index(drop=True))
    return out, f.to_frame()


def fit_price_bounds(train: pd.DataFrame, quantiles=(0.01, 0.99)) -> dict:
    """Quantiles du prix/m² par type de bien, estimés sur `train` UNIQUEMENT."""
    lo, hi = quantiles
    g = train.groupby("type_local")["prix_m2"]
    return {t: (float(g.quantile(lo)[t]), float(g.quantile(hi)[t])) for t in g.groups}


def apply_price_bounds(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Filtre `df` avec des bornes déjà estimées (jamais recalculées sur `df`)."""
    lo = df["type_local"].map({t: b[0] for t, b in bounds.items()})
    hi = df["type_local"].map({t: b[1] for t, b in bounds.items()})
    return df[(df["prix_m2"] >= lo) & (df["prix_m2"] <= hi)]


def run(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    clean, funnel = clean_dvf(load_raw(cfg), cfg)

    out_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    rep_dir = PROJECT_ROOT / cfg["clean"]["report_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_dir / "ventes_clean.parquet", index=False)
    funnel.to_csv(rep_dir / "funnel.csv", index=False)
    print(funnel.to_string(index=False))
    print(f"\n{len(clean)} ventes -> {out_dir / 'ventes_clean.parquet'}")
    return clean


if __name__ == "__main__":
    run()
