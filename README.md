# Prédiction du prix au m² — DVF (Gironde)

Projet data : prédire le prix au m² des appartements et maisons à partir des
[Demandes de valeurs foncières](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres-geolocalisees/)
(DGFiP / Etalab, Licence Ouverte 2.0).

**Stack** : Python, pandas, scikit-learn, Streamlit. **MVP** : Gironde (33), 2021–2025.

> Statut : sprints 1 à 3 terminés (données, nettoyage, validation + baseline). Voir l'avancement ci-dessous.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt
```

## Reproduire les étapes

```bash
set PYTHONPATH=src              # PowerShell : $env:PYTHONPATH="src"
python -m immo.download         # télécharge data/raw/*.csv.gz + data/manifest.json
pytest                          # tests unitaires + tests sur les vrais fichiers
```

La configuration (département, années) est dans `config.yaml`.

## Structure

```
config.yaml          paramètres (département, années, chemins)
src/immo/            download.py, loader.py, clean.py, split.py, baseline.py, metrics.py, evaluate.py, config.py
tests/               pytest
notebooks/           01_exploration.ipynb
reports/             funnel.csv (entonnoir), metrics.json (validation)
data/                raw/ (gitignoré), manifest.json (versions + SHA-256), README.md
```

## Données

Voir [data/README.md](data/README.md) : source, schéma, périmètre, version figée.

## Nettoyage (sprint 2)

```bash
python -m immo.clean    # -> data/processed/ventes_clean.parquet + reports/funnel.csv
```

Règles : ventes uniquement ; une seule disposition ; aucun local commercial ; **un seul
logement par mutation** (les lignes répétées d'un même bien, une par nature de culture, sont
fusionnées ; les mutations multi-logements sont écartées car le prix n'est pas ventilable) ;
dépendances tolérées (`has_dependance`) ; surface de terrain = somme des morceaux de terrain
distincts ; bornes de surface, pièces, prix minimal et garde-fous prix/m² (`config.yaml`).
Les bornes par quantiles (P1–P99) sont **estimées sur le train uniquement**
(`fit_price_bounds` / `apply_price_bounds`, utilisées au sprint 3).

Entonnoir (Gironde 2021–2025, en mutations) :

| Étape | Mutations restantes | Perdues |
|---|---:|---:|
| Lignes brutes | 187 909 | — |
| Nature = Vente | 171 718 | 16 191 |
| Une seule disposition | 171 573 | 145 |
| Sans local industriel/commercial | 162 903 | 8 670 |
| Au moins un appartement ou une maison | 122 615 | 40 288 |
| Exactement un local principal | 117 275 | 5 340 |
| Prix renseigné | 117 213 | 62 |
| Surface bâtie dans les bornes | 117 139 | 74 |
| Nombre de pièces dans les bornes | 117 072 | 67 |
| Prix minimal (15 000 €) | 116 773 | 299 |
| Terrain maison ≤ 10 000 m² | 115 109 | 1 664 |
| Prix/m² entre 200 et 25 000 € | **114 887** | 222 |

Résultat : 46 221 appartements et 68 666 maisons, 1,7 % sans coordonnées (conservées avec
l'indicateur `coords_missing`, à imputer dans le pipeline de modélisation).

## Validation et baseline (sprint 3)

```bash
python -m immo.evaluate          # ajuste sur 2021-2023, évalue sur la validation 2024 -> reports/metrics.json
```

- **Découpage temporel** (`config.yaml`, section `split`) : train 2021–2023 (75 169 ventes),
  validation 2024 (17 793), **test 2025 (21 925) réservé à l'évaluation finale** (`--test`, une seule fois,
  avec ajustement sur train+validation).
- **Anti-fuite** : les bornes P1–P99 du prix/m² sont estimées sur les données d'ajustement et ne servent
  qu'à les filtrer ; les jeux d'évaluation ne sont jamais tronqués. La baseline ne lit jamais la cible
  du jeu évalué (testé).
- **Baseline** : médiane du prix/m² par (commune, type), avec repli sur la médiane du type si la commune a
  moins de 5 ventes dans le train (0,65 % des ventes de 2024).

Résultats de la baseline sur la validation 2024 :

| | n | MAE (€/m²) | MAPE | MAPE médiane | biais (€/m²) | R² |
|---|---:|---:|---:|---:|---:|---:|
| Ensemble | 17 793 | 899 | 33,0 % | 17,3 % | +64 | 0,46 |
| Appartements | 7 433 | 851 | 26,0 % | 15,5 % | +119 | 0,42 |
| Maisons | 10 360 | 933 | 38,1 % | 18,8 % | +25 | 0,47 |

Pour situer la difficulté : prédire la médiane globale du type de bien donne une MAE de 1 321 €/m²
(MAPE 52 %). La médiane par commune la réduit déjà de 32 %. La baseline ignore la surface, les
pièces, la date et les coordonnées ; les sprints suivants les ajoutent.
La MAPE (moyenne) est tirée vers le haut par les biens à faible prix/m² ; la MAPE médiane est plus robuste.

## Avancement

| Sprint | Contenu | État |
|---|---|---|
| S1 | Données : téléchargement, lecture typée, exploration | ✅ |
| S2 | Nettoyage (multi-lots, doublons, aberrants) | ✅ |
| S3 | Découpage temporel + baseline | ✅ |
| S4 | Ridge + HistGradientBoosting | à faire |
| S5 | Variables géographiques, validation spatiale, cartes | à faire |
| S6 | App Streamlit, déploiement | à faire |
