# Prédiction du prix au m² — DVF (Gironde)

Projet data : prédire le prix au m² des appartements et maisons à partir des
[Demandes de valeurs foncières](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres-geolocalisees/)
(DGFiP / Etalab, Licence Ouverte 2.0).

**Stack** : Python, pandas, scikit-learn, Streamlit. **MVP** : Gironde (33), 2021–2025.

> Statut : Sprint 1 (données) terminé. Voir l'avancement ci-dessous.

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
src/immo/            download.py, loader.py, config.py
tests/               pytest
notebooks/           01_exploration.ipynb
data/                raw/ (gitignoré), manifest.json (versions + SHA-256), README.md
```

## Données

Voir [data/README.md](data/README.md) : source, schéma, périmètre, version figée.

## Avancement

| Sprint | Contenu | État |
|---|---|---|
| S1 | Données : téléchargement, lecture typée, exploration | ✅ |
| S2 | Nettoyage (multi-lots, doublons, aberrants) | à faire |
| S3 | Découpage temporel + baseline | à faire |
| S4 | Ridge + HistGradientBoosting | à faire |
| S5 | Variables géographiques, validation spatiale, cartes | à faire |
| S6 | App Streamlit, déploiement | à faire |
