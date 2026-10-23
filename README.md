# Prix au m² en Gironde : prédire à partir des ventes réelles (DVF)

Estimer le **prix au m²** d'un appartement ou d'une maison en Gironde à partir des
[Demandes de valeurs foncières](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres-geolocalisees/)
(DGFiP / Etalab, Licence Ouverte 2.0), avec une **fourchette de prix** calibrée sur les erreurs réellement observées.

**Statut : version 1 (v1).** Un projet de portfolio, pas un outil d'expertise : la précision est limitée par
les informations disponibles dans les données publiques (voir [Limites](#limites-et-pistes-pour-la-v2)).

> **Application en ligne :** _(lien à ajouter après le déploiement sur Streamlit Community Cloud)_

**Stack** : Python, pandas, scikit-learn, Streamlit, pydeck. Données gratuites uniquement.

## En bref

Résultats mesurés **une seule fois** sur les 21 925 ventes de 2025, jamais vues pendant la conception :

| | Ensemble | Appartements | Maisons |
|---|---:|---:|---:|
| Erreur moyenne du modèle (MAE) | **684 €/m²** | 598 €/m² | 745 €/m² |
| Erreur relative médiane | **13,3 %** | 11,0 % | 15,1 % |
| Erreur de la baseline (médiane par commune) | 884 €/m² | 823 €/m² | 927 €/m² |
| Gain par rapport à la baseline | **−22,6 %** | −27,3 % | −19,6 % |
| Couverture de la fourchette à 80 % | **80,4 %** | 80,4 % | 80,3 % |

Pour un bien dont le prix prédit est de 3 500 €/m², la fourchette affichée est d'environ 2 700 à 4 270 €/m² pour
un appartement et 2 170 à 4 550 €/m² pour une maison. **Elle est large : c'est voulu**, elle reflète l'erreur réelle du
modèle. Sur le test 2025, l'écart entre prix prédit et prix réel est inférieur à 10 % pour 39 % des ventes, à 20 % pour
67 %, à 30 % pour 81 %, et supérieur à 50 % pour 9 %.

## Question et données

*Étant donné un appartement ou une maison (commune, type, surface, pièces, terrain, dépendance), quel est son prix de
vente au m² attendu, et avec quelle marge d'erreur ?*

- **Source** : DVF géolocalisées (Etalab), un fichier par année et par département :
  `files.data.gouv.fr/geo-dvf/latest/csv/{année}/departements/33.csv.gz`. Gironde, 2021-2025
  (version d'avril 2026). Hors périmètre DVF : Alsace, Moselle, Mayotte.
- **Version figée** : `data/manifest.json` enregistre la taille et l'empreinte SHA-256 de chaque fichier (les
  fichiers sont remplacés à chaque mise à jour semestrielle). Détails dans [data/README.md](data/README.md).
- **Contours et populations des communes** : `geo.api.gouv.fr`.

## Démarche

### 1. Nettoyage : une ligne = une vente simple

Le fichier DVF compte une ligne par parcelle, nature de culture ou dépendance, pas une ligne par vente.
Règles (`src/immo/clean.py`, bornes dans `config.yaml`) : ventes ordinaires uniquement ; une seule disposition ;
aucun local commercial ; **un seul logement par transaction** (le prix d'une transaction regroupant plusieurs
logements n'est pas ventilable) ; dépendances tolérées ; lignes répétées d'un même bien fusionnées, surface de terrain =
somme des morceaux distincts ; bornes de surface, de pièces et de prix.

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

Résultat : 46 221 appartements et 68 666 maisons (1,7 % sans coordonnées, complétées par le centroïde de la commune).

### 2. Découpage dans le temps et baseline

| Jeu | Années | Ventes | Rôle |
|---|---|---:|---|
| Entraînement | 2021-2023 | 75 169 | apprendre, régler les hyperparamètres (validation croisée temporelle) |
| Validation | 2024 | 17 793 | comparer les modèles, calibrer la fourchette |
| **Test** | **2025** | 21 925 | **évalué une seule fois** (`python -m immo.evaluate --test`, avec garde-fou contre un second lancement) |

Baseline : médiane du prix/m² par (commune, type de bien), repli sur la médiane du type si la commune a moins de
5 ventes. Les seuils P1-P99 du prix/m² sont estimés sur les données d'ajustement et ne servent qu'à les filtrer :
**les jeux d'évaluation ne sont jamais tronqués.**

### 3. Modèles et variables

- **Ridge** (régression linéaire régularisée, interactions d'ordre 2) et **HistGradientBoostingRegressor** (arbres),
  tous deux sur le **logarithme** du prix/m², dans un `Pipeline` scikit-learn. Hyperparamètres choisis par validation
  croisée temporelle à fenêtre croissante sur l'entraînement seul (`reports/best_params.json`).
- **Variables** : surface, pièces, surface par pièce, type, terrain (maisons), dépendance, nombre de lots, index
  mensuel, commune et commune×type (encodage cible), latitude/longitude et axes tournés à 45°.
- **Variables géographiques dérivées** (variantes `_geo`) : distance au centre de Bordeaux et au littoral (tracé
  approximatif à ±3 km), population et densité de la commune, et **prix des 10 ventes voisines antérieures** de même type.
- **Anti-fuite** : tout ce qui s'apprend (centroïdes, encodage, prix des voisins, normalisation) est dans le `Pipeline`
  et ajusté sur les données d'ajustement seules. Le prix des voisins ne regarde que des ventes **strictement
  antérieures** (une vente ne se voit jamais elle-même, ni le futur, ni une vente du même jour). Des tests
  automatiques vérifient ces propriétés.

### 4. Fourchette de prix

Sur la validation, on mesure le rapport *prix réel / prix prédit* par type de bien. Ses quantiles à 10 % et 90 % donnent
des facteurs multiplicatifs : fourchette = prédiction × (facteur bas, facteur haut), soit × 0,76 à × 1,22 pour les
appartements et × 0,62 à × 1,30 pour les maisons. La couverture réelle est mesurée sur le test 2025, qui n'a servi
ni à ajuster ni à calibrer : 80,4 % pour 80 % visés.

## Résultats

### Sélection des modèles (validation 2024, ajustement sur 2021-2023)

| Modèle | MAE (€/m²) | vs baseline | MAPE | MAPE médiane | R² |
|---|---:|---:|---:|---:|---:|
| Baseline (médiane par commune) | 899 | — | 33,0 % | 17,3 % | 0,46 |
| Ridge | 833 | −7,3 % | 29,2 % | 16,0 % | 0,50 |
| Ridge + géographie | 774 | −13,9 % | 28,3 % | 15,1 % | 0,61 |
| HGB | 724 | −19,5 % | 26,8 % | 13,9 % | 0,65 |
| **HGB + géographie** | **717** | **−20,2 %** | 26,5 % | 13,6 % | 0,66 |

### Évaluation finale (test 2025, ajustement sur 2021-2024)

Le modèle retenu d'après la validation (HGB + géographie) a été évalué une seule fois ; tous les modèles sont
rapportés par transparence (`reports/metrics_test.json`).

| Modèle | MAE (€/m²) | vs baseline | MAPE | MAPE médiane | R² |
|---|---:|---:|---:|---:|---:|
| Baseline (médiane par commune) | 884 | — | 33,2 % | 17,5 % | 0,46 |
| Ridge | 811 | −8,3 % | 29,0 % | 15,9 % | 0,51 |
| Ridge + géographie | 742 | −16,1 % | 27,7 % | 14,7 % | 0,62 |
| HGB | 698 | −21,0 % | 25,8 % | 13,5 % | 0,65 |
| **HGB + géographie** | **684** | **−22,6 %** | **25,2 %** | **13,3 %** | **0,66** |

Le classement des modèles est le même que sur la validation. Les appartements sont mieux prédits (MAPE 18,2 %) que les
maisons (30,2 %) : le prix d'une maison dépend du terrain, de l'état et de la situation exacte, absents des données.

| Écart entre prix prédit et prix réel | < 10 % | < 20 % | < 30 % | < 50 % |
|---|---:|---:|---:|---:|
| Ensemble | 39 % | 67 % | 81 % | 91 % |
| Appartements | 46 % | 75 % | 87 % | 95 % |
| Maisons | 35 % | 61 % | 76 % | 88 % |

### Validation spatiale (communes jamais vues)

GroupKFold à 5 plis sur les ventes 2021-2024, en ignorant volontairement le temps pour isoler l'effet de la
géographie. Dans le schéma « commune », les communes évaluées sont absentes de l'entraînement.

| Schéma | Baseline | HGB | HGB + géographie |
|---|---:|---:|---:|
| Commune (534 groupes), MAE €/m² | 1 446 | 1 173 | **1 044** |
| Commune, MAPE | 52,2 % | 37,2 % | **32,9 %** |
| Blocs de 5 km (409 groupes), MAE €/m² | 1 079 | 842 | 837 |

Sur une commune jamais vue, les variables géographiques réduisent l'erreur de l'HGB de 11 %. Dans le schéma par blocs,
l'écart disparaît : une commune est souvent répartie sur plusieurs blocs, donc l'encodage de la commune reste
informatif ; ce schéma est moins sévère qu'il n'y paraît.

### Cartes

![Prix médian par commune](reports/figures/carte_prix.png)
![Erreurs du modèle sur le test 2025](reports/figures/carte_erreurs.png)

Les prix les plus élevés se concentrent sur le littoral du Bassin d'Arcachon et autour de Bordeaux, les plus bas
dans l'est et le sud ruraux. Les erreurs ne sont calculées que pour les communes avec au moins 10 ventes de test
(236 sur 501) ; le reste, surtout rural, est grisé.

## Application

`app/streamlit_app.py`, trois onglets :

- **Estimer un bien** : commune, type, surface, pièces, terrain, dépendance → fourchette en €/m² et en prix total,
  repère du prix médian de la commune, avertissement si la commune compte peu de ventes.
- **Cartes** : prix médian, biais et erreur relative médiane du modèle, cartes interactives.
- **Méthodologie et limites** : précision mesurée, lecture de la fourchette, limites.

```bash
streamlit run app/streamlit_app.py
```

**Le modèle en ligne est ré-ajusté sur 2021-2025** (même recette, une année de données en plus). Les chiffres de
précision ci-dessus décrivent le modèle ajusté sur 2021-2024 et évalué sur 2025 ; la précision exacte du modèle en
ligne n'est donc pas mesurée (il n'existe plus de données qu'il n'ait pas vues). Une vérification sur les ventes de
2026, à la prochaine mise à jour DVF, serait un vrai test dans le futur.

L'application ne lit que `app/artifacts/` (modèle, contours simplifiés, tables par commune), régénéré par
`python -m immo.export`. Le fichier du modèle embarque les coordonnées et prix (en logarithme) des ventes d'ajustement,
nécessaires à la variable « prix des voisins » : ce sont des données publiques DVF, et l'application n'affiche aucune adresse.

## Limites et pistes pour la v2

**Limites connues**

- **Variables absentes des données publiques** : état du bien, étage, vue, exposition, DPE, qualité de la rue. Deux biens
  identiques dans la même rue peuvent différer de 30 % : c'est la principale limite de précision. Le modèle donne un
  ordre de grandeur, pas un prix de vente ou d'achat.
- **Localisation à la commune** : le bien est placé au centre de la commune ; le quartier n'est pas pris en compte
  dans les grandes communes.
- **Maisons moins bien prédites** que les appartements ; fourchette plus large (× 0,62 à × 1,30).
- **Communes rurales** : peu de ventes, erreurs plus fortes, cartes d'erreurs grisées.
- **Marché** : le prix médian du m² a baissé entre 2022 et 2025 ; un retournement brusque ne serait pas anticipé.
- **Distance au littoral** : tracé à la main (±3 km).
- **Choix guidés par la validation** (variables retenues, index temporel pour les arbres, grille de réglage élargie
  une fois) : le test 2025 n'a été utilisé qu'une fois, à la fin, mais ces choix de conception ont vu les données 2021-2024.
- **Modèle en ligne** différent du modèle évalué (voir plus haut).
- **Ventes exclues** : ventes sur plan (VEFA), transactions regroupant plusieurs logements, locaux commerciaux.

**Pistes pour la v2** : joindre le DPE (ADEME, gratuit) par adresse ; saisie d'une adresse précise pour tenir compte du
quartier ; régression par quantiles pour une fourchette qui s'adapte au bien ; explicabilité (importance par permutation) ;
extension à d'autres départements puis à la France entière (lecture par morceaux, Parquet) ; test dans le futur sur les
ventes de 2026.

## Reproduire

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
set PYTHONPATH=src                     # PowerShell : $env:PYTHONPATH="src"  ·  Linux/macOS : export PYTHONPATH=src
```

```bash
python -m immo.download                # ventes DVF + manifeste (taille, SHA-256)
python -m immo.geo                     # contours et populations des communes
python -m immo.clean                   # nettoyage -> data/processed + reports/funnel.csv
python -m immo.train                   # réglage (validation croisée temporelle) + validation 2024 (~15 min)
python -m immo.spatial                 # validation spatiale -> reports/spatial_cv.json
python -m immo.evaluate --test         # évaluation finale 2025 : UNE SEULE FOIS
python -m immo.maps                    # cartes -> reports/figures
python -m immo.export                  # modèle 2021-2025 + artefacts de l'application
pytest                                 # 84 tests
```

La configuration (département, années, bornes de nettoyage, découpage, modèle retenu) est dans `config.yaml`.
Les graines aléatoires sont fixées ; les données intermédiaires sont régénérables et non versionnées.

## Structure du dépôt

```
config.yaml          paramètres (département, années, bornes, découpage)
app/                 streamlit_app.py, artifacts/ (modèle, contours, tables), requirements.txt
src/immo/            download, loader, clean, split, baseline, features, geo, neighbors, models, train,
                     evaluate, intervals, spatial, maps, predict, export, metrics, config
tests/               pytest (84 tests : nettoyage, absence de fuite, fourchette, validation spatiale, application)
notebooks/           01_exploration.ipynb
reports/             funnel.csv, metrics.json (validation), metrics_test.json (test 2025), best_params.json,
                     spatial_cv.json, intervalles.json, erreurs_par_commune.csv, figures/ (cartes)
data/                raw/ et processed/ (gitignorés), manifest.json, README.md
```

## Tests

Les 84 tests couvrent : schéma des données et manifeste ; règles de nettoyage sur des cas construits à la main
(deux appartements écartés, bien répété fusionné, terrain non compté deux fois) ; **absence de fuite** (encodeur ajusté
sur l'entraînement seul, prédiction indépendante de la cible, prix des voisins sans le futur, calibration de la
fourchette sans les prix du test, refus de relancer le test) ; distances géographiques ; validation spatiale ; et
l'application complète (`streamlit.testing`).

## Données, licence et vie privée

Données DVF © DGFiP, géolocalisation Etalab, **Licence Ouverte 2.0** ; contours et populations © geo.api.gouv.fr.
Aucune donnée personnelle n'est ajoutée, aucune adresse n'est affichée ; le jeu est public.

## Avancement

| Étape | Contenu | État |
|---|---|---|
| Sprint 1 | Données : téléchargement, lecture typée, exploration | ✅ |
| Sprint 2 | Nettoyage et entonnoir | ✅ |
| Sprint 3 | Découpage temporel, baseline, métriques | ✅ |
| Sprint 4 | Ridge et HistGradientBoosting | ✅ |
| Sprint 5 | Géographie, validation spatiale, cartes | ✅ |
| Sprint 6 | Évaluation finale 2025, fourchette, application, README | ✅ |
| Sprint 6 | Déploiement sur Streamlit Community Cloud | à faire |
