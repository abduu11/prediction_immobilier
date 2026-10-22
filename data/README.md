# Données

## Source
DVF géolocalisées (Etalab), fichiers par année et par département :
`https://files.data.gouv.fr/geo-dvf/latest/csv/{année}/departements/{dep}.csv.gz`

- Producteur : DGFiP (DVF), géolocalisation Etalab. Licence Ouverte 2.0.
- Années disponibles : 2021–2025 (version d'avril 2026). Mises à jour en avril et octobre ;
  chaque mise à jour **remplace** les fichiers et peut ajouter des transactions à tous les millésimes.
- Hors périmètre : Alsace (67, 68), Moselle (57), Mayotte (976).
- Les coordonnées sont celles du centre de la parcelle (WGS-84) ; ~2,5 % de lignes sans coordonnées.

## Communes (contours, population)
`https://geo.api.gouv.fr/departements/{dep}/communes?format=geojson&geometry=contour&fields=nom,code,population,surface,codesPostaux`
(Licence Ouverte, Etalab), téléchargé par `python -m immo.geo` dans `raw/communes_{dep}.geojson` et enregistré dans
le manifeste. La population est un instantané récent, utilisé comme variable de contexte (pas une information sur
la vente). Une commune de nos ventes (33409, 7 ventes) n'existe plus dans l'API : sa population est imputée.

## Version figée
`manifest.json` enregistre pour chaque fichier téléchargé : URL, date de téléchargement,
taille et SHA-256. Les fichiers bruts (`raw/`) ne sont pas versionnés dans git.
Si l'empreinte change après une mise à jour (octobre 2026), les métriques du README
ne sont plus directement comparables.

## Schéma réel (40 colonnes, vérifié sur `dvf_2024_33.csv.gz`)
`id_mutation, date_mutation, numero_disposition, nature_mutation, valeur_fonciere,
adresse_numero, adresse_suffixe, adresse_nom_voie, adresse_code_voie, code_postal,
code_commune, nom_commune, code_departement, ancien_code_commune, ancien_nom_commune,
id_parcelle, ancien_id_parcelle, numero_volume, lot1_numero … lot5_surface_carrez,
nombre_lots, code_type_local, type_local, surface_reelle_bati, nombre_pieces_principales,
code_nature_culture, nature_culture, code_nature_culture_speciale, nature_culture_speciale,
surface_terrain, longitude, latitude`

Valeurs observées : `nature_mutation` ∈ {Vente, Vente en l'état futur d'achèvement, Echange,
Adjudication, Vente terrain à bâtir, Expropriation} ; `type_local` ∈ {Maison, Appartement,
Dépendance, « Local industriel. commercial ou assimilé »} (ou vide).

## Chiffres Gironde 2021–2025 (sprint 1)
- 525 352 lignes brutes (≈ 270 Mo en mémoire avec les colonnes utiles).
- 167 120 lignes de vente d'appartement/maison, réparties sur 125 513 mutations
  (107 690 mutations à une seule ligne de bien, 17 823 à plusieurs lignes → traitées au sprint 2).
- Prix au m² brut : médiane ≈ 4 091 €/m² (appartements) et 3 344 €/m² (maisons), avec des
  valeurs extrêmes (0 à > 3 M€/m²) : le nettoyage du sprint 2 est indispensable.
