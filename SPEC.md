# SPEC — Le graphe mérite-t-il sa complexité ?

> Issu de l'interview de cadrage (étape « lever les inconnues »). Ce fichier fixe le
> protocole **avant** les résultats : on ne le modifie pas pour accommoder un chiffre.

## Question

Sur METR-LA (207 capteurs de vitesse, Los Angeles), un STGNN (Graph WaveNet) bat-il
un baseline **non-graphe** fort (STID) d'une marge qui justifie son surcoût
compute / latence / complexité ?

## Décisions de cadrage

| Sujet | Décision | Raison |
|---|---|---|
| Dépôt | `gwils28/spatio-temporal-forecasting`, public | site Quarto sur GitHub Pages |
| Modèle graphe | Graph WaveNet + ablation de la source du graphe | rapide, adjacence route **et** adaptative |
| Implémentation | PyTorch pur, modèles réécrits (~150 lignes chacun) | lisible, testable, pas de boîte noire |
| Budget | protocole complet des papiers, 1 seed | ≈ 100 époques, early stopping |
