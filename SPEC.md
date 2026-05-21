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

## Protocole (figé)

- Données : `metr-la.h5` + `adj_mx.pkl` du dépôt DCRNN, 34 272 pas × 207 capteurs.
- Tâche : 12 → 12 (1 h d'historique → 1 h d'horizon, pas de 5 min).
- Split chronologique 70 / 10 / 20 sur les fenêtres ; z-score ajusté sur le train.
- Zéros = valeurs manquantes : masqués dans la perte **et** les métriques.
- Métriques : MAE, RMSE, MAPE masqués à 15 / 30 / 60 min (pas 3 / 6 / 12) + moyenne.
- Sélection : meilleure MAE validation ; le test n'est lu qu'une fois par run.

## Expériences

| Run | Rôle |
|---|---|
| `ha`, `last_value` | bornes basses sans apprentissage |
| `stid` | baseline non-graphe fort (identités espace + temps) |
| `stid_no_node`, `stid_no_time`, `stid_no_ids` | ablation : démontrer l'indistinguabilité |
| `gwnet` | Graph WaveNet complet (route + adaptative) |
| `gwnet_road`, `gwnet_adaptive`, `gwnet_nograph` | ablation : d'où vient le graphe |

## Critère de décision (fixé avant résultats)

Le graphe « gagne son pari » si **les deux** conditions tiennent :
1. gain de MAE à 60 min ≥ **3 %** relatif vs STID (≈ 0,1 mph, au-delà du bruit inter-seed
   typique rapporté par BasicTS) ;
2. et ce gain est attribuable au graphe : `gwnet` bat nettement `gwnet_nograph`.

Sinon la recommandation par défaut est STID, et le graphe reste une option à
re-tester quand la topologie apporte une information absente des identités
(réseau neuf, nœuds sans historique).

## Livrables

1. Code + tests (`src/stforecast`, `tests/`).
2. Tableau métriques par horizon + ablations + coût (`results/`).
3. Site Quarto (veille, concepts, EDA, résultats, system design, lecture DCRNN).
