| Modèle | Paramètres | s / époque | Meilleure époque | Minutes jusqu'au meilleur | Latence bs=1 (ms) | Latence bs=64 (ms) | Mémoire GPU pic (Mo) | MAE 60 min |
|---|---|---|---|---|---|---|---|---|
| STID | 118 k | 5.8 | 86 | 8.3 | 0.17 | 0.59 | 44 | 3.537 |
| STID sans id. temporelles | 34 k | 5.5 | 84 | 7.7 | 0.16 | 0.42 | 23 | 3.600 |
| STID sans id. spatiale | 68 k | 5.7 | 81 | 7.7 | 0.17 | 0.54 | 33 | 4.391 |
| MLP sans identités | 8 k | 5.4 | 98 | 8.7 | 0.20 | 0.25 | 13 | 4.535 |
| GWNet (route + adaptative) | 301 k | 53.8 | 94 | 82.7 | 1.92 | 70.12 | 1352 | 3.533 |
| GWNet (route) | 280 k | 58.7 | 90 | 88.3 | 1.46 | 56.94 | 1037 | 3.554 |
| GWNet (adaptative) | 268 k | 54.6 | 83 | 75.4 | 1.18 | 47.62 | 1040 | 3.503 |
| GWNet sans graphe (TCN) | 248 k | 39.2 | 64 | 45.1 | 0.56 | 17.44 | 654 | 4.452 |

: Coûts mesurés sur NVIDIA GeForce RTX 5070 Laptop GPU (deux entraînements en parallèle sur le GPU, donc temps absolus pessimistes ; les ratios sont comparables). {#tbl-costs}
