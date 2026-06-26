# Le graphe mérite-t-il sa complexité ?

**Prévision spatio-temporelle sur réseau de capteurs : STID (sans graphe) vs Graph WaveNet (STGNN) sur METR-LA.**
Travaux de veille et d'expérimentation, cas d'usage logistique (vitesses et temps de trajet sur un réseau de transport).

📖 **Site de documentation :** <https://gwils28.github.io/spatio-temporal-forecasting/>

## Question

Un STGNN exploite la corrélation spatiale (la congestion se propage le long du réseau).
Mais **le graphe est une hypothèse, pas une donnée** : avant de déployer un STGNN, il faut
montrer qu'il bat un baseline *non-graphe* bien conçu, d'une marge qui justifie son
surcoût en calcul, en latence et en dette technique.

Protocole et critère de décision fixés **avant** les résultats : [`SPEC.md`](SPEC.md).

## Structure

```
src/stforecast/      données, métriques, graphe, baselines, modèles (STID, Graph WaveNet), entraînement
configs/             une config YAML par run (8 runs)
scripts/             téléchargement, lancement, baselines, figures, notebook
tests/               24 tests unitaires (pytest, < 1 s)
results/             métriques, courbes et coûts de chaque run (versionnés)
site/                site Quarto (veille, concepts, expérience, system design, journal)
docs/notes/          fiches de veille vérifiées, notes d'implémentation
```

## Reproduire

```bash
uv sync                                   # Python 3.12, torch CUDA 12.8
./scripts/download_data.sh                # METR-LA + graphe capteurs (dépôt DCRNN)
make check                                # ruff + pytest
./scripts/run_all.sh                      # 8 entraînements (≈ 3 h sur RTX 5070 Laptop)
uv run python scripts/eval_baselines.py   # moyenne historique, persistance
make figures site                         # figures, tableaux, notebook, site
```

## Site

| Section | Contenu |
|---|---|
| Veille | survey des modèles de fondation ST, UrbanFM, STID, lecture DCRNN |
| Concepts | le graphe-hypothèse, l'indistinguabilité des échantillons, la diffusion, les architectures |
| Expérience | données, notebook d'exploration, protocole, résultats, ablations, coûts |
| System design | prévision ST pour un réseau logistique : origine du graphe, échelle, robustesse |
| Journal | déroulé daté du projet |

## Résultats

| Modèle (METR-LA, test) | MAE 15 min | MAE 30 min | MAE 60 min | s / époque | Latence batch 64 |
|---|---|---|---|---|---|
| Moyenne historique | 4.19 | 4.19 | 4.19 | — | — |
| **STID** (sans graphe) | 2.80 | 3.17 | 3.54 | **5.8** | **0.6 ms** |
| **Graph WaveNet** (route + adaptative) | **2.69** | **3.07** | **3.53** | 53.8 | 70 ms |
| Graph WaveNet *sans graphe* | 2.99 | 3.59 | 4.45 | 39.2 | — |

**Verdict (critère fixé avant les résultats) : le graphe ne gagne pas son pari en moyenne.**
Même MAE à 60 min, pour un coût d'entraînement 10 fois plus élevé et une latence en batch
119 fois plus élevée. Le graphe aide pourtant à court terme (−4 % à 15 min) et **en début de
congestion (−23 % d'erreur)**. Graphe et identités apprises sont deux leviers substituables
pour lever l'indistinguabilité des échantillons.
→ [Résultats détaillés](https://gwils28.github.io/spatio-temporal-forecasting/experience/resultats.html)
