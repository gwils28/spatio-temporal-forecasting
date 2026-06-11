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
