# CLAUDE.md — spatio-temporal-forecasting

Question du projet : un STGNN (Graph WaveNet) bat-il STID (MLP + identités) sur METR-LA
d'une marge qui justifie son coût ? Protocole figé dans `SPEC.md` — ne pas le modifier
pour accommoder un résultat.

## Commandes
- Tests : `uv run pytest` · Lint : `uv run ruff check src tests scripts`
- Un run : `uv run python -m stforecast.train --config configs/<run>.yaml`
- Tous les runs : `./scripts/run_all.sh` · Baselines : `uv run python scripts/eval_baselines.py`
- Figures : `uv run python scripts/make_figures.py` · Site : `quarto render site`

## Invariants (ne jamais violer)
- Les zéros de METR-LA sont des trous capteurs : toujours masqués (perte et métriques).
- Split chronologique, scaler ajusté sur le train seulement, test lu une fois par run.
- `data/raw/` est en lecture seule et non versionné (`.gitignore`).
- Tout chiffre publié dans `site/` vient d'un fichier de `results/` ou d'une source
  citée dans `docs/notes/veille_fiches.md` — jamais de mémoire.

## Conventions
- Code et commentaires : docstrings en français, noms en anglais.
- Commits : Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `exp:`, `chore:`).
- Figures générées par script dans `site/figures/` (reproductibles), pas à la main.
- Déviations au plan : consigner dans `docs/notes/implementation.md`.

## Compaction
En cas de compaction, conserver : la question, le critère de décision de `SPEC.md`,
l'état des runs (`results/`), et les déviations consignées.
