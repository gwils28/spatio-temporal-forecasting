.PHONY: data test lint check train baselines figures site all

data:        ## Télécharge METR-LA + graphe capteurs
	./scripts/download_data.sh
test:        ## Suite de tests rapide (< 1 s)
	uv run pytest
lint:
	uv run ruff check src tests scripts
check: lint test
train:       ## 8 entraînements (≈ 3 h sur GPU laptop)
	./scripts/run_all.sh
baselines:
	uv run python scripts/eval_baselines.py
figures:     ## Toutes les figures + tableaux du site
	uv run python scripts/make_figures.py
	uv run python scripts/build_notebook.py
	uv run jupyter nbconvert --to notebook --execute --inplace site/experience/exploration.ipynb
site:
	quarto render site
all: check baselines figures site
