"""Régénère toutes les figures du site (EDA, concepts, résultats)."""

import runpy

for script in ["figures_eda", "figures_concepts", "figures_results"]:
    print(f"→ {script}")
    runpy.run_path(f"scripts/{script}.py", run_name="__main__")
