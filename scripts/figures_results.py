"""Figures et tableaux de résultats, lus exclusivement depuis results/ et artifacts/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stforecast import viz
from stforecast.data import load_locations, load_metr_la, make_windows, split_indices, time_features

RES, ART, OUT, TAB = Path("results"), Path("artifacts"), Path("site/figures"), Path("site/experience/_tables")
L = viz.MODEL_LABELS
MAIN = ["ha", "last_value", "stid", "gwnet"]
STID_ABL = ["stid", "stid_no_time", "stid_no_node", "stid_no_ids"]
GW_ABL = ["gwnet", "gwnet_road", "gwnet_adaptive", "gwnet_nograph"]


def load(name, kind="metrics"):
    p = RES / name / f"{kind}.json"
    return json.loads(p.read_text()) if p.exists() else None


def available(names):
    return [n for n in names if load(n)]


# ---------------------------------------------------------------- tableaux
def metric_table(names, path):
    rows = []
    best = {}
    for h in ["15min", "30min", "60min", "avg"]:
        for k in ["mae", "rmse", "mape"]:
            best[(h, k)] = min(load(n)[h][k] for n in names)
    for n in names:
        m = load(n)
        cells = []
        for h in ["15min", "30min", "60min", "avg"]:
            vals = []
            for k in ["mae", "rmse", "mape"]:
                v = m[h][k]
                s = f"{v:.2f}" + (" %" if k == "mape" else "")
                vals.append(f"**{s}**" if abs(v - best[(h, k)]) < 1e-9 else s)
            cells.append(" / ".join(vals))
        rows.append(f"| {L[n]} | " + " | ".join(cells) + " |")
    head = "| Modèle | 15 min | 30 min | 60 min | Moyenne 12 pas |\n|---|---|---|---|---|\n"
    path.write_text(head + "\n".join(rows) + "\n\n: MAE / RMSE / MAPE sur le test (mph). Gras : meilleur. "
                    "Généré par `scripts/figures_results.py`. {#tbl-" + path.stem + "}\n")
