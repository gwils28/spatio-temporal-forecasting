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


def cost_table(names, path):
    rows = []
    ref = load("stid", "cost")
    for n in names:
        c, m = load(n, "cost"), load(n)
        rows.append(f"| {L[n]} | {c['params'] / 1e3:.0f} k | {c['epoch_s_mean']:.1f} | {c['best_epoch']} | "
                    f"{c['train_time_to_best_s'] / 60:.1f} | {c['latency_ms_bs1']:.2f} | "
                    f"{c['latency_ms_bs64']:.2f} | {c['peak_gpu_mem_mb']:.0f} | {m['60min']['mae']:.3f} |")
    head = ("| Modèle | Paramètres | s / époque | Meilleure époque | Minutes jusqu'au meilleur | "
            "Latence bs=1 (ms) | Latence bs=64 (ms) | Mémoire GPU pic (Mo) | MAE 60 min |\n"
            + "|---" * 9 + "|\n")
    path.write_text(head + "\n".join(rows) + f"\n\n: Coûts mesurés sur {ref['device']} (deux entraînements "
                    "en parallèle sur le GPU, donc temps absolus pessimistes ; les ratios sont comparables). "
                    "{#tbl-" + path.stem + "}\n")


# ---------------------------------------------------------------- figures
def horizon_curve(names):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ends = []
    for n in names:
        mae = [s["mae"] for s in load(n)["per_step"]]
        x = np.arange(1, 13) * 5
        style = {"ls": "--"} if n in ("ha", "last_value") else {}
        ax.plot(x, mae, color=viz.MODEL_COLORS[n], marker="o", ms=4, **style)
        ends.append(mae[-1])
    for n, y in zip(names, viz.spread_labels(ends, 0.17)):
        ax.text(62, y, L[n], color=viz.INK2, va="center", fontsize=9)
    ax.set_xlim(3, 82)
    ax.set_xticks([5, 15, 30, 45, 60])
    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("MAE (mph)")
    ax.set_title("L'erreur croît avec l'horizon ; l'écart STID / GWNet reste faible")
    viz.save(fig, OUT / "results_horizon.png")
