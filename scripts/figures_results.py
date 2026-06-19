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


def horizon_zoom(names):
    """Zoom STID vs GWNet : gain relatif par pas."""
    s = np.array([x["mae"] for x in load("stid")["per_step"]])
    fig, ax = plt.subplots(figsize=(8, 4))
    for n in names:
        g = np.array([x["mae"] for x in load(n)["per_step"]])
        rel = (s - g) / s * 100
        ax.plot(np.arange(1, 13) * 5, rel, color=viz.MODEL_COLORS[n], marker="o", ms=4, label=L[n])
    ax.axhline(0, color=viz.AXIS, lw=1)
    ax.axhline(3, color=viz.MUTED, lw=1, ls=":")
    ax.text(61, 3, "seuil de\ndécision 3 %", color=viz.MUTED, fontsize=8, va="center")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_xlim(3, 70)
    ax.set_xticks([5, 15, 30, 45, 60])
    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("Gain de MAE vs STID (%)")
    ax.set_title("Gain relatif de chaque variante Graph WaveNet sur STID (> 0 : meilleur que STID)")
    viz.save(fig, OUT / "results_gain_vs_stid.png")


def ablation_bars(names, fname, title):
    """Graphique à points (axe non nul autorisé, contrairement aux barres)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, h in zip(axes, ["15min", "30min", "60min"]):
        vals = [load(n)[h]["mae"] for n in names]
        y = np.arange(len(names))[::-1]
        ax.hlines(y, min(vals) * 0.97, vals, color=viz.GRID, lw=2, zorder=1)
        ax.scatter(vals, y, s=90, color=[viz.MODEL_COLORS[n] for n in names], edgecolor=viz.SURFACE, lw=1.5,
                   zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v, yi + 0.22, f"{v:.2f}", ha="center", fontsize=9, color=viz.INK2)
        ax.set_yticks(y, [L[n] for n in names])
        ax.set_xlim(min(vals) * 0.97, max(vals) * 1.04)
        ax.set_ylim(-0.5, len(names) - 0.3)
        ax.set_title(h.replace("min", " min"), fontsize=10)
        ax.set_xlabel("MAE (mph)")
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    viz.save(fig, OUT / fname)


def pareto(names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, xl in [(axes[0], "train_time_to_best_s", "Temps d'entraînement jusqu'au meilleur modèle (min, log)"),
                        (axes[1], "latency_ms_bs64", "Latence d'inférence, batch 64 (ms, log)")]:
        for n in names:
            c, m = load(n, "cost"), load(n)
            x = c[key] / (60 if key.startswith("train") else 1)
            ax.scatter(x, m["60min"]["mae"], s=30 + c["params"] / 2500, color=viz.MODEL_COLORS[n],
                       edgecolor=viz.SURFACE, lw=1.5, zorder=3)
            if n in ("gwnet_road", "gwnet_adaptive"):
                continue
            lab = "GWNet avec graphe (3 variantes)" if n == "gwnet" else L[n]
            ax.annotate(lab, (x, m["60min"]["mae"]), xytext=(-6, 10) if n == "gwnet" else (6, 4),
                        textcoords="offset points", fontsize=8.5, color=viz.INK2,
                        ha="right" if n == "gwnet" else "left")
        ax.set_xscale("log")
        viz.plain_log_ticks(ax)
        ax.set_xlabel(xl)
        ax.set_ylabel("MAE à 60 min (mph)")
    fig.suptitle("Précision vs coût — en bas à gauche, c'est mieux (taille ∝ paramètres)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    viz.save(fig, OUT / "results_pareto.png")


def learning_curves(names):
    fig, ax = plt.subplots(figsize=(8, 4))
    ends = []
    for n in names:
        h = load(n, "history")
        t = np.cumsum([e["epoch_s"] for e in h]) / 60
        v = [e["val_mae"] for e in h]
        ax.plot(t, v, color=viz.MODEL_COLORS[n], lw=1.6)
        ends.append((t[-1], v[-1]))
    for n, (tx, _), y in zip(names, ends, viz.spread_labels([e[1] for e in ends], 0.03)):
        ax.text(tx * 1.05, y, L[n], color=viz.INK2, fontsize=8.5, va="center")
    ax.set_xscale("log")
    viz.plain_log_ticks(ax)
    ax.set_ylim(2.6, 3.4)
    ax.set_xlabel("Temps d'entraînement cumulé (min, log)")
    ax.set_ylabel("MAE validation (mph)")
    ax.set_title("Courbes d'apprentissage en temps réel : STID converge en quelques minutes")
    viz.save(fig, OUT / "results_learning_curves.png")


def substitution():
    """Identités (STID) et graphe (GWNet) : deux leviers qui résolvent le même problème."""
    pairs = [("stid_no_ids", "stid", "STID : + identités", viz.CAT[0]),
             ("gwnet_nograph", "gwnet", "Graph WaveNet : + graphe", viz.CAT[1])]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for i, (a, b, _, c) in enumerate(pairs):
        va, vb = load(a)["60min"]["mae"], load(b)["60min"]["mae"]
        cb = load(b, "cost")["epoch_s_mean"]
        y = 1 - i
        ax.annotate("", xy=(vb, y), xytext=(va, y),
                    arrowprops={"arrowstyle": "-|>", "color": c, "lw": 2.2, "mutation_scale": 16})
        ax.scatter([va], [y], s=60, color=viz.SURFACE, edgecolor=c, lw=2, zorder=3)
        ax.scatter([vb], [y], s=80, color=c, edgecolor=viz.SURFACE, lw=1.5, zorder=3)
        ax.text(va + 0.02, y + 0.18, f"sans : {va:.2f}", fontsize=9, color=viz.INK2)
        ax.text(vb + 0.02, y + 0.18, f"avec : {vb:.2f}  ({(vb - va) / va * 100:+.0f} %, {cb:.0f} s/époque)",
                fontsize=9, color=viz.INK2, ha="left")
    ax.set_yticks([1, 0], [p[2] for p in pairs])
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlim(3.3, 4.7)
    ax.invert_xaxis()
    ax.set_xlabel("MAE à 60 min (mph) — vers la droite, c'est mieux")
    ax.grid(axis="y", visible=False)
    ax.set_title("Deux leviers, même effet : lever l'indistinguabilité")
    viz.save(fig, OUT / "results_substitution.png")


def _test_frame():
    df = load_metr_la()
    speed = df.to_numpy()
    test = split_indices(make_windows(len(df)))["test"]
    y = speed[test[:, None] + 12 + np.arange(12)[None, :]]
    last_in = speed[test + 11]
    return df, test, y, last_in


def per_sensor_map(df, y, preds):
    loc = load_locations().set_index("sensor_id").loc[df.columns]
    m = y[:, -1] > 0
    def mae_node(p):
        return (np.abs(p[:, -1] - y[:, -1]) * m).sum(0) / m.sum(0)
    diff = mae_node(preds["stid"]) - mae_node(preds["gwnet"])  # > 0 : GWNet meilleur
    lim = np.abs(diff).max()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    sc = axes[0].scatter(loc.longitude, loc.latitude, c=diff, cmap=viz.DIV.reversed(), vmin=-lim, vmax=lim,
                         s=36, edgecolor=viz.SURFACE, lw=0.8)
    fig.colorbar(sc, ax=axes[0], label="MAE STID − MAE GWNet (mph)", shrink=0.85)
    axes[0].set_title("Où le graphe aide-t-il ? (bleu : GWNet meilleur, rouge : STID meilleur)", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].grid(False)
    axes[1].hist(diff, bins=30, color=viz.CAT[0], edgecolor=viz.SURFACE)
    axes[1].axvline(0, color=viz.INK, lw=1)
    axes[1].set_xlabel("MAE STID − MAE GWNet à 60 min (mph)")
    axes[1].set_ylabel("Capteurs")
    axes[1].set_title(f"GWNet meilleur sur {(diff > 0).mean() * 100:.0f} % des capteurs", fontsize=10)
    fig.tight_layout()
    viz.save(fig, OUT / "results_sensor_gain.png")
    return diff


def regimes(y, last_in, preds):
    """MAE à 60 min par régime de trafic, défini par l'entrée et la cible."""
    yt, li = y[:, -1], last_in
    valid = (yt > 0) & (li > 0)
    regs = {
        "Fluide stable": valid & (li >= 55) & (yt >= 55),
        "Début de congestion": valid & (li >= 55) & (yt < 40),
        "Congestion": valid & (li < 40) & (yt < 40),
        "Résorption": valid & (li < 40) & (yt >= 55),
    }
    rows = []
    for r, m in regs.items():
        row = {"Régime": r, "Part du test": f"{m.sum() / valid.sum() * 100:.1f} %"}
        for n, p in preds.items():
            row[L[n]] = np.abs(p[:, -1] - yt)[m].mean()
        rows.append(row)
    t = pd.DataFrame(rows)
    names = list(preds)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(t))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * w, t[L[n]], width=w * 0.92, color=viz.MODEL_COLORS[n], label=L[n])
    ax.set_xticks(x, [f"{a}\n({b})" for a, b in zip(t["Régime"], t["Part du test"])])
    ax.set_ylabel("MAE à 60 min (mph)")
    ax.legend(loc="lower left", ncol=len(names), bbox_to_anchor=(0, 1.0))
    ax.set_title("Erreur par régime : les débuts de congestion concentrent la difficulté", pad=34)
    viz.save(fig, OUT / "results_regimes.png")
    return t


def error_by_hour(df, test, y, preds):
    hours = df.index[test + 12 + 11].hour
    fig, ax = plt.subplots(figsize=(9, 3.8))
    m = y[:, -1] > 0
    ends = []
    for n, p in preds.items():
        ok = m.sum(1) > 0
        err = (np.abs(p[:, -1] - y[:, -1]) * m).sum(1)[ok] / m.sum(1)[ok]
        e = pd.Series(err).groupby(np.asarray(hours)[ok]).mean()
        ax.plot(e.index, e.to_numpy(), color=viz.MODEL_COLORS[n], marker="o", ms=3)
        ends.append(e.iloc[-1])
    for n, y_ in zip(preds, viz.spread_labels(ends, 0.3)):
        ax.text(23.4, y_, L[n], color=viz.INK2, fontsize=8.5, va="center")
    ax.set_xlim(-0.5, 28)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Heure de la cible")
    ax.set_ylabel("MAE à 60 min (mph)")
    ax.set_title("Erreur selon l'heure : les pointes concentrent l'erreur pour tous les modèles")
    viz.save(fig, OUT / "results_error_by_hour.png")
