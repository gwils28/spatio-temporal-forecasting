"""Figures pédagogiques : diffusion sur graphe, frise du domaine."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stforecast import viz
from stforecast.data import load_adjacency, load_locations
from stforecast.graph import random_walk

OUT = Path("site/figures")


def diffusion_maps():
    """Masse d'une impulsion après k pas de marche aléatoire, aval (P) vs amont (P_b)."""
    ids, adj = load_adjacency()
    loc = load_locations().set_index("sensor_id").loc[ids]
    a = adj.copy()
    np.fill_diagonal(a, 0)
    fwd, bwd = random_walk(a), random_walk(a.T)
    src = int(np.argmax(((a > 0).sum(1) > 4) & ((a > 0).sum(0) > 4)))
    x0 = np.zeros(len(ids))
    x0[src] = 1.0
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    for row, (P, name) in enumerate([(fwd, "aval  (D_O⁻¹W)ᵏ"), (bwd, "amont  (D_I⁻¹Wᵀ)ᵏ")]):
        x = x0.copy()
        for k in range(4):
            ax = axes[row, k]
            ax.scatter(loc.longitude, loc.latitude, s=8, color=viz.GRID, zorder=1)
            m = x > 1e-4
            ax.scatter(loc.longitude[m], loc.latitude[m], s=20 + 400 * np.sqrt(x[m]), c=x[m],
                       cmap=viz.SEQ, vmin=0, vmax=max(x.max(), 1e-3), edgecolor=viz.SURFACE, lw=0.6, zorder=2)
            ax.scatter(loc.longitude.iloc[src], loc.latitude.iloc[src], marker="x", color=viz.INK, s=40, zorder=3)
            ax.set_title(f"k = {k}" + ("" if row else ""), fontsize=10)
            if k == 0:
                ax.set_ylabel(name, color=viz.INK2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            x = x @ P  # un pas de diffusion
    fig.suptitle("Convolution par diffusion : où va l'information d'un capteur (×) en k pas de marche aléatoire",
                 x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.01, "Taille / couleur des points : masse de probabilité. Un filtre DCRNN d'ordre K combine "
             "les termes k = 0..K-1 des deux directions avec des poids θ appris.", color=viz.MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    viz.save(fig, OUT / "concept_diffusion.png")


def timeline():
    events = [
        (2018.0, "DCRNN", "diffusion, ICLR", 0),
        (2018.5, "STGCN", "spectral, IJCAI", 0),
        (2019.5, "Graph WaveNet", "graphe adaptatif, IJCAI", 0),
        (2020.5, "AGCRN / MTGNN", "graphe appris (NeurIPS / KDD)", 0),
        (2022.5, "STID", "MLP + identités (CIKM)", 1),
        (2022.6, "STEP", "pré-entraînement (KDD)", 2),
        (2024.3, "OpenCity / UniST", "FM urbains", 2),
        (2025.2, "Survey STFM", "Liang et al., arXiv:2503.13502", 2),
        (2026.15, "UrbanFM", "zero-shot, 100+ villes", 2),
    ]
    colors = [viz.CAT[1], viz.CAT[0], viz.CAT[2]]
    lanes = ["STGNN :\nplus de graphe", "Contre-pied :\nsimplicité", "Fondation :\npré-entraîner"]
    gap = 1.3
    fig, ax = plt.subplots(figsize=(12, 5))
    for lane in range(3):
        ax.axhline(lane * gap, color=viz.GRID, lw=1, zorder=0)
    seen = {0: 0, 1: 0, 2: 0}
    for x, name, desc, lane in events:
        y = lane * gap
        up = seen[lane] % 2 == 0
        seen[lane] += 1
        ax.scatter(x, y, s=70, color=colors[lane], edgecolor=viz.SURFACE, lw=1.5, zorder=3)
        if up:
            ax.text(x, y + 0.12, desc, fontsize=8, ha="center", va="bottom", color=viz.INK2)
            ax.text(x, y + 0.30, name, fontsize=9.5, ha="center", va="bottom", fontweight="bold")
        else:
            ax.text(x, y - 0.12, name, fontsize=9.5, ha="center", va="top", fontweight="bold")
            ax.text(x, y - 0.32, desc, fontsize=8, ha="center", va="top", color=viz.INK2)
    ax.set_yticks([i * gap for i in range(3)], lanes)
    for t, c in zip(ax.get_yticklabels(), colors):
        t.set_color(c)
        t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, 2 * gap + 0.7)
    ax.set_xlim(2017.4, 2026.9)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_title("Prévision de trafic sur graphe : trois trajectoires de recherche")
    viz.save(fig, OUT / "concept_timeline.png")
