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
