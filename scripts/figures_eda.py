"""Figures d'exploration de METR-LA (données + graphe), sans modèle entraîné."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from stforecast import viz
from stforecast.data import load_adjacency, load_locations, load_metr_la, make_windows, split_indices
from stforecast.graph import k_hop_reach

OUT = Path("site/figures")


def sensor_map(df, adj, loc):
    loc = loc.set_index("sensor_id").loc[df.columns]
    speed = df.replace(0, np.nan).mean().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ii, jj = np.nonzero(adj > 0.1)
    for i, j in zip(ii, jj):
        if i != j:
            ax.plot(loc.longitude.iloc[[i, j]], loc.latitude.iloc[[i, j]], color=viz.AXIS, lw=0.6, zorder=1)
    sc = ax.scatter(loc.longitude, loc.latitude, c=speed, cmap=viz.SEQ.reversed(), s=28,
                    edgecolor=viz.SURFACE, linewidth=0.8, zorder=2)
    fig.colorbar(sc, ax=ax, label="Vitesse moyenne (mph)", shrink=0.8)
    ax.set_title("207 capteurs de boucle, autoroutes de Los Angeles")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1.2)
    ax.text(0.01, 0.01, "Arêtes : adjacence routière DCRNN (poids > 0,1)", transform=ax.transAxes,
            color=viz.MUTED, fontsize=8)
    viz.save(fig, OUT / "eda_sensor_map.png")


def speed_heatmap(df, adj):
    week = df.loc["2012-03-05":"2012-03-11"].replace(0, np.nan)
    order = np.argsort(week.mean().to_numpy())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(week.to_numpy()[:, order].T, aspect="auto", cmap=viz.SEQ.reversed(),
                   interpolation="nearest", vmin=10, vmax=70)
    ticks = np.arange(0, len(week), 288)
    ax.set_xticks(ticks, [d.strftime("%a %d/%m") for d in week.index[ticks]])
    ax.set_ylabel("Capteurs (triés par vitesse moyenne)")
    ax.grid(False)
    ax.set_title("Une semaine de vitesses : congestions récurrentes et trous capteurs (blanc)")
    fig.colorbar(im, ax=ax, label="mph", shrink=0.85)
    viz.save(fig, OUT / "eda_speed_heatmap.png")
