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


def daily_profile(df):
    s = df.replace(0, np.nan)
    tod = s.index.hour + s.index.minute / 60
    wk = s.index.day_of_week < 5
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for mask, lab, c in [(wk, "Semaine", viz.CAT[0]), (~wk, "Week-end", viz.CAT[1])]:
        prof = s[mask].groupby(tod[mask]).mean().mean(axis=1)
        q = s[mask].groupby(tod[mask]).mean()
        ax.fill_between(prof.index, q.quantile(0.1, axis=1), q.quantile(0.9, axis=1), color=c, alpha=0.12, lw=0)
        ax.plot(prof.index, prof, color=c, label=lab)
        ax.text(prof.index[-1] + 0.2, prof.iloc[-1], lab, color=viz.INK2, va="center", fontsize=9)
    ax.set_xlim(0, 26)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("Heure de la journée")
    ax.set_ylabel("Vitesse (mph)")
    ax.set_title("Profil journalier : deux pointes en semaine, quasi plat le week-end")
    ax.text(0.01, 0.02, "Ligne : moyenne réseau · bande : capteurs P10–P90", transform=ax.transAxes,
            color=viz.MUTED, fontsize=8)
    viz.save(fig, OUT / "eda_daily_profile.png")


def missing(df):
    miss = (df == 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [2, 1]})
    daily = miss.resample("D").mean().mean(axis=1) * 100
    axes[0].bar(daily.index, daily.to_numpy(), width=0.8, color=viz.CAT[0])
    axes[0].set_ylabel("% de valeurs manquantes")
    axes[0].set_title("Trous dans le temps (par jour)")
    axes[0].tick_params(axis="x", rotation=30)
    per = miss.mean().to_numpy() * 100
    axes[1].hist(per, bins=30, color=viz.CAT[0], edgecolor=viz.SURFACE)
    axes[1].set_xlabel("% manquant par capteur")
    axes[1].set_ylabel("Capteurs")
    axes[1].set_title("…et par capteur")
    fig.suptitle(f"Données manquantes : {miss.to_numpy().mean() * 100:.1f} % des valeurs sont des zéros",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    viz.save(fig, OUT / "eda_missing.png")
