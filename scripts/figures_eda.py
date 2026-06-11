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


def adjacency(adj):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    im = axes[0].imshow(np.where(adj > 0, adj, np.nan), cmap=viz.SEQ, interpolation="nearest")
    axes[0].set_title("Matrice d'adjacence W")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    a = adj.copy()
    np.fill_diagonal(a, 0)
    axes[1].hist((a > 0).sum(1), bins=range(0, 30), color=viz.CAT[0], edgecolor=viz.SURFACE)
    axes[1].set_title("Degré sortant")
    axes[1].set_xlabel("Voisins (poids > 0)")
    reach = [k_hop_reach(a, k) for k in range(1, 7)]
    axes[2].boxplot(reach, tick_labels=range(1, 7), patch_artist=True,
                    boxprops={"facecolor": viz.BLUE_RAMP[1], "color": viz.BLUE_RAMP[4]},
                    medianprops={"color": viz.INK}, whiskerprops={"color": viz.AXIS},
                    capprops={"color": viz.AXIS}, flierprops={"markeredgecolor": viz.MUTED})
    axes[2].set_title("Champ réceptif d'une diffusion d'ordre K")
    axes[2].set_xlabel("K (sauts)")
    axes[2].set_ylabel("Nœuds atteints")
    one_way = ((a > 0) & (a.T == 0)).sum() / (a > 0).sum()
    fig.suptitle(f"Le graphe routier : creux ({(a > 0).mean() * 100:.1f} % d'arêtes), orienté "
                 f"({one_way * 100:.0f} % des arêtes sans arête retour)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    viz.save(fig, OUT / "eda_adjacency.png")


def propagation(df, adj):
    """Décalage de corrélation croisée amont→aval : la congestion se propage-t-elle ?"""
    x = df.replace(0, np.nan).interpolate(limit_direction="both").to_numpy()
    dx = np.diff(x, axis=0)
    dx = (dx - dx.mean(0)) / (dx.std(0) + 1e-6)
    a = adj.copy()
    np.fill_diagonal(a, 0)
    ii, jj = np.nonzero((a > 0.5) & (a.T < 0.1))  # arêtes nettement orientées
    lags = np.arange(-6, 7)
    best = []
    for i, j in zip(ii, jj):
        cc = [np.mean(dx[max(0, -k): len(dx) - max(0, k), i] * dx[max(0, k): len(dx) - max(0, -k), j])
              for k in lags]
        best.append(lags[int(np.argmax(cc))])
    best = np.array(best)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    counts = np.array([(best == k).sum() for k in lags])
    colors = [viz.CAT[0] if k < 0 else viz.MUTED if k == 0 else viz.CAT[1] for k in lags]
    ax.bar(lags * 5, counts, width=4, color=colors)
    neg, pos = (best < 0).mean() * 100, (best > 0).mean() * 100
    ax.annotate(f"{neg:.0f} % : l'aval ralentit d'abord\n(onde de congestion qui remonte)",
                xy=(-7, counts[lags == -1][0] * 0.6), xytext=(-33, counts.max() * 0.55),
                color=viz.INK2, arrowprops={"arrowstyle": "-", "color": viz.MUTED})
    ax.annotate(f"{pos:.0f} % : l'amont d'abord", xy=(7, counts[lags == 1][0] + 2),
                xytext=(12, counts.max() * 0.3), color=viz.INK2,
                arrowprops={"arrowstyle": "-", "color": viz.MUTED})
    ax.set_xlabel("Décalage du pic de corrélation, aval vs amont (minutes)")
    ax.set_ylabel("Arêtes orientées")
    ax.set_title(f"La congestion se propage vers l'amont ({len(best)} arêtes, Δvitesse corrélée)")
    viz.save(fig, OUT / "eda_propagation.png")
    return best


def indistinguishability(df):
    """Paires de fenêtres d'entrée quasi identiques mais aux futurs divergents."""
    rng = np.random.default_rng(1)
    v = df.to_numpy()
    starts = make_windows(int(len(v) * 0.7))
    n = 6000
    t = rng.choice(starts, n)
    s = rng.integers(0, v.shape[1], n)
    xin = v[t[:, None] + np.arange(12), s[:, None]]
    yout = v[t[:, None] + 12 + np.arange(12), s[:, None]]
    ok = (xin > 0).all(1) & (yout > 0).all(1)
    xin, yout, t, s = xin[ok], yout[ok], t[ok], s[ok]
    d_in = np.sqrt(((xin[:, None] - xin[None]) ** 2).mean(-1))
    d_out = np.abs(yout[:, None, -1] - yout[None, :, -1])
    np.fill_diagonal(d_in, np.inf)
    score = np.where(d_in < 0.8, d_out, -1)
    i, j = np.unravel_index(np.argmax(score), score.shape)
    idx = df.index
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k, c in [(i, viz.CAT[0]), (j, viz.CAT[1])]:
        lab = f"capteur {df.columns[s[k]]}, {idx[t[k] + 11]:%a %H:%M}"
        ax.plot(range(-11, 1), xin[k], color=c, marker="o", ms=4)
        ax.plot(range(1, 13), yout[k], color=c, ls="--", label=lab)
    ax.axvline(0.5, color=viz.AXIS, lw=1)
    ax.text(-11, ax.get_ylim()[0] + 1, "1 h observée (identique)", color=viz.INK2)
    ax.text(1.3, ax.get_ylim()[0] + 1, "1 h future (divergente)", color=viz.INK2)
    ax.set_xlabel("Pas de 5 min")
    ax.set_ylabel("Vitesse (mph)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2)
    ax.set_title(f"Indistinguabilité : RMS d'entrée {d_in[i, j]:.2f} mph, écart à 60 min {d_out[i, j]:.0f} mph")
    viz.save(fig, OUT / "concept_indistinguishability.png")
    # Statistique globale : pour les paires d'entrées proches, dispersion des futurs.
    close = d_in < 1.0
    return float(np.median(d_out[close])), int(close.sum() // 2)


def split_timeline(df):
    splits = split_indices(make_windows(len(df)))
    idx = df.index
    fig, ax = plt.subplots(figsize=(9, 1.8))
    for (name, st), c in zip(splits.items(), [viz.CAT[0], viz.CAT[3], viz.CAT[1]]):
        a, b = idx[st[0]], idx[min(st[-1] + 23, len(idx) - 1)]
        ax.barh(0, (b - a).days + 1, left=a, color=c, height=0.5, edgecolor=viz.SURFACE, lw=2)
        ax.text(a + (b - a) / 2, 0, f"{name}\n{len(st):,}".replace(",", " "), ha="center", va="center",
                color="#ffffff" if c != viz.CAT[3] else viz.INK, fontsize=9)
    ax.set_yticks([])
    ax.grid(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.spines["left"].set_visible(False)
    ax.set_title("Split chronologique 70 / 10 / 20 — nombre de fenêtres 12→12 par bloc")
    viz.save(fig, OUT / "eda_split.png")


def main():
    viz.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_metr_la()
    _, adj = load_adjacency()
    loc = load_locations()
    sensor_map(df, adj, loc)
    speed_heatmap(df, adj)
    daily_profile(df)
    missing(df)
    adjacency(adj)
    best = propagation(df, adj)
    med, npairs = indistinguishability(df)
    split_timeline(df)
    print(f"propagation : {len(best)} arêtes, lag médian {np.median(best) * 5:.0f} min")
    print(f"indistinguabilité : {npairs} paires proches, écart médian à 60 min {med:.1f} mph")


if __name__ == "__main__":
    main()
