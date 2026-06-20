"""Charte graphique commune (palette catégorielle validée CVD, encre, grille discrète)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# Ordre catégoriel fixe (jamais cyclé) : bleu, orange, aqua, jaune, magenta, vert, violet, rouge
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b52b2a"])

# Couleur par entité (stable d'une figure à l'autre) — la couleur suit le modèle, pas son rang.
MODEL_COLORS = {
    "stid": CAT[0], "gwnet": CAT[1], "ha": MUTED, "last_value": AXIS,
    "gwnet_road": CAT[3], "gwnet_adaptive": CAT[2], "gwnet_nograph": CAT[4],
    "stid_no_node": CAT[6], "stid_no_time": CAT[5], "stid_no_ids": CAT[7],
}
MODEL_LABELS = {
    "ha": "Moyenne historique", "last_value": "Persistance", "stid": "STID",
    "stid_no_node": "STID sans id. spatiale", "stid_no_time": "STID sans id. temporelles",
    "stid_no_ids": "MLP sans identités", "gwnet": "GWNet (route + adaptative)",
    "gwnet_road": "GWNet (route)", "gwnet_adaptive": "GWNet (adaptative)",
    "gwnet_nograph": "GWNet sans graphe (TCN)",
}


def setup() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 2, "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=CAT),
    })


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)


def spread_labels(values, min_gap):
    """Positions verticales d'étiquettes de fin de ligne, écartées d'au moins `min_gap`."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = [0.0] * len(values)
    last = -float("inf")
    for i in order:
        pos[i] = max(values[i], last + min_gap)
        last = pos[i]
    return pos


def plain_log_ticks(ax, axis="x"):
    """Graduations lisibles (1, 2, 5, 10…) sur un axe logarithmique."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())
