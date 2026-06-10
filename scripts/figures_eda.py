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
