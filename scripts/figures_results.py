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
