"""Figures pédagogiques : diffusion sur graphe, frise du domaine."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stforecast import viz
from stforecast.data import load_adjacency, load_locations
from stforecast.graph import random_walk

OUT = Path("site/figures")
