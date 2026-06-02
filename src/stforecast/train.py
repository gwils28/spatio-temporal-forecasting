"""Entraînement / évaluation d'un modèle à partir d'une config YAML.

    uv run python -m stforecast.train --config configs/stid.yaml

Écrit dans `results/<run>/` : metrics.json (test par horizon), history.json (courbes),
cost.json (paramètres, s/époque, latence) ; checkpoint et prédictions dans
`artifacts/<run>/` (non versionné).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import build_data
from .graph import double_transition
from .metrics import horizon_metrics, masked_mae, per_step_metrics
from .models import MODELS


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
