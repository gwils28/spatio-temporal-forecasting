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


def build_model(cfg: dict, bundle, device) -> torch.nn.Module:
    name = cfg["model"]
    params = dict(cfg.get("model_params", {}))
    if name == "gwnet":
        graph = params.pop("graph", "road+adaptive")
        supports = []
        if "road" in graph:
            supports = [torch.tensor(a) for a in double_transition(bundle.adj)]
        params["supports"] = supports
        params["adaptive"] = "adaptive" in graph
    return MODELS[name](num_nodes=bundle.num_nodes, **params).to(device)


def _forward(model, x, tod, dow):
    return model(x, tod=tod, dow=dow)
