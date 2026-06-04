"""Registre des modèles comparés."""

from .gwnet import GraphWaveNet
from .stid import STID

MODELS = {"stid": STID, "gwnet": GraphWaveNet}

__all__ = ["MODELS", "STID", "GraphWaveNet"]
