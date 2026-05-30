"""Matrices de transition pour la convolution par diffusion (DCRNN / Graph WaveNet)."""

from __future__ import annotations

import numpy as np


def random_walk(adj: np.ndarray) -> np.ndarray:
    """P = D_O^{-1} W : transition d'une marche aléatoire *aval* (lignes stochastiques)."""
    d = adj.sum(1)
    d_inv = np.where(d > 0, 1.0 / np.maximum(d, 1e-12), 0.0)
    return (d_inv[:, None] * adj).astype(np.float32)


def double_transition(adj: np.ndarray) -> list[np.ndarray]:
    """Marches avant (W) et arrière (W^T) : la congestion remonte *et* descend le flux."""
    return [random_walk(adj), random_walk(adj.T)]
