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


def sym_norm(adj: np.ndarray) -> np.ndarray:
    """D^{-1/2} (W + I) D^{-1/2}, normalisation GCN (pour comparaison pédagogique)."""
    a = adj + np.eye(adj.shape[0], dtype=adj.dtype)
    d = a.sum(1)
    d_is = 1.0 / np.sqrt(d)
    return (d_is[:, None] * a * d_is[None, :]).astype(np.float32)


def k_hop_reach(adj: np.ndarray, k: int) -> np.ndarray:
    """Nombre de nœuds atteignables en ≤ k sauts (champ réceptif d'une diffusion d'ordre k)."""
    a = (adj > 0).astype(np.int64)
    reach = np.eye(adj.shape[0], dtype=np.int64)
    for _ in range(k):
        reach = ((reach + reach @ a) > 0).astype(np.int64)
    return reach.sum(1)
