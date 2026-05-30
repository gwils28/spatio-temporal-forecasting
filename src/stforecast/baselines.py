"""Baselines sans apprentissage : bornes basses que tout modèle doit battre."""

from __future__ import annotations

import numpy as np

from .data import STEPS_PER_DAY


def historical_average(speed: np.ndarray, tod: np.ndarray, dow: np.ndarray,
                       train_end: int) -> np.ndarray:
    """Moyenne historique par (créneau de la semaine, capteur), zéros exclus.

    Retourne un tableau (7*288, N) indexé par dow*288 + tod.
    """
    slot = dow * STEPS_PER_DAY + tod
    n_slots = 7 * STEPS_PER_DAY
    s, v = slot[:train_end], speed[:train_end]
    valid = (v > 0).astype(np.float64)
    sums = np.zeros((n_slots, speed.shape[1]))
    counts = np.zeros((n_slots, speed.shape[1]))
    np.add.at(sums, s, v * valid)
    np.add.at(counts, s, valid)
    node_mean = (v * valid).sum(0) / np.maximum(valid.sum(0), 1)
    table = np.where(counts > 0, sums / np.maximum(counts, 1), node_mean[None, :])
    return table.astype(np.float32)


def predict_ha(table: np.ndarray, tod: np.ndarray, dow: np.ndarray, starts: np.ndarray,
               in_len: int = 12, out_len: int = 12) -> np.ndarray:
    """(n_windows, out_len, N) : profil hebdomadaire lu aux instants cibles."""
    idx = starts[:, None] + in_len + np.arange(out_len)[None, :]
    return table[dow[idx] * STEPS_PER_DAY + tod[idx]]


def predict_last_value(speed: np.ndarray, starts: np.ndarray, in_len: int = 12,
                       out_len: int = 12) -> np.ndarray:
    """Persistance : dernière valeur *non nulle* observée dans la fenêtre, répétée."""
    win = speed[starts[:, None] + np.arange(in_len)[None, :]]  # (W, L, N)
    last = np.zeros((win.shape[0], win.shape[2]), dtype=np.float32)
    for k in range(in_len):
        v = win[:, k]
        last = np.where(v > 0, v, last)
    return np.repeat(last[:, None, :], out_len, axis=1)
