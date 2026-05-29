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
