"""Chargement de METR-LA, fenêtrage 12→12 et features d'identité temporelle.

Protocole aligné sur DCRNN / Graph WaveNet / BasicTS :
- split chronologique 70 / 10 / 20 sur les fenêtres ;
- normalisation z-score ajustée sur les entrées d'entraînement ;
- les zéros sont des valeurs manquantes (capteur muet), masquées dans la perte.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

STEPS_PER_DAY = 288  # 24 h à 5 min
DEFAULT_RAW = Path("data/raw")


def load_metr_la(path: str | Path = DEFAULT_RAW / "metr-la.h5") -> pd.DataFrame:
    """Lit le HDF5 d'origine (format pandas 2019) via h5py.

    Retourne un DataFrame (T, N) indexé par horodatage, colonnes = ids capteurs.
    """
    with h5py.File(path, "r") as f:
        g = f["df"]
        values = g["block0_values"][:].astype(np.float32)
        cols = [c.decode() for c in g["axis0"][:]]
        index = pd.to_datetime(g["axis1"][:])
    return pd.DataFrame(values, index=index, columns=cols)


def load_adjacency(path: str | Path = DEFAULT_RAW / "adj_mx.pkl") -> tuple[list[str], np.ndarray]:
    """Adjacence routière de DCRNN : noyau gaussien sur distances réseau, seuillé."""
    with open(path, "rb") as f:
        sensor_ids, _, adj = pickle.load(f, encoding="latin1")
    return list(sensor_ids), adj.astype(np.float32)


def load_locations(path: str | Path = DEFAULT_RAW / "graph_sensor_locations.csv") -> pd.DataFrame:
    """Coordonnées (lat, lon) des capteurs, dans l'ordre du fichier."""
    df = pd.read_csv(path)
    df["sensor_id"] = df["sensor_id"].astype(str)
    return df


def time_features(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Identités temporelles : créneau dans la journée (0..287) et jour de semaine (0..6)."""
    tod = (index.hour * 60 + index.minute) // 5
    dow = index.dayofweek
    return np.asarray(tod, dtype=np.int64), np.asarray(dow, dtype=np.int64)


@dataclass
class StandardScaler:
    """Z-score scalaire (une moyenne / un écart-type pour tout le réseau)."""

    mean: float
    std: float

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


def make_windows(n_steps: int, in_len: int = 12, out_len: int = 12) -> np.ndarray:
    """Indices de début des fenêtres valides : entrée [t, t+in), cible [t+in, t+in+out)."""
    return np.arange(0, n_steps - in_len - out_len + 1)


def split_indices(starts: np.ndarray, ratios=(0.7, 0.1, 0.2)) -> dict[str, np.ndarray]:
    """Split chronologique des fenêtres (pas de mélange : pas de fuite du futur)."""
    n = len(starts)
    n_train = round(n * ratios[0])
    n_test = round(n * ratios[2])
    n_val = n - n_train - n_test
    return {
        "train": starts[:n_train],
        "val": starts[n_train : n_train + n_val],
        "test": starts[n_train + n_val :],
    }
