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


class WindowDataset(Dataset):
    """Fenêtres glissantes paresseuses : aucune copie (T, 12, N) en mémoire.

    x : (in_len, N, 3) = [vitesse normalisée, tod/288, dow/7]
    y : (out_len, N)   = vitesse brute (mph), 0 = manquant
    tod, dow : indices entiers du dernier pas d'entrée (identités STID)
    """

    def __init__(self, speed, tod, dow, starts, scaler, in_len=12, out_len=12):
        self.speed = torch.as_tensor(speed, dtype=torch.float32)
        self.speed_n = torch.as_tensor(scaler.transform(speed), dtype=torch.float32)
        self.tod = torch.as_tensor(tod)
        self.dow = torch.as_tensor(dow)
        self.starts = np.asarray(starts)
        self.in_len, self.out_len = in_len, out_len
        n = speed.shape[1]
        self._tod_f = (self.tod.float() / STEPS_PER_DAY)[:, None].expand(-1, n)
        self._dow_f = (self.dow.float() / 7.0)[:, None].expand(-1, n)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = int(self.starts[i])
        a, b, c = t, t + self.in_len, t + self.in_len + self.out_len
        x = torch.stack([self.speed_n[a:b], self._tod_f[a:b], self._dow_f[a:b]], dim=-1)
        y = self.speed[b:c]
        return x, y, self.tod[b - 1], self.dow[b - 1]
