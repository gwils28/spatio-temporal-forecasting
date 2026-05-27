"""Métriques masquées : les zéros de la cible sont des trous capteurs, pas des vitesses."""

from __future__ import annotations

import torch

HORIZONS = {"15min": 3, "30min": 6, "60min": 12}  # pas de 5 min, indexés à partir de 1


def _mask(target: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    mask = (target != null_val).float()
    return mask / mask.mean().clamp_min(1e-8)


def masked_mae(pred, target, null_val=0.0):
    m = _mask(target, null_val)
    return torch.nan_to_num((pred - target).abs() * m).mean()
