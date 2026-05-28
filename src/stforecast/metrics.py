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


def masked_rmse(pred, target, null_val=0.0):
    m = _mask(target, null_val)
    return torch.nan_to_num((pred - target) ** 2 * m).mean().sqrt()


def masked_mape(pred, target, null_val=0.0):
    m = _mask(target, null_val)
    ape = (pred - target).abs() / target.abs().clamp_min(1e-8)
    return torch.nan_to_num(ape * m).mean() * 100.0


def all_metrics(pred, target) -> dict[str, float]:
    return {
        "mae": masked_mae(pred, target).item(),
        "rmse": masked_rmse(pred, target).item(),
        "mape": masked_mape(pred, target).item(),
    }


def horizon_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, dict[str, float]]:
    """pred, target : (B, out_len, N). Retourne métriques à 15/30/60 min + moyenne."""
    out = {name: all_metrics(pred[:, h - 1], target[:, h - 1]) for name, h in HORIZONS.items()}
    out["avg"] = all_metrics(pred, target)
    return out


def per_step_metrics(pred, target) -> list[dict[str, float]]:
    return [all_metrics(pred[:, i], target[:, i]) for i in range(pred.shape[1])]
