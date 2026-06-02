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


@torch.no_grad()
def predict(model, loader, scaler, device):
    model.eval()
    preds, ys = [], []
    for x, y, tod, dow in loader:
        x, tod, dow = x.to(device), tod.to(device), dow.to(device)
        out = scaler.inverse_transform(_forward(model, x, tod, dow))
        preds.append(out.cpu())
        ys.append(y)
    return torch.cat(preds), torch.cat(ys)


@torch.no_grad()
def measure_latency(model, loader, device, n_iter=50):
    """Latence d'inférence (ms) pour un batch de taille 1 (dispatch temps réel) et 64."""
    model.eval()
    x, _, tod, dow = next(iter(loader))
    res = {}
    for bs in (1, 64):
        xb, tb, db = x[:bs].to(device), tod[:bs].to(device), dow[:bs].to(device)
        for _ in range(10):
            _forward(model, xb, tb, db)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _forward(model, xb, tb, db)
        if device.type == "cuda":
            torch.cuda.synchronize()
        res[f"latency_ms_bs{bs}"] = (time.perf_counter() - t0) / n_iter * 1000
    return res


def train(cfg: dict, out_root: Path = Path("results"), art_root: Path = Path("artifacts")) -> dict:
    run = cfg["name"]
    set_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc = cfg["train"]
    bundle = build_data(batch_size=tc["batch_size"])
    model = build_model(cfg, bundle, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{run}] {n_params:,} paramètres sur {device}")

    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0))
    sched = None
    if tc.get("milestones"):
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, tc["milestones"], tc.get("gamma", 0.5))
    scaler = bundle.scaler
    out_dir, art_dir = out_root / run, art_root / run
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    best_val, best_epoch, patience = float("inf"), -1, tc.get("patience", 30)
    history, epoch_times = [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        t0 = time.perf_counter()
        losses = []
        for x, y, tod, dow in bundle.loaders["train"]:
            x, y, tod, dow = x.to(device), y.to(device), tod.to(device), dow.to(device)
            pred = scaler.inverse_transform(_forward(model, x, tod, dow))
            loss = masked_mae(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tc.get("clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip"])
            opt.step()
            losses.append(loss.item())
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)
        if sched:
            sched.step()
        vp, vy = predict(model, bundle.loaders["val"], scaler, device)
        val_mae = masked_mae(vp, vy).item()
        history.append({"epoch": epoch, "train_mae": float(np.mean(losses)), "val_mae": val_mae,
                        "epoch_s": epoch_times[-1]})
        flag = ""
        if val_mae < best_val:
            best_val, best_epoch, flag = val_mae, epoch, " *"
            torch.save(model.state_dict(), art_dir / "best.pt")
        print(f"[{run}] ep {epoch:3d} train {history[-1]['train_mae']:.3f} "
              f"val {val_mae:.3f} ({epoch_times[-1]:.1f}s){flag}", flush=True)
        if epoch - best_epoch >= patience:
            print(f"[{run}] early stopping (meilleure époque {best_epoch})")
            break

    model.load_state_dict(torch.load(art_dir / "best.pt", map_location=device))
    tp, ty = predict(model, bundle.loaders["test"], scaler, device)
    metrics = horizon_metrics(tp, ty)
    metrics["per_step"] = per_step_metrics(tp, ty)
    np.savez_compressed(art_dir / "test_pred.npz", pred=tp.numpy().astype(np.float16))
    cost = {
        "params": n_params,
        "epoch_s_mean": float(np.mean(epoch_times)),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "train_time_to_best_s": float(np.sum(epoch_times[:best_epoch])),
        "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        **measure_latency(model, bundle.loaders["test"], device),
    }
    if getattr(model, "adaptive", False):
        np.save(art_dir / "adaptive_adj.npy", model.adaptive_adj().detach().cpu().numpy())
    if hasattr(model, "node_emb"):
        np.save(art_dir / "node_emb.npy", model.node_emb.detach().cpu().numpy())
        if model.use_tod:
            np.save(art_dir / "tod_emb.npy", model.tod_emb.detach().cpu().numpy())
        if model.use_dow:
            np.save(art_dir / "dow_emb.npy", model.dow_emb.detach().cpu().numpy())
    for fname, obj in [("metrics.json", metrics), ("history.json", history), ("cost.json", cost),
                       ("config.json", cfg)]:
        (out_dir / fname).write_text(json.dumps(obj, indent=2))
    h = metrics
    print(f"[{run}] TEST MAE 15/30/60 = {h['15min']['mae']:.2f} / {h['30min']['mae']:.2f} / "
          f"{h['60min']['mae']:.2f}")
    return metrics
