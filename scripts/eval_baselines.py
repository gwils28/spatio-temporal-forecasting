"""Évalue les baselines sans apprentissage (moyenne historique, persistance) sur le test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from stforecast.baselines import historical_average, predict_ha, predict_last_value
from stforecast.data import load_metr_la, make_windows, split_indices, time_features
from stforecast.metrics import horizon_metrics, per_step_metrics


def main(out_root: Path = Path("results")) -> None:
    df = load_metr_la()
    speed = df.to_numpy()
    tod, dow = time_features(df.index)
    splits = split_indices(make_windows(len(df)))
    test = splits["test"]
    y = speed[test[:, None] + 12 + np.arange(12)[None, :]]
    train_end = int(splits["train"][-1] + 12)
    preds = {
        "ha": predict_ha(historical_average(speed, tod, dow, train_end), tod, dow, test),
        "last_value": predict_last_value(speed, test),
    }
    for name, p in preds.items():
        m = horizon_metrics(torch.tensor(p), torch.tensor(y))
        m["per_step"] = per_step_metrics(torch.tensor(p), torch.tensor(y))
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps(m, indent=2))
        print(name, {k: round(v["mae"], 3) for k, v in m.items() if k != "per_step"})


if __name__ == "__main__":
    main()
