import torch

from stforecast.metrics import horizon_metrics, masked_mae, masked_mape, masked_rmse


def test_masked_mae_ignores_zeros():
    target = torch.tensor([[10.0, 0.0], [20.0, 30.0]])
    pred = torch.tensor([[12.0, 999.0], [20.0, 27.0]])
    assert torch.isclose(masked_mae(pred, target), torch.tensor(5.0 / 3))


def test_perfect_prediction_is_zero():
    t = torch.rand(8, 12, 5) * 50 + 1
    assert masked_mae(t, t) == 0 and masked_rmse(t, t) == 0 and masked_mape(t, t) == 0


def test_mape_percent():
    assert torch.isclose(masked_mape(torch.tensor([11.0]), torch.tensor([10.0])), torch.tensor(10.0))


def test_horizon_keys_and_index():
    target = torch.ones(2, 12, 3) * 50
    pred = target.clone()
    pred[:, 11] += 6  # erreur seulement au pas 12 (60 min)
    m = horizon_metrics(pred, target)
    assert set(m) == {"15min", "30min", "60min", "avg"}
    assert m["15min"]["mae"] == 0 and abs(m["60min"]["mae"] - 6) < 1e-6
