import torch

from stforecast.metrics import horizon_metrics, masked_mae, masked_mape, masked_rmse


def test_masked_mae_ignores_zeros():
    target = torch.tensor([[10.0, 0.0], [20.0, 30.0]])
    pred = torch.tensor([[12.0, 999.0], [20.0, 27.0]])
    assert torch.isclose(masked_mae(pred, target), torch.tensor(5.0 / 3))


def test_perfect_prediction_is_zero():
    t = torch.rand(8, 12, 5) * 50 + 1
    assert masked_mae(t, t) == 0 and masked_rmse(t, t) == 0 and masked_mape(t, t) == 0
