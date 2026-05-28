import torch

from stforecast.metrics import horizon_metrics, masked_mae, masked_mape, masked_rmse


def test_masked_mae_ignores_zeros():
    target = torch.tensor([[10.0, 0.0], [20.0, 30.0]])
    pred = torch.tensor([[12.0, 999.0], [20.0, 27.0]])
    assert torch.isclose(masked_mae(pred, target), torch.tensor(5.0 / 3))
