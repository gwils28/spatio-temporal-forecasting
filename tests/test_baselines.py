import numpy as np

from stforecast.baselines import historical_average, predict_ha, predict_last_value


def test_last_value_skips_missing(toy_series):
    speed, *_ = toy_series
    speed = speed.copy()
    speed[16, 0] = 0.0
    p = predict_last_value(speed, np.array([5]))
    assert p.shape == (1, 12, 4)
    assert p[0, 0, 0] == speed[15, 0]
    assert np.all(p[0, :, 1] == speed[16, 1])


def test_historical_average_constant_profile():
    tod = np.tile(np.arange(288), 14)
    dow = np.repeat(np.arange(14) % 7, 288)
    speed = (tod[:, None] / 10 + 30).astype(np.float32) * np.ones((1, 2), np.float32)
    table = historical_average(speed, tod, dow, train_end=len(tod))
    p = predict_ha(table, tod, dow, np.array([0]))
    np.testing.assert_allclose(p[0, :, 0], speed[12:24, 0], rtol=1e-5)
