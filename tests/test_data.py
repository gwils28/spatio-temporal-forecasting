import numpy as np
import pandas as pd

from stforecast.data import WindowDataset, make_windows, split_indices, time_features


def test_make_windows_count():
    assert len(make_windows(100, 12, 12)) == 100 - 24 + 1


def test_split_is_chronological_and_disjoint():
    s = split_indices(make_windows(1000))
    assert s["train"].max() < s["val"].min() < s["val"].max() < s["test"].min()
    assert sum(len(v) for v in s.values()) == len(make_windows(1000))


def test_time_features_ranges():
    idx = pd.date_range("2012-03-01", periods=2016, freq="5min")  # une semaine
    tod, dow = time_features(idx)
    assert tod.min() == 0 and tod.max() == 287
    assert set(dow) == set(range(7))
    assert tod[12] == 12  # 01:00 → 12e créneau


def test_window_alignment(toy_series):
    speed, tod, dow, scaler = toy_series
    ds = WindowDataset(speed, tod, dow, np.array([5]), scaler)
    x, y, t, d = ds[0]
    assert x.shape == (12, 4, 3) and y.shape == (12, 4)
    np.testing.assert_allclose(y.numpy(), speed[17:29])
    np.testing.assert_allclose(scaler.inverse_transform(x[..., 0].numpy()), speed[5:17], rtol=1e-5)
    assert int(t) == tod[16]  # identité = dernier pas observé


def test_scaler_roundtrip(toy_series):
    speed, *_, scaler = toy_series
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(speed)), speed, rtol=1e-5)
