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
