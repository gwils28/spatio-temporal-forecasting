import numpy as np
import pandas as pd

from stforecast.data import WindowDataset, make_windows, split_indices, time_features


def test_make_windows_count():
    assert len(make_windows(100, 12, 12)) == 100 - 24 + 1
