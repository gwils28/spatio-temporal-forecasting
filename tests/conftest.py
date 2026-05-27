import numpy as np
import pytest

from stforecast.data import StandardScaler


@pytest.fixture
def toy_series():
    """Série jouet (T=100, N=4) avec trous (zéros) et horodatages réguliers."""
    rng = np.random.default_rng(0)
    speed = rng.uniform(20, 70, size=(100, 4)).astype(np.float32)
    speed[10:15, 2] = 0.0
    tod = np.arange(100) % 288
    dow = (np.arange(100) // 288) % 7
    return speed, tod, dow, StandardScaler(float(speed.mean()), float(speed.std()))
