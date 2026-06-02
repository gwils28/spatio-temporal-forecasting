import pytest
import torch

from stforecast.models import STID

B, L, N = 4, 12, 6


def _inputs():
    x = torch.randn(B, L, N, 3)
    return x, torch.randint(0, 288, (B,)), torch.randint(0, 7, (B,))


@pytest.mark.parametrize("flags", [{}, {"use_node": False}, {"use_tod": False, "use_dow": False},
                                   {"use_node": False, "use_tod": False, "use_dow": False}])
def test_stid_shapes(flags):
    x, tod, dow = _inputs()
    assert STID(num_nodes=N, **flags)(x, tod, dow).shape == (B, 12, N)


def test_stid_node_identity_breaks_symmetry():
    """Même fenêtre sur deux capteurs : sans identité → même prédiction ; avec → différente."""
    x, tod, dow = _inputs()
    x[:, :, 1] = x[:, :, 0]
    plain = STID(num_nodes=N, use_node=False, use_tod=False, use_dow=False).eval()
    ided = STID(num_nodes=N).eval()
    p, q = plain(x, tod, dow), ided(x, tod, dow)
    assert torch.allclose(p[:, :, 0], p[:, :, 1])
    assert not torch.allclose(q[:, :, 0], q[:, :, 1])
