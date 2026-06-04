import numpy as np
import pytest
import torch

from stforecast.graph import double_transition, k_hop_reach, random_walk
from stforecast.models import STID, GraphWaveNet

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


@pytest.mark.parametrize("n_sup,adaptive", [(2, True), (2, False), (0, True), (0, False)])
def test_gwnet_shapes(n_sup, adaptive):
    x, *_ = _inputs()
    sup = [torch.tensor(a) for a in double_transition(np.random.rand(N, N).astype(np.float32))][:n_sup]
    m = GraphWaveNet(num_nodes=N, supports=sup, adaptive=adaptive)
    assert m.receptive_field == 13
    assert m(x).shape == (B, 12, N)


def test_gwnet_nograph_has_no_spatial_mixing():
    """Sans graphe, perturber le capteur 0 ne doit pas changer la prédiction du capteur 1."""
    m = GraphWaveNet(num_nodes=N, supports=[], adaptive=False).eval()
    x, *_ = _inputs()
    x2 = x.clone()
    x2[:, :, 0] += 5
    assert torch.allclose(m(x)[:, :, 1], m(x2)[:, :, 1], atol=1e-6)
