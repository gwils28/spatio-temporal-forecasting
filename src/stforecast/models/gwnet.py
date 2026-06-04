"""Graph WaveNet (Wu et al., IJCAI 2019), réécrit pour la lisibilité.

Bloc élémentaire = convolution temporelle dilatée à porte (TCN, façon WaveNet)
suivie d'une convolution par diffusion sur graphe. Deux sources de graphe :
- `supports`  : transitions avant/arrière de l'adjacence routière (comme DCRNN) ;
- `adaptive`  : Ã = softmax(ReLU(E1 E2)), **apprise** de bout en bout sans a priori.
`supports=[]` et `adaptive=False` donne un TCN pur : l'ablation « sans graphe ».
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def nconv(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Propagation sur graphe : x (B, C, N, T), a (N, N) → Σ_v x[..., v, :] a[v, w]."""
    return torch.einsum("ncvl,vw->ncwl", x, a).contiguous()


class DiffusionGCN(nn.Module):
    """Σ_k Σ_A A^k X W_{k,A} : diffusion d'ordre `order` sur chaque matrice de support."""

    def __init__(self, c_in, c_out, n_supports, order=2, dropout=0.3):
        super().__init__()
        self.order, self.dropout = order, dropout
        self.mlp = nn.Conv2d((order * n_supports + 1) * c_in, c_out, kernel_size=1)

    def forward(self, x, supports):
        out = [x]
        for a in supports:
            x1 = nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = nconv(x1, a)
                out.append(x1)
        h = self.mlp(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes: int, supports: list[torch.Tensor] | None = None,
                 adaptive: bool = True, in_dim: int = 2, out_len: int = 12,
                 residual_channels: int = 32, dilation_channels: int = 32,
                 skip_channels: int = 256, end_channels: int = 512, kernel_size: int = 2,
                 blocks: int = 4, layers: int = 2, adp_dim: int = 10, dropout: float = 0.3,
                 **_):
        super().__init__()
        self.in_dim = in_dim
        supports = list(supports or [])
        for i, s in enumerate(supports):
            self.register_buffer(f"support_{i}", s)
        self.n_fixed = len(supports)
        self.adaptive = adaptive
        if adaptive:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, adp_dim))
            self.nodevec2 = nn.Parameter(torch.randn(adp_dim, num_nodes))
        n_supports = self.n_fixed + int(adaptive)
        self.use_gcn = n_supports > 0

        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=1)
        self.filter_convs, self.gate_convs = nn.ModuleList(), nn.ModuleList()
        self.skip_convs, self.bn = nn.ModuleList(), nn.ModuleList()
        self.gconv, self.residual_convs = nn.ModuleList(), nn.ModuleList()
        receptive_field = 1
        for _ in range(blocks):
            dilation, extra = 1, kernel_size - 1
            for _ in range(layers):
                self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels,
                                                   (1, kernel_size), dilation=dilation))
                self.gate_convs.append(nn.Conv2d(residual_channels, dilation_channels,
                                                 (1, kernel_size), dilation=dilation))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, 1))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                if self.use_gcn:
                    self.gconv.append(DiffusionGCN(dilation_channels, residual_channels,
                                                   n_supports, dropout=dropout))
                else:
                    self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, 1))
                receptive_field += extra
                extra *= 2
                dilation *= 2
        self.receptive_field = receptive_field
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, 1)
        self.end_conv_2 = nn.Conv2d(end_channels, out_len, 1)

    def fixed_supports(self) -> list[torch.Tensor]:
        return [getattr(self, f"support_{i}") for i in range(self.n_fixed)]

    def adaptive_adj(self) -> torch.Tensor:
        return F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)

    def forward(self, x, **_):
        """x : (B, L, N, C) → (B, out_len, N)."""
        x = x[..., : self.in_dim].permute(0, 3, 2, 1)  # (B, C, N, L)
        if x.shape[-1] < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.shape[-1], 0))
        x = self.start_conv(x)
        supports = self.fixed_supports()
        if self.adaptive:
            supports = supports + [self.adaptive_adj()]
        skip = 0
        for i in range(len(self.filter_convs)):
            residual = x
            x = torch.tanh(self.filter_convs[i](residual)) * torch.sigmoid(self.gate_convs[i](residual))
            s = self.skip_convs[i](x)
            skip = s + (skip[..., -s.shape[-1]:] if torch.is_tensor(skip) else 0)
            x = self.gconv[i](x, supports) if self.use_gcn else self.residual_convs[i](x)
            x = self.bn[i](x + residual[..., -x.shape[-1]:])
        x = F.relu(self.end_conv_1(F.relu(skip)))
        return self.end_conv_2(x).squeeze(-1)  # (B, out_len, N)
