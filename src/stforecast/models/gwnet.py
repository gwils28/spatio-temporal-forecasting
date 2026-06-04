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
