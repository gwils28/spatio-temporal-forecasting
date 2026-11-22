"""STID : Spatial-Temporal IDentity (Shao et al., CIKM 2022), réécrit pour la lisibilité.

Idée : un MLP ne voit qu'une fenêtre de 12 valeurs. Deux capteurs (ou deux instants)
qui produisent la même fenêtre sont *indistinguables* pour lui, alors que leurs futurs
diffèrent. On lève l'ambiguïté en concaténant des **identités apprises** :
- E_node[n]   : « qui » (quel capteur, sa position implicite dans le réseau) ;
- T_tod[t]    : « quand dans la journée » (288 créneaux) ;
- T_dow[t]    : « quel jour de la semaine ».
Aucune adjacence n'est utilisée.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualMLP(nn.Module):
    """Z ← FC2(Dropout(ReLU(FC1(Z)))) + Z, en convolution 1×1 sur l'axe des nœuds."""

    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.fc2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, z):
        return self.fc2(self.drop(self.act(self.fc1(z)))) + z


class STID(nn.Module):
    def __init__(self, num_nodes: int, in_len: int = 12, out_len: int = 12, input_dim: int = 3,
                 embed_dim: int = 32, node_dim: int = 32, tod_dim: int = 32, dow_dim: int = 32,
                 num_layers: int = 3, steps_per_day: int = 288, use_node: bool = True,
                 use_tod: bool = True, use_dow: bool = True, dropout: float = 0.15, **_):
        super().__init__()
        self.input_dim = input_dim
        self.use_node, self.use_tod, self.use_dow = use_node, use_tod, use_dow
        self.ts_embed = nn.Conv2d(in_len * input_dim, embed_dim, kernel_size=1)
        hidden = embed_dim
        if use_node:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, node_dim))
            nn.init.xavier_uniform_(self.node_emb)
            hidden += node_dim
        if use_tod:
            self.tod_emb = nn.Parameter(torch.empty(steps_per_day, tod_dim))
            nn.init.xavier_uniform_(self.tod_emb)
            hidden += tod_dim
        if use_dow:
            self.dow_emb = nn.Parameter(torch.empty(7, dow_dim))
            nn.init.xavier_uniform_(self.dow_emb)
            hidden += dow_dim
        self.encoder = nn.Sequential(*[ResidualMLP(hidden, dropout) for _ in range(num_layers)])
        self.regression = nn.Conv2d(hidden, out_len, kernel_size=1)
        self.hidden_dim = hidden

    def forward(self, x, tod, dow, **_):
        """x : (B, L, N, C) ; tod, dow : (B,) indices du dernier pas observé → (B, out_len, N)."""
        b, seq_len, n, _ = x.shape
        x = x[..., : self.input_dim]
        # (B, L, N, C) → (B, L*C, N, 1) : chaque nœud devient un vecteur « fenêtre ».
        h = x.permute(0, 2, 1, 3).reshape(b, n, seq_len * self.input_dim).transpose(1, 2).unsqueeze(-1)
        parts = [self.ts_embed(h)]
        if self.use_node:
            parts.append(self.node_emb.T.unsqueeze(0).expand(b, -1, -1).unsqueeze(-1))
        if self.use_tod:
            parts.append(self.tod_emb[tod].unsqueeze(-1).expand(-1, -1, n).unsqueeze(-1))
        if self.use_dow:
            parts.append(self.dow_emb[dow].unsqueeze(-1).expand(-1, -1, n).unsqueeze(-1))
        z = self.encoder(torch.cat(parts, dim=1))
        return self.regression(z).squeeze(-1)  # (B, out_len, N)
