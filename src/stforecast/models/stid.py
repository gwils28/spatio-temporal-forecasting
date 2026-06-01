"""STID — Spatial-Temporal IDentity (Shao et al., CIKM 2022), réécrit pour la lisibilité.

Idée : un MLP ne voit qu'une fenêtre de 12 valeurs. Deux capteurs (ou deux instants)
qui produisent la même fenêtre sont *indistinguables* pour lui, alors que leurs futurs
diffèrent. On lève l'ambiguïté en concaténant des **identités apprises** :
- E_node[n]   : « qui » (quel capteur — sa position implicite dans le réseau) ;
- T_tod[t]    : « quand dans la journée » (288 créneaux) ;
- T_dow[t]    : « quel jour de la semaine ».
Aucune adjacence n'est utilisée.
"""

from __future__ import annotations

import torch
from torch import nn
