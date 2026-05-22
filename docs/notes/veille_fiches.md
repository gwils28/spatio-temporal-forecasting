# Fiches de veille — sources vérifiées

Fiches établies à partir des pages arXiv (abstract + version HTML).
Légende : **[V]** lu dans le texte, **[R]** seulement dans un résumé, à recouper
avec le PDF avant toute citation chiffrée.

## 1. Foundation Models for Spatio-Temporal Data Science: A Tutorial and Survey

- arXiv:2503.13502, 12 mars 2025 — Liang, Wen, Xia, Jin, Yang, Salim, Wen, Pan, Cong. [V]
- Thèse : les modèles ST profonds restent spécifiques à une tâche et gourmands en
  étiquettes ; les *Spatio-Temporal Foundation Models* (STFM) doivent couvrir tout le
  workflow : **sensing → management → mining**. [V]
- Taxonomie à deux axes [V] :
  - *Workflow* : sensing (données réelles, synthétiques), management (nettoyage,
    requête, intégration), mining (perception, optimisation, raisonnement) ;
    applications numériques (prévision, imputation, anomalies) et inférentielles
    (« What happened? », « Where is it? », « What to do? », « What if? »).
  - *Méthodologie* : LLM (zero-shot, fine-tuning supervisé) vs **PFM** —
    modèles pré-entraînés from scratch (architecture, schéma de pré-entraînement,
    modalité).
- Exemples cités [V] : STEP (KDD 2022) et GPT-ST (pré-entraînement agnostique au
  domaine), OpenCity et TFM (FM graphiques pour le transport), UniST (« one-for-all »),
  UrbanGPT (KDD 2024), Pangu (Nature 2023), Chronos, Time-MoE.
- Pas d'expérience propre : c'est une carte du domaine.

## 2. UrbanFM: Scaling Urban Spatio-Temporal Foundation Models

- arXiv:2602.20677, v1 24 fév. 2026, v2 2 mars 2026 — Chen, Wu, Chen, Zhou, Liang
  (HKUST(GZ), HKUST). [V]
- Trois axes de passage à l'échelle [V] :
  - **Données** : *WorldST*, > 1 milliard de points, 100+ villes, 8 domaines
    (flux, vitesse…) standardisés.
  - **Calcul** : *MiniST*, découpage des champs ST (grille ou capteurs) en unités
    d'apprentissage (clustering glouton à capacité fixe via KD-Tree, puis patchs
    spatio-temporels).
  - **Architecture** : self-attention minimaliste, factorisée temps puis espace,
    RoPE temporel et spatial, RevIN, objectif génératif à futur masqué.
- Benchmark *EvalST* ; METR-LA et PEMS-BAY en font partie. [R]
- Résultats annoncés [R, à confirmer] : zero-shot +39 % à +70 % vs FM existants ;
  few-shot (10 % des données) au-dessus des modèles experts entraînés sur 100 %.
- Limites [R] : coût d'ingénierie des grands modèles, peu de données grille,
  dégradation avec des contextes trop longs.

## 3. STID — Spatial-Temporal Identity

- arXiv:2208.05233, CIKM 2022 (short) — Shao, Zhang, Wang, Wei, Xu. [V]
- Diagnostic : l'**indistinguabilité des échantillons** en espace et en temps est le
  goulot ; les STGNN se complexifient pour des gains marginaux. [V]
- Modèle [V] : `Z = FC(X) ‖ E_node ‖ T_TiD ‖ T_DiW ∈ R^{4D}`, puis L blocs MLP
  résiduels `Z ← FC2(ReLU(FC1(Z))) + Z`, puis régression linéaire vers F pas.
  D = 32, L = 3, P = F = 12, perte MAE, Adam, lr = 1e-3.
- Datasets : PEMS04/07/08, PEMS-BAY, Electricity. **METR-LA n'est pas évalué dans
  l'article** ; les chiffres STID METR-LA courants proviennent de BasicTS.
- PEMS-BAY MAE @15/30/60 min [V] : STID 1.30/1.62/1.89 · GWNet 1.30/1.63/1.95 ·
  DCRNN 1.31/1.67/1.99.
- Efficacité (s/époque, PEMS-BAY) [V] : STID 15.8 · GWNet 112.0 · DCRNN 351.4.
- Ablation [V] : l'identité spatiale est la plus importante.

## 4. DCRNN — Diffusion Convolutional Recurrent Neural Network

- arXiv:1707.01926, ICLR 2018 — Li, Yu, Shahabi, Liu. [V]
- Convolution par diffusion [V] :
  `X ⋆_G f_θ = Σ_{k=0}^{K-1} (θ_{k,1} (D_O^{-1}W)^k + θ_{k,2} (D_I^{-1}W^T)^k) X`
  — marches aléatoires avant/arrière sur graphe **orienté**, coût O(K|E|).
- DCGRU [V] : les produits matriciels du GRU sont remplacés par la convolution par
  diffusion ; encodeur-décodeur seq2seq ; *scheduled sampling*.
- Données [V] : METR-LA, 207 capteurs, 1er mars → 30 juin 2012, pas de 5 min,
  split 70/10/20, z-score.
- METR-LA MAE/RMSE/MAPE [V] :

| Horizon | HA | FC-LSTM | DCRNN |
|---|---|---|---|
| 15 min | 4.16 / 7.80 / 13.0 % | 3.44 / 6.30 / 9.6 % | 2.77 / 5.38 / 7.3 % |
| 30 min | 4.16 / 7.80 / 13.0 % | 3.77 / 7.23 / 10.9 % | 3.15 / 6.45 / 8.8 % |
| 60 min | 4.16 / 7.80 / 13.0 % | 4.37 / 8.69 / 13.2 % | 3.60 / 7.59 / 10.5 % |

- Ablations [V] : sans diffusion, nettement pire ; bidirectionnel > unidirectionnel ;
  seq2seq + scheduled sampling > DCNN.
