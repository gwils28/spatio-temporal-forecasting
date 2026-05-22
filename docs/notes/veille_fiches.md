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
