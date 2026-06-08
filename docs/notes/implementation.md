# Notes d'implémentation — déviations et choix conservateurs

Ce fichier consigne ce qui s'écarte de l'esquisse initiale, et pourquoi. Un écart non
consigné est un écart caché.

## Lecture de METR-LA : h5py au lieu de pandas

`pd.read_hdf("metr-la.h5")` échoue avec pandas 3 :
`TypeError: a bytes-like object is required, not 'str'`. Le fichier a été écrit en 2019
au format *fixed* de PyTables. On lit directement les blocs `df/block0_values`,
`df/axis0` (ids capteurs) et `df/axis1` (horodatages en ns) via `h5py`. On vérifie que
l'ordre des colonnes est identique à celui de `adj_mx.pkl` (assertion dans `build_data`).

## Pas de `torch-geometric-temporal` ni de BasicTS

L'esquisse proposait `torch-geometric-temporal` ou BasicTS. Décision d'interview : PyTorch
pur, modèles réécrits. `torch-geometric-temporal` n'est plus activement maintenu, et
BasicTS est une boîte noire lourde. Les hyperparamètres restent ceux de BasicTS (STID) et
de l'article (Graph WaveNet). La reproduction de STID (2.80 / 3.17 / 3.54) recoupe les
chiffres publiés par BasicTS (≈ 2.82 / 3.19 / 3.55) : l'implémentation est validée.

## STID : entrée à 3 canaux

L'article STID n'utilise que la vitesse dans l'embedding de série. La configuration
BasicTS pour METR-LA passe `input_dim = 3` (vitesse, heure, jour), et nous suivons
BasicTS. Les identités temporelles sont lues au **dernier pas observé**, comme dans
BasicTS.

## Une seule graine

Le budget retenu est le protocole complet avec une graine. L'écart STID / GWNet étant
mince, on ne peut pas exclure qu'il soit dans le bruit inter-seed. C'est pour ça que le
critère de décision (SPEC) demande une marge de 3 % **et** une attribution au graphe via
l'ablation. Piste : 3 graines sur `stid` et `gwnet` uniquement (≈ 3 h).

## Entraînements en parallèle sur un GPU

Deux files d'entraînement partagent le GPU (RTX 5070 Laptop, 8 Go). Les temps par époque
absolus sont donc **pessimistes** (contention), mais les **ratios** STID / GWNet restent
comparables. La latence d'inférence est mesurée après l'entraînement, avec la même
contention possible. Elle est indicative.

## Graph WaveNet plutôt que DCRNN

DCRNN est le papier de lecture du jour, mais sa récurrence est environ 3 fois plus lente
que Graph WaveNet et 13 à 36 fois plus lente que STID, selon le tableau d'efficacité de
l'article STID. Graph WaveNet réutilise **exactement** le support de
diffusion bidirectionnel de DCRNN (`double_transition`), auquel il ajoute l'adjacence
apprise. C'est ce qui permet l'ablation « d'où vient le graphe ».
