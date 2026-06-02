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
