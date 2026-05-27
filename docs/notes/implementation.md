# Notes d'implémentation — déviations et choix conservateurs

Ce fichier consigne ce qui s'écarte de l'esquisse initiale, et pourquoi. Un écart non
consigné est un écart caché.

## Lecture de METR-LA : h5py au lieu de pandas

`pd.read_hdf("metr-la.h5")` échoue avec pandas 3 :
`TypeError: a bytes-like object is required, not 'str'`. Le fichier a été écrit en 2019
au format *fixed* de PyTables. On lit directement les blocs `df/block0_values`,
`df/axis0` (ids capteurs) et `df/axis1` (horodatages en ns) via `h5py`. On vérifie que
l'ordre des colonnes est identique à celui de `adj_mx.pkl` (assertion dans `build_data`).
