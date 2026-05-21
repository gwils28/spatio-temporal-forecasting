# Le graphe mérite-t-il sa complexité ?

**Prévision spatio-temporelle sur réseau de capteurs : STID (sans graphe) vs Graph WaveNet (STGNN) sur METR-LA.**
Travaux de veille et d'expérimentation, cas d'usage logistique (vitesses et temps de trajet sur un réseau de transport).

📖 **Site de documentation :** <https://gwils28.github.io/spatio-temporal-forecasting/>

## Question

Un STGNN exploite la corrélation spatiale (la congestion se propage le long du réseau).
Mais **le graphe est une hypothèse, pas une donnée** : avant de déployer un STGNN, il faut
montrer qu'il bat un baseline *non-graphe* bien conçu, d'une marge qui justifie son
surcoût en calcul, en latence et en dette technique.

Protocole et critère de décision fixés **avant** les résultats : [`SPEC.md`](SPEC.md).
