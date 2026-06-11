"""Construit le notebook d'exploration (cellules versionnées ici, exécuté par nbconvert)."""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Exploration de METR-LA, pas à pas\n\nCe notebook reproduit l'analyse exploratoire du site "
       "et ajoute deux questions : **la corrélation décroît-elle avec la distance routière ?** et "
       "**quelle part de la variance est expliquée par l'identité (capteur × créneau) seule ?** "
       "Cette dernière borne ce qu'un modèle sans graphe peut capturer."),
    code("import os, sys\nfrom pathlib import Path\nroot = Path.cwd()\nwhile not (root / 'pyproject.toml').exists():\n"
         "    root = root.parent\nos.chdir(root)\nimport numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
         "from stforecast import viz\nfrom stforecast.data import load_metr_la, load_adjacency, time_features\n"
         "viz.setup()\ndf = load_metr_la()\nids, adj = load_adjacency()\ndf.shape, df.index.min(), df.index.max()"),
    md("## 1. Statistiques de base"),
    code("s = df.replace(0, np.nan)\npd.DataFrame({'vitesse moyenne (mph)': [s.stack().mean()],\n"
         "              'écart-type': [s.stack().std()],\n              '% manquants': [(df == 0).to_numpy().mean() * 100],\n"
         "              'capteurs > 20 % manquants': [((df == 0).mean() > 0.2).sum()]}).round(2)"),
    md("## 2. Corrélation en fonction de la distance routière\n\nSi le graphe porte du signal, deux capteurs "
       "proches *sur la route* doivent être plus corrélés que deux capteurs lointains. On utilise les "
       "distances du dépôt DCRNN (`distances_la_2012.csv`)."),
    code("dist = pd.read_csv('data/raw/distances_la_2012.csv', dtype={'from': str, 'to': str})\n"
         "pos = {k: i for i, k in enumerate(ids)}\ndist = dist[dist['from'].isin(pos) & dist['to'].isin(pos)]\n"
         "x = s.interpolate(limit_direction='both').diff().dropna().to_numpy()\n"
         "corr = np.corrcoef(x.T)\n"
         "dist['corr'] = [corr[pos[a], pos[b]] for a, b in zip(dist['from'], dist['to'])]\n"
         "dist = dist[dist['from'] != dist['to']]\n"
         "bins = pd.cut(dist['cost'] / 1609, [0, 1, 2, 4, 8])\n"
         "g = dist.groupby(bins, observed=True)['corr'].agg(['median', 'count'])\ng"),
    code("fig, ax = plt.subplots(figsize=(7, 3.5))\nax.plot(range(len(g)), g['median'], marker='o', color=viz.CAT[0])\n"
         "ax.set_xticks(range(len(g)), [str(i) for i in g.index], rotation=20)\n"
         "ax.set_xlabel('Distance routière (miles)')\nax.set_ylabel('Corrélation médiane des Δvitesse')\n"
         "ax.set_title('Le signal spatial décroît vite avec la distance');"),
    md("**Lecture :** la corrélation des variations (à 5 min, donc bruitées) est nettement plus élevée à "
       "moins d'un mile (≈ 0,17) qu'au-delà (≈ 0,01-0,02). Le signal spatial est **très local**. Une diffusion d'ordre 1-2 suffit, ce qui "
       "explique pourquoi des graphes profonds n'apportent que peu."),
    md("## 3. Quelle part de la variance l'identité explique-t-elle seule ?\n\nOn prédit chaque vitesse par "
       "la moyenne de son (capteur, créneau de la semaine) sur le train, puis on mesure le $R^2$ sur le test. "
       "C'est l'information d'**identité pure** que STID encode dans ses embeddings."),
    code("tod, dow = time_features(df.index)\nslot = dow * 288 + tod\nn_tr = int(len(df) * 0.7)\n"
         "v = df.to_numpy()\ntr = pd.DataFrame(v[:n_tr]).replace(0, np.nan).groupby(slot[:n_tr]).mean()\n"
         "pred = tr.reindex(slot[n_tr:]).to_numpy()\ny = v[n_tr:]\nm = (y > 0) & ~np.isnan(pred)\n"
         "r2 = 1 - ((pred[m] - y[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()\n"
         "print(f'R² identité seule (capteur × créneau) : {r2:.3f}')"),
    md("L'identité seule explique déjà une grande part de la variance. Le reste vient de la dynamique "
       "récente (la fenêtre d'entrée) et, éventuellement, des voisins (le graphe)."),
    md("## 4. Indistinguabilité : combien de fenêtres jumelles ?\n\nOn tire des fenêtres (capteur, instant) "
       "au hasard. Parmi les paires dont les entrées sont quasi identiques (RMS < 1 mph), quelle part "
       "a des futurs qui divergent de plus de 10 mph à 60 minutes ?"),
    code("rng = np.random.default_rng(1)\nn = 6000\nt = rng.integers(0, n_tr - 24, n)\nk = rng.integers(0, v.shape[1], n)\n"
         "xin = v[t[:, None] + np.arange(12), k[:, None]]\nyout = v[t + 23, k]\nok = (xin > 0).all(1) & (yout > 0)\n"
         "xin, yout = xin[ok], yout[ok]\nd_in = np.sqrt(((xin[:, None] - xin[None]) ** 2).mean(-1))\n"
         "iu = np.triu_indices(len(xin), 1)\nclose = d_in[iu] < 1.0\ngap = np.abs(yout[:, None] - yout[None])[iu][close]\n"
         "print(f'{close.sum():,} paires jumelles ; {(gap > 10).mean() * 100:.1f} % divergent de > 10 mph à 60 min')"),
    md("Une petite fraction de jumelles diverge fortement. Ce sont **précisément les cas difficiles**, "
       "le début d'une congestion, où l'identité (quel capteur ? quelle heure ?) fait la différence "
       "entre « ça reste fluide » et « ça bouchonne »."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                               "language": "python"},
                                                "title": "Exploration (notebook)"})
out = Path("site/experience/exploration.ipynb")
nbf.write(nb, out)
print(out)
