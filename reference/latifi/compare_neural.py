"""Compare the neural LASTFM runs (results/latifi_lastfm/neural/<model>/seed<k>/) with Latifi et al. 2021, Table 6.

Per seed: mean over the five slices (as in the paper). Printed: paper value, mean over seeds, [min, max] over seeds,
and whether the paper value lies inside that range. The paper reports one run per model.
usage: python reference/latifi/compare_neural.py
"""
import glob
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'results/latifi_lastfm/neural')
COLS = ['MAP@20', 'Precision@20', 'Recall@20', 'HitRate@20', 'MRR@20', 'Coverage@20', 'Popularity@20']
SHORT = ['MAP', 'P', 'R', 'HR', 'MRR', 'COV', 'POP']
# Latifi et al. 2021 (arXiv 2011.03424 v2), Table 6, LASTFM: MAP P R HR MRR COV POP @20
PAPER = {
    'iirnn': [0.0311, 0.0724, 0.3270, 0.4729, 0.3491, 0.4443, 0.0732],
    'gru4rec': [0.0307, 0.0701, 0.3175, 0.4681, 0.3342, 0.5124, 0.0468],
    'nsar': [0.0280, 0.0675, 0.3044, 0.4350, 0.2906, 0.4937, 0.0483],
    'narm': [0.0272, 0.0658, 0.3002, 0.4510, 0.3192, 0.4748, 0.0633],
    'ncsf': [0.0219, 0.0556, 0.2639, 0.4393, 0.2434, 0.4875, 0.0712],
    'hgru4rec': [0.0208, 0.0517, 0.2464, 0.4206, 0.3167, 0.5647, 0.0404],
    'shan': [0.0072, 0.0223, 0.0920, 0.1149, 0.0345, 0.0824, 0.1640],
}


def seed_means(model):
    out = {}
    for d in sorted(glob.glob(f'{RES}/{model}/seed*')):
        files = sorted(glob.glob(f'{d}/test_window_{model}_lastfm.[0-9].csv'))
        if len(files) != 5:
            continue
        per = []
        for f in files:
            t = pd.read_csv(f, sep=';', index_col=0)
            t.columns = [c.strip().rstrip(':').strip() for c in t.columns]
            per.append(t.iloc[0][COLS].astype(float))
        out[os.path.basename(d)] = pd.concat(per, axis=1).mean(axis=1)
    return pd.DataFrame(out)


print('LASTFM @20, per seed the mean of 5 slices.  paper | port mean [min, max]  (* = paper outside range)')
for model, ref in PAPER.items():
    s = seed_means(model)
    if s.empty:
        print(f'{model:9} no complete seed yet')
        continue
    cells = []
    for c, short, p in zip(COLS, SHORT, ref):
        lo, hi, m = s.loc[c].min(), s.loc[c].max(), s.loc[c].mean()
        cells.append(f'{short} {p:.4f}|{m:.4f}[{lo:.4f},{hi:.4f}]{"*" if not lo <= p <= hi else " "}')
    print(f'{model:9} n={s.shape[1]} ' + '  '.join(cells))
