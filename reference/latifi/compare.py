"""Compare the LASTFM runs (results/latifi_lastfm/{orig,port}) with Latifi et al. 2021, Table 6.

Metrics are averaged over the five slices (as in the paper). Lists of orig and port are compared per slice
with reference/parity_lists.py (score-equivalence).
usage: python reference/latifi/compare.py [--no-lists]
"""
import glob
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'results/latifi_lastfm')
COLS = ['MAP@20', 'Precision@20', 'Recall@20', 'HitRate@20', 'MRR@20', 'Coverage@20', 'Popularity@20']
# Latifi et al. 2021 (arXiv 2011.03424 v2), Table 6, LASTFM: MAP P R HR MRR COV POP @20
PAPER = {
    'vsknn_eb': [0.0493, 0.1003, 0.4248, 0.5383, 0.1771, 0.5053, 0.0499],
    'vsknn': [0.0474, 0.0964, 0.4104, 0.5227, 0.1725, 0.5091, 0.0479],
    'vstan_eb': [0.0449, 0.0943, 0.4088, 0.5291, 0.1938, 0.5080, 0.0528],
    'stan_ebr': [0.0442, 0.0924, 0.3972, 0.5159, 0.1902, 0.4965, 0.0564],
    'stan': [0.0400, 0.0856, 0.3760, 0.5012, 0.1886, 0.5067, 0.0557],
    'vstan': [0.0394, 0.0852, 0.3791, 0.5100, 0.1935, 0.5115, 0.0555],
    'sr_b': [0.0359, 0.0778, 0.3408, 0.4622, 0.3487, 0.5010, 0.0549],
    'sr': [0.0348, 0.0765, 0.3383, 0.4615, 0.3350, 0.4982, 0.0544],
}
PAPER['vstan_ipw4'] = PAPER['vstan']
PAPER['vstan_eb_spw'] = PAPER['vstan_eb']  # lambda_spw 1.4525 instead of Table A.14's 0.4525


def load(stack, row):
    files = sorted(glob.glob(f'{RES}/{stack}/{row}/test_window_{row}_lastfm.[0-9].csv'))
    if len(files) != 5:
        return None, len(files)
    per = []
    for f in files:
        d = pd.read_csv(f, sep=';', index_col=0)
        d.columns = [c.strip().rstrip(':').strip() for c in d.columns]
        per.append(d.iloc[0][COLS].astype(float))
    return pd.concat(per, axis=1).mean(axis=1), 5


def main():
    out = []
    for row in PAPER:
        r = {'row': row}
        for stack in ('orig', 'port'):
            m, n = load(stack, row)
            r[stack] = m if m is not None else f'{n}/5 slices'
        out.append(r)
    short = ['MAP', 'P', 'R', 'HR', 'MRR', 'COV', 'POP']
    print('LASTFM @20, mean of 5 slices.  paper / orig / port   (port-paper)')
    print('%-11s ' % 'row' + ' '.join('%-27s' % s for s in short))
    for r in out:
        cells = []
        for i, c in enumerate(COLS):
            p = PAPER[r['row']][i]
            o = r['orig'][c] if isinstance(r['orig'], pd.Series) else float('nan')
            n = r['port'][c] if isinstance(r['port'], pd.Series) else float('nan')
            cells.append('%.4f/%.4f/%.4f(%+.4f)' % (p, o, n, n - p))
        print('%-11s ' % r['row'] + ' '.join('%-27s' % x for x in cells))
    if '--no-lists' in sys.argv:
        return
    print('\nList parity orig vs port (score-equivalent per slice):')
    for row in PAPER:
        verdicts = []
        for s in range(5):
            a = f'{RES}/orig/{row}/test_window_{row}_lastfm.{s}-Saver@20.csv'
            b = f'{RES}/port/{row}/test_window_{row}_lastfm.{s}-Saver@20.csv'
            if not (os.path.exists(a) and os.path.exists(b)):
                verdicts.append('missing')
                continue
            p = subprocess.run([sys.executable, os.path.join(ROOT, 'reference/parity_lists.py'), a, b],
                               capture_output=True, text=True)
            verdicts.append('ok' if p.returncode == 0 else 'DIFF: ' + p.stdout.strip().splitlines()[-1])
        print('%-11s %s' % (row, ' | '.join(verdicts)))


if __name__ == '__main__':
    main()
