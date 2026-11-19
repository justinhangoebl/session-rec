"""Compare a results folder against a reference folder.

usage: python reference/parity.py <reference_dir> <new_dir> [--tol 1e-12]
Exact check: Saver@50 recommendation lists (byte-identical) and every non-timing metric.
With --tol, metrics may differ by that absolute amount (list equality is then reported only).
Exit code 0 = parity, 1 = differences.
"""
import csv, glob, os, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument('ref'); ap.add_argument('new'); ap.add_argument('--tol', type=float, default=0.0)
a = ap.parse_args()
bad = 0
for rf in sorted(glob.glob(os.path.join(a.ref, '*.csv'))):
    nf = os.path.join(a.new, os.path.basename(rf))
    if not os.path.exists(nf): print('MISSING', nf); bad += 1; continue
    if 'Saver@' in rf:
        same = open(rf, 'rb').read() == open(nf, 'rb').read()
        print('lists ', 'IDENTICAL' if same else 'DIFFERENT', os.path.basename(rf)); bad += (not same); continue
    A = list(csv.reader(open(rf), delimiter=';')); B = list(csv.reader(open(nf), delimiter=';'))
    if len(A) != len(B) or A[0] != B[0]: print('HEADER/ROWS differ', os.path.basename(rf)); bad += 1; continue
    n = 0
    for i, (x, y) in enumerate(zip(A[1:], B[1:]), 1):
        for h, u, v in zip(A[0], x, y):
            if h.strip().startswith(('Training time', 'Testing time')) or u == v: continue
            try: ok = abs(float(u) - float(v)) <= a.tol
            except ValueError: ok = False
            if not ok: n += 1; print(f'  row {i} {x[0][:40]!r} {h.strip()}: ref={u} new={v}')
    print('metrics', 'MATCH' if n == 0 else f'{n} DIFFS', os.path.basename(rf)); bad += n > 0
sys.exit(1 if bad else 0)
