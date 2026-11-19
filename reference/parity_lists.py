"""Tie-aware comparison of two Saver@50 recommendation files (score-equivalent lists).

usage: python reference/parity_lists.py <ref_saver.csv> <new_saver.csv> [--rtol 1e-9]
For every (session, position) row:
  1. score sequences must be equal within rtol (float summation order differs between library versions),
  2. items must have the same SET within every group of (near-)equal scores, except the last group,
     which is cut off by the list length and may legitimately contain different tied items.
Items tied with each other may be ordered differently (sort stability differs between numpy/pandas versions).
Exit 0 = score-equivalent, 1 = real differences.
"""
import sys, argparse, csv
csv.field_size_limit(10**9)
ap = argparse.ArgumentParser(); ap.add_argument('ref'); ap.add_argument('new'); ap.add_argument('--rtol', type=float, default=1e-9)
a = ap.parse_args()

def rows(path):
    with open(path, newline='') as f:
        r = csv.reader(f, delimiter=';'); next(r)
        for sid, pos, items, scores in r:
            yield (sid, pos), items.split(','), [float(x) for x in scores.split(',')]

def groups(items, scores, rtol):
    out, cur, ref = [], [], None
    for it, sc in zip(items, scores):
        if ref is not None and abs(sc - ref) <= rtol * max(1.0, abs(ref)): cur.append(it)
        else:
            if cur: out.append(cur)
            cur, ref = [it], sc
    if cur: out.append(cur)
    return out

n = bad_scores = bad_sets = order_only = 0
for (k1, i1, s1), (k2, i2, s2) in zip(rows(a.ref), rows(a.new)):
    n += 1
    if k1 != k2: print('KEY MISMATCH', k1, k2); sys.exit(1)
    if len(s1) != len(s2) or any(abs(x - y) > a.rtol * max(1.0, abs(x)) for x, y in zip(s1, s2)):
        bad_scores += 1
        if bad_scores <= 3: print('score diff at', k1)
        continue
    g1, g2 = groups(i1, s1, a.rtol), groups(i2, s2, a.rtol)
    if len(g1) != len(g2) or any(set(x) != set(y) for x, y in zip(g1[:-1], g2[:-1])) or len(g1[-1]) != len(g2[-1]):
        bad_sets += 1
        if bad_sets <= 3: print('tie-group set diff at', k1)
    elif i1 != i2: order_only += 1
print(f'{n} rows: {n-bad_scores-bad_sets} score-equivalent ({order_only} differ only in tie order), {bad_scores} score diffs, {bad_sets} tie-set diffs')
sys.exit(1 if bad_scores or bad_sets else 0)
