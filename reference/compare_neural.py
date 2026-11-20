"""Compare seeded neural runs: reference (old stack) vs port.

usage: python reference/compare_neural.py <ref_dir> <new_dir> [metric ...]
<dir>/seed*/<results>.csv. Plan criterion: the old mean must lie inside [min, max] of the new runs.
We also report whether the two seed ranges overlap (a looser check that is fairer with only 3 seeds)."""
import csv, glob, statistics as st, sys
ref, new = sys.argv[1:3]
metrics = sys.argv[3:] or ['HitRate@5', 'HitRate@20', 'MRR@5', 'MRR@20', 'NDCG@20', 'Coverage@20', 'Popularity@20']
def load(d):
    rs = []
    for p in sorted(glob.glob(f'{d}/seed*/*.csv')):
        if 'Saver' in p: continue
        r = list(csv.reader(open(p), delimiter=';')); h = [x.strip().rstrip(':').strip() for x in r[0]]
        rs.append({a: float(b) for a, b in zip(h[1:], r[1][1:]) if b.strip()})
    return rs
A, B = load(ref), load(new)
print(f'{len(A)} reference runs, {len(B)} port runs')
print(f"{'metric':14} {'old (ref)':>18} {'port':>18} {'diff':>8}  old-mean-in-new-range  ranges-overlap")
ok_all = True
for m in metrics:
    a = [x[m] for x in A if m in x]; b = [x[m] for x in B if m in x]
    if not a or not b: continue
    inside = min(b) <= st.mean(a) <= max(b); overlap = max(min(a), min(b)) <= min(max(a), max(b))
    ok_all &= inside
    print(f'{m:14} {st.mean(a):>10.4f} ±{st.pstdev(a):.4f} {st.mean(b):>10.4f} ±{st.pstdev(b):.4f} {st.mean(b)-st.mean(a):>+8.4f}  {"yes" if inside else "NO":>10}            {"yes" if overlap else "NO"}')
print('PLAN CRITERION (all metrics):', 'MET' if ok_all else 'NOT MET')
