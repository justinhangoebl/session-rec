"""Mean/std over seeds for each neural model -> reference/neural/summary.csv (+ printed table)."""
import csv, glob, os, statistics as st
F = 'test_single_baselines_and_models_and_hybrids_rsc15-100k.csv'
COLS = ['HitRate@20', 'MRR@20', 'NDCG@20', 'Coverage@20', 'Popularity@20', 'Training time']
rows, out = {}, []
for m in ['gru4rec', 'narm', 'stamp', 'sgnn']:
    for p in sorted(glob.glob(f'results/ref/neural/{m}/seed*/{F}')):
        r = list(csv.reader(open(p), delimiter=';'))
        hdr = [h.strip().rstrip(':').strip() for h in r[0]]
        rows.setdefault(m, []).append({h: float(v) for h, v in zip(hdr[1:], r[1][1:]) if v.strip()})
print(f"{'model':8} {'n':>2} " + ' '.join(f'{c:>22}' for c in COLS))
for m, rs in rows.items():
    cells = []
    for c in COLS:
        v = [x[c] for x in rs if c in x]
        cells.append((st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0, v))
    print(f'{m:8} {len(rs):>2} ' + ' '.join(f'{a:>12.4f} ±{b:.4f}  ' if 'time' not in c else f'{a:>12.1f} ±{b:.1f}   ' for (a, b, _), c in zip(cells, COLS)))
    out.append([m, len(rs)] + [x for a, b, _ in cells for x in (a, b)])
    print('         per-seed HR@20:', [round(x['HitRate@20'], 4) for x in rs], ' MRR@20:', [round(x['MRR@20'], 4) for x in rs])
with open('reference/neural/summary.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['model', 'n_seeds'] + [f'{c}_{s}' for c in COLS for s in ('mean', 'std')]); w.writerows(out)
