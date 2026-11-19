"""Compare results/port/pm/<set>/<key> against reference/pm/<set>/<key> for every model:
tie-aware list check (parity_lists.py) + max absolute difference of the non-timing metrics."""
import csv, glob, os, subprocess, sys
REF, NEW = (sys.argv[1:3] if len(sys.argv) > 2 else ('reference/pm', 'results/port/pm'))
def metrics(p):
    r = list(csv.reader(open(p), delimiter=';')); return {h.strip(): float(v) for h, v in zip(r[0][1:], r[1][1:]) if v.strip() and not h.strip().startswith(('Training time', 'Testing time'))}
print(f"{'model':28} {'lists':32} {'max |d metric|':>15}  worst metric")
fails = 0
for d in sorted(glob.glob(f'{REF}/*/*')):
    s, k = d.split('/')[-2:]
    if s == 'conf': continue
    rs, ns = glob.glob(f'{d}/*Saver*'), glob.glob(f'{NEW}/{s}/{k}/*Saver*')
    rm = [f for f in glob.glob(f'{d}/*.csv') if 'Saver' not in f]; nm = [f for f in glob.glob(f'{NEW}/{s}/{k}/*.csv') if 'Saver' not in f]
    if not (rs and ns and rm and nm): print(f'{s}/{k:25} MISSING'); fails += 1; continue
    p = subprocess.run([sys.executable, 'reference/parity_lists.py', rs[0], ns[0]], capture_output=True, text=True)
    summ = p.stdout.strip().splitlines()[-1]
    a, b = metrics(rm[0]), metrics(nm[0]); worst = max(a, key=lambda m: abs(a[m] - b.get(m, 1e9)))
    print(f'{s+"/"+k:28} {"OK" if p.returncode==0 else "FAIL":4} {summ.split(":")[0]+": "+summ.split(": ")[1][:20] if p.returncode==0 else summ[:60]:27} {abs(a[worst]-b[worst]):>10.5f}  {worst}')
    fails += p.returncode != 0
print('FAILS:', fails)
