"""Split a multi-algorithm config into one config per algorithm (the Saver file is overwritten per algorithm,
so only one-algorithm configs keep every model's recommendation lists).
usage: python reference/per_model.py <config.yml> <out_conf_dir> <results_root>
Writes <out_conf_dir>/<key>.yml whose results folder is <results_root>/<key>/"""
import re, sys, os
src, outdir, root = sys.argv[1:4]
s = open(src).read(); head, algs = s.split('algorithms:\n', 1)
os.makedirs(outdir, exist_ok=True)
for blk in re.split(r'\n(?=- class:)', algs):
    blk = blk.rstrip()
    m = re.search(r'^  key: (\S+)', blk, re.M)
    if not m: continue
    key = m.group(1)
    h = re.sub(r'(?m)^(\s+folder: )results/.*$', lambda x: x.group(1) + f'{root}/{key}/', head)
    open(f'{outdir}/{key}.yml', 'w').write(h + 'algorithms:\n' + blk + '\n')
    print(key)
