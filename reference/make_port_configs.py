"""Make conf/port_neural/<model>_seed<N>.yml from a reference config, swapping only the algorithm class.

usage: python reference/make_port_configs.py <ref_conf_dir> <model> <old.class.Path> <new.class.Path> [results_root]
Same data, metrics and params as the reference; adds seed: N and device: 'cuda:0' to params; results go to
<results_root>/<model>/seed<N>/ (default results/port/neural)."""
import glob, os, re, sys
ref_dir, model, old_cls, new_cls = sys.argv[1:5]
root = sys.argv[5] if len(sys.argv) > 5 else 'results/port/neural'
os.makedirs('conf/port_neural', exist_ok=True)
for f in sorted(glob.glob(f'{ref_dir}/{model}_seed*.yml')):
    seed = int(re.search(r'_seed(\d+)\.yml$', f).group(1))
    s = open(f).read()
    assert f'class: {old_cls}' in s, (f, old_cls)
    s = s.replace(f'class: {old_cls}', f'class: {new_cls}')
    s = re.sub(r'(?m)^(  params: \{)(.*)\}\s*$', lambda m: f"{m.group(1)}{m.group(2).rstrip().rstrip(',')}, seed: {seed}, device: 'cuda:0' }}", s, count=1)
    s = re.sub(r'(?m)^(\s+folder: )results/.*$', lambda m: f'{m.group(1)}{root}/{model}/seed{seed}/', s)
    out = f'conf/port_neural/{model}_seed{seed}.yml'; open(out, 'w').write(s); print(out)
