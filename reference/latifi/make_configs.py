"""Write the LASTFM run configs for reproducing Latifi et al. 2021, Table 6 (non-neural rows).

The algorithm entries are copied from the repo's own experiment configs
(conf/save/lastfm/session_aware/window/exp/), which match the paper's Table 14 except vstan's lambda_ipw
(repo 0.00001, paper 0.0001); both are run. Each config gets the paper metrics plus saver.Saver, so the
lists of the original and the port can be compared slice by slice.

usage: python reference/latifi/make_configs.py   -> conf/latifi_lastfm/{orig,port}/<row>.yml
"""
import copy
import os

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC = os.path.join(ROOT, 'conf/save/lastfm/session_aware/window/exp')

# paper row -> (repo config, index of the algorithm entry in it)
ROWS = {
    'vsknn': ('vsknn/window_lastfm_vsknn.yml', 0),
    'vsknn_eb': ('vsknn/window_lastfm_vsknn_EB.yml', 0),
    'stan': ('stan/window_lastfm_stan.yml', 0),
    'stan_ebr': ('stan/window_lastfm_stan_EBR.yml', 0),
    'vstan': ('vstan/window_lastfm_vstan.yml', 0),
    'vstan_eb': ('vstan/window_lastfm_vstan_EB.yml', 0),
    'sr': ('sr/window_lastfm_sr.yml', 0),
    'sr_b': ('sr/window_lastfm_usr.yml', 0),
    'vstan_ipw4': ('vstan/window_lastfm_vstan.yml', 0),  # vstan with the paper's lambda_ipw = 0.0001
    # vstan_eb as Table A.14 prints it (lambda_spw 0.4525) misses Table 6 on the original code too; with
    # lambda_spw 1.4525 (the off-grid value in the repo's vstan_EBR config) it reproduces Table 6 exactly
    'vstan_eb_spw': ('vstan/window_lastfm_vstan_EB.yml', 0),
}
# one algorithm per config: the Saver file name has no algorithm key, so a second algorithm would overwrite it

for stack in ('orig', 'port'):
    out = os.path.join(ROOT, 'conf/latifi_lastfm', stack)
    os.makedirs(out, exist_ok=True)
    for row, (path, idx) in ROWS.items():
        with open(os.path.join(SRC, path)) as f:
            conf = yaml.safe_load(f)
        algo = copy.deepcopy(conf['algorithms'][idx])
        algo['key'] = row
        if row == 'vstan_ipw4':
            algo['params']['lambda_ipw'] = 0.0001
        if row == 'vstan_eb_spw':
            algo['params']['lambda_spw'] = 1.4525
        conf['key'] = row
        conf['algorithms'] = [algo]
        conf['data']['folder'] = os.path.join(ROOT, 'data/lastfm', 'window_' + stack) + '/'
        conf['results'] = {'folder': os.path.join(ROOT, 'results/latifi_lastfm', stack, row) + '/'}
        conf['metrics'] = [m for m in conf['metrics'] if 'time_memory' not in m['class']]
        conf['metrics'].append({'class': 'saver.Saver', 'length': [20]})
        with open(os.path.join(out, row + '.yml'), 'w') as f:
            yaml.safe_dump(conf, f, sort_keys=False)
print('written to conf/latifi_lastfm/{orig,port}/')
