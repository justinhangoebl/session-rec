'''
Weights & Biases logging.

The runner (run_config.py) opens one wandb run per algorithm and split / slice: the run config holds the experiment
config, the algorithm class and its params, the slice and the data statistics; the evaluation metrics are logged at
the end. Models log training curves through log() and register their torch module with watch(); both are no-ops
when no run is active, so the models still work without wandb.

Config (optional; without the section runs go to the project "session-rec"):

    wandb:
      project: session-rec
      entity: null              # team / user, default: the logged-in user
      mode: online              # online | offline | disabled (the WANDB_MODE env variable takes precedence)
      tags: [rsc15]
      watch: {log: gradients, log_freq: 100}   # wandb.watch() of torch models; false turns it off
      log_model: true           # upload the saved .pt model files (evaluation/model_store.py) as artifacts
      log_model_max_mb: 2048    # larger files (e.g. EASE's dense B) stay local; the run summary gets their path

`wandb: false` disables logging for a config.
'''
import os
import subprocess

try:
    import wandb
except ImportError:  # wandb is a core dependency, but keep the framework usable without it
    wandb = None

DEFAULTS = {'project': 'session-rec', 'entity': None, 'mode': None, 'tags': [],
            'watch': {'log': 'gradients', 'log_freq': 100}}

_settings = None


def configure(conf):
    '''Read the wandb section of an experiment config; returns True if runs will be logged.'''
    global _settings
    section = conf.get('wandb', {})
    if wandb is None or section is False or os.environ.get('WANDB_MODE') == 'disabled':
        _settings = None
        return False
    _settings = {**DEFAULTS, **(section or {})}
    return _settings.get('mode') != 'disabled'


def active():
    return wandb is not None and wandb.run is not None


def start(conf, key, spec, train, test, slice=None, phase='test'):
    '''Start the run of one algorithm on one split / slice.'''
    if _settings is None or _settings.get('mode') == 'disabled':
        return None
    if active():
        wandb.finish()

    data_name = conf['data']['name']
    algo_key = spec.get('key', key) if spec else key
    name = '{}/{}'.format(algo_key, data_name) + ('/s{}'.format(slice) if slice is not None else '')
    config = {
        'experiment': {k: conf.get(k) for k in ('type', 'key', 'evaluation')},
        'data': {k: v for k, v in conf['data'].items()},
        'slice': slice,
        'phase': phase,
        'algorithm': {'key': algo_key, 'result_key': key, 'class': spec.get('class') if spec else None},
        'params': _plain(spec.get('params', {})) if spec else {},
        'stats': {**_stats(train, 'train'), **_stats(test, phase)},
        'git_commit': _git_commit(),
    }
    tags = list(_settings.get('tags') or []) + [data_name, conf['type'], algo_key]
    wandb.init(project=_settings['project'], entity=_settings.get('entity'), mode=_settings.get('mode'),
               name=name, group='{}/{}/{}'.format(conf['key'], data_name, algo_key), job_type=conf['type'],
               tags=sorted(set(map(str, tags))), config=config, reinit='finish_previous')
    wandb.define_metric('train/epoch')
    wandb.define_metric('train/*', step_metric='train/epoch')
    return wandb.run


def log(data, step=None):
    '''Log a dict of values to the active run (no-op without one), e.g. {'train/epoch': 3, 'train/loss': 0.4}.'''
    if active():
        wandb.log(data, step=step)


def summary(values):
    '''Set run summary values (no-op without an active run).'''
    if active():
        for k, v in values.items():
            wandb.run.summary[k] = v


def watch(model, **kwargs):
    '''wandb.watch() a torch module with the configured settings (no-op without an active run).'''
    if not active() or _settings is None or not _settings.get('watch'):
        return
    options = {**(_settings['watch'] if isinstance(_settings['watch'], dict) else {}), **kwargs}
    wandb.watch(model, **options)


def log_model(path, conf, spec, key, slice=None):
    '''Log a saved .pt model file as a wandb artifact (type model; wandb.log_model: false turns it off).'''
    if not active() or _settings is None or _settings.get('log_model', True) is False:
        return
    size_mb = os.path.getsize(path) / 2 ** 20
    if size_mb > float(_settings.get('log_model_max_mb', 2048)):
        print('wandb: not uploading {} ({:.0f} MB > log_model_max_mb)'.format(path, size_mb))
        wandb.run.summary['model_file'] = os.path.abspath(path)
        return
    algo_key = (spec or {}).get('key') or key
    name = '{}-{}-{}'.format(conf['key'], conf['data']['name'], algo_key)
    name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in name)[:128]
    artifact = wandb.Artifact(name, type='model', metadata={'result_key': key, 'slice': slice,
                                                            'class': (spec or {}).get('class')})
    artifact.add_file(path)
    aliases = ['latest'] + (['slice-{}'.format(slice)] if slice is not None else [])
    wandb.run.log_artifact(artifact, aliases=aliases)


def finish(results=None, fit_time=None, phase='test', failed=False):
    '''Log the evaluation results [(metric_name, value[, DataFrame])] and close the run.'''
    if not active():
        return
    values = {}
    if fit_time is not None:
        values['time/fit_s'] = fit_time
    rows = []
    for entry in results or []:
        metric, value = entry[0].strip().rstrip(':').strip(), entry[1]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values['{}/{}'.format(phase, metric)] = value
            rows.append([metric, float(value)])
    if rows:
        values['{}/results'.format(phase)] = wandb.Table(columns=['metric', 'value'], data=rows)
    if values:
        wandb.log(values)
        for k, v in values.items():
            if not isinstance(v, wandb.Table):
                wandb.run.summary[k] = v
    wandb.finish(exit_code=1 if failed else 0)


def _stats(df, prefix):
    if df is None or len(df) == 0:
        return {}
    stats = {'{}_events'.format(prefix): int(len(df)),
             '{}_sessions'.format(prefix): int(df['SessionId'].nunique()),
             '{}_items'.format(prefix): int(df['ItemId'].nunique())}
    if 'UserId' in df.columns:
        stats['{}_users'.format(prefix)] = int(df['UserId'].nunique())
    return stats


def _plain(value):
    '''Make params JSON-friendly (hybrids get their component instances as params).'''
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return type(value).__name__


def _git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return None
