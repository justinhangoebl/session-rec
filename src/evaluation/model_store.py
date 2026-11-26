'''
Saving the fitted models. The runner calls save() for every algorithm on every split / slice, right after fit.
Each model gets its own folder:

    saved/<algorithm key>-<data name>-<YYYYMMDD-HHMM>-<config key>[-s<slice>]/
        model.pkl     the fitted object (dill), for dill.load(...).predict_next(...) / filemodel.FileModel
        model.pt      torch.load-able dict:
                          meta        config key, data, slice, class, key, result key, params, git commit, fit time
                          state_dict  of the model's torch module, if it has one (on cpu)
                          export      algorithm.export(), if it has one: item_ids (original ItemIds) and
                                      item_embeddings [n_items, d] with row i <-> item_ids[i], plus model extras
        meta.json     the meta dict (readable without torch)
        candidates/   top-N candidate pools per data part, if the config asks for them (evaluation/candidates.py)

e.g. saved/gru4rec-yambda50m-20261125-1830-backbones/. Algorithms with pickle_model = False (EASE) only get model.pt;
their class rebuilds them from it (EASE.load).

Config (results section): save_models: false turns saving off; save_folder: <folder> sets the base folder
(default saved/). The model.pt file is also logged to wandb as a model artifact (wandb.log_model: false turns that off).

Loading:
    ckpt = torch.load('saved/.../model.pt', weights_only=False)
    E, ids = ckpt['export']['item_embeddings'], ckpt['export']['item_ids']
    model = load('saved/...')    # the fitted object (model.pkl, or EASE.load for EASE)
'''
import json
import os
import time
import traceback

import dill
import torch

from evaluation.tracking import _plain, _git_commit

_written = set()


def enabled(conf):
    return conf.get('results', {}).get('save_models', True) is not False


def base_folder(conf):
    return conf.get('results', {}).get('save_folder', 'saved/')


def _clean(name):
    return ''.join(c if c.isalnum() or c in '_.' else '' for c in str(name).replace('-', '')).lower() or 'x'


def model_folder(conf, key, spec, slice=None, iteration=None):
    '''saved/<algo>-<data>-<date>-<config key>[-s<slice>]; a numbered suffix if the folder already exists.'''
    algo = _clean((spec or {}).get('key') or key)
    name = '{}-{}-{}-{}'.format(algo, _clean(conf['data']['name']), time.strftime('%Y%m%d-%H%M'), _clean(conf['key']))
    if slice is not None:
        name += '-s{}'.format(slice)
    elif iteration is not None:
        name += '-i{}'.format(iteration)
    path = os.path.join(base_folder(conf), name)
    n = 1
    while path in _written or os.path.exists(path):
        n += 1
        path = os.path.join(base_folder(conf), '{}-{}'.format(name, n))
    _written.add(path)
    return path


def find_module(algorithm):
    '''The torch module of a fitted algorithm: .model, or .model of a wrapped object (e.g. GRU4Rec's .gru).'''
    candidates = [getattr(algorithm, 'model', None)]
    for value in vars(algorithm).values():
        candidates.append(getattr(value, 'model', None) if not isinstance(value, torch.nn.Module) else value)
    for c in candidates:
        if isinstance(c, torch.nn.Module):
            return c
    return None


def save(conf, key, spec, algorithm, slice=None, iteration=None, meta=None):
    '''Write model.pkl / model.pt / meta.json of a fitted algorithm; returns its folder.'''
    folder = model_folder(conf, key, spec, slice, iteration)
    os.makedirs(folder, exist_ok=True)
    info = {'config_key': conf['key'], 'data': conf['data'], 'slice': slice, 'iteration': iteration,
            'result_key': key, 'saved': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_commit': _git_commit(),
            **_plain(spec or {}), **_plain(meta or {})}

    if getattr(algorithm, 'pickle_model', True):  # e.g. EASE: its .pt export already is the whole model
        try:
            with open(os.path.join(folder, 'model.pkl'), 'wb') as f:
                dill.dump(algorithm, f)
        except Exception:
            print('WARNING: could not pickle model ' + key)
            traceback.print_exc()

    module = find_module(algorithm)
    exported = None
    if hasattr(algorithm, 'export'):
        try:
            exported = algorithm.export()
        except Exception:
            print('WARNING: export() failed for ' + key)
            traceback.print_exc()
    if module is not None or exported is not None:
        torch.save({'meta': info,
                    'state_dict': {k: v.detach().cpu() for k, v in module.state_dict().items()}
                    if module is not None else None,
                    'export': exported}, os.path.join(folder, 'model.pt'))
    with open(os.path.join(folder, 'meta.json'), 'w') as f:
        json.dump(info, f, indent=1, default=str)
    print('saved model', folder)
    return folder


def load(folder, device=None):
    '''The fitted algorithm saved in folder: model.pkl, or the class' load(model.pt) (EASE).'''
    pkl, pt = os.path.join(folder, 'model.pkl'), os.path.join(folder, 'model.pt')
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            model = dill.load(f)
        if device is not None and hasattr(model, 'to'):
            model.to(device)
        return model
    meta = json.load(open(os.path.join(folder, 'meta.json')))
    from run_config import load_class
    return load_class('algorithms.' + meta['class']).load(pt, device=device)
