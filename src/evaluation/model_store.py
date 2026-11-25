'''
Saving the fitted models. The runner calls save() for every algorithm on every split / slice, right after fit:

    <results.folder>/models/<config key>_<data name>/<algorithm key>[.<slice>].pkl   the fitted object (dill), for
        reuse with filemodel.FileModel or dill.load(...).predict_next(...)
    <results.folder>/models/<config key>_<data name>/<algorithm key>[.<slice>].pt    torch.load-able dict:
        meta        config key, data, slice, class, key, result key, params, git commit
        state_dict  of the model's torch module, if it has one (on cpu)
        export      algorithm.export(), if it has one: item_ids (original ItemIds) and item_embeddings
                    [n_items, d] with row i <-> item_ids[i], plus model specific extras

Algorithms with pickle_model = False (EASE) only get the .pt file; their class rebuilds them from it (EASE.load).

Config (results section): save_models: false turns saving off; pickle_models: <folder> (the original session-rec
option) sets the folder instead of <results.folder>/models/. The .pt file is also logged to wandb as a model
artifact (wandb.log_model: false turns that off).

Loading:
    ckpt = torch.load(path, weights_only=False)
    E, ids = ckpt['export']['item_embeddings'], ckpt['export']['item_ids']
'''
import os
import traceback

import dill
import torch

from evaluation.tracking import _plain

_written = {}


def enabled(conf):
    return conf.get('results', {}).get('save_models', True) is not False


def folder(conf):
    results = conf.get('results', {})
    base = results.get('pickle_models')
    if not isinstance(base, str):
        base = os.path.join(results.get('folder', 'results/'), 'models')
    return os.path.join(base, '{}_{}'.format(conf['key'], conf['data']['name']))


def file_stem(conf, key, spec, iteration=None):
    '''<short key>[.<iteration>]; a numbered suffix if another result key of this run already used the name.'''
    short = (spec or {}).get('key') or key
    stem = short + ('.{}'.format(iteration) if iteration is not None else '')
    path = os.path.join(folder(conf), stem)
    n = 1
    while _written.get(path, key) != key:
        n += 1
        path = os.path.join(folder(conf), '{}-{}'.format(stem, n))
    _written[path] = key
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
    '''Write the .pkl and .pt files of a fitted algorithm; returns the .pt path (None if it has no torch module).'''
    stem = file_stem(conf, key, spec, iteration)
    os.makedirs(os.path.dirname(stem), exist_ok=True)

    if getattr(algorithm, 'pickle_model', True):  # e.g. EASE: its .pt export already is the whole model
        try:
            with open(stem + '.pkl', 'wb') as f:
                dill.dump(algorithm, f)
            print('saved model', stem + '.pkl')
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
    if module is None and exported is None:
        return None

    ckpt = {
        'meta': {'config_key': conf['key'], 'data': conf['data'], 'slice': slice, 'result_key': key,
                 **_plain(spec or {}), **(meta or {})},
        'state_dict': {k: v.detach().cpu() for k, v in module.state_dict().items()} if module is not None else None,
        'export': exported,
    }
    torch.save(ckpt, stem + '.pt')
    print('saved model', stem + '.pt')
    return stem + '.pt'
