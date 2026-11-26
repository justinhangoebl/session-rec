'''
Candidate pools of saved models (evaluation/candidates.py) without retraining:

    session-rec-candidates <saved model folder> [<folder> ...] [--parts valid_test test] [--k 200]
                           [--history rolling|train] [--data <hdf file>] [--device cuda:0]

The data file defaults to the one the model was trained on (meta.json: data.folder + data.prefix + .hdf); its
'train' key is the training data (for the rolling user history). Output: <folder>/candidates/<part>/.
'''
import argparse
import json
import os
import sys

import pandas as pd


def cli():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folders', nargs='+')
    p.add_argument('--parts', nargs='+', default=None, help='hdf keys in time order (default: valid_test test, '
                                                            'plus calib if the file has it)')
    p.add_argument('--k', type=int, default=200)
    p.add_argument('--history', default='rolling', choices=['rolling', 'train'])
    p.add_argument('--data', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--batch-requests', type=int, default=2048)
    args = p.parse_args()

    from evaluation import candidates, model_store
    for folder in args.folders:
        meta = json.load(open(os.path.join(folder, 'meta.json')))
        file = args.data or os.path.join(meta['data']['folder'], meta['data']['prefix'] + '.hdf')
        with pd.HDFStore(file, 'r') as store:
            keys = [k.strip('/') for k in store.keys()]
        parts = args.parts or [k for k in ('valid_test', 'calib', 'test') if k in keys]
        print('{}: {} on {}'.format(folder, parts, file))
        train = pd.read_hdf(file, key='train')
        model = model_store.load(folder, device=args.device)
        candidates.score_parts(model, [(k, pd.read_hdf(file, key=k)) for k in parts], train, folder, k=args.k,
                               history=args.history, batch_requests=args.batch_requests)
        del model


if __name__ == '__main__':
    cli()
