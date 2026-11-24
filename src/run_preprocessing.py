import importlib
import os
import sys
import traceback
from pathlib import Path

import yaml


def main(conf):
    '''
    Run one preprocessing config: load the raw data with preprocessing.<mode>.preprocess_<preprocessor>, filter it
    and write one train/test split (type single) or several time slices (type window).
    '''
    print('Checking {}'.format(conf))

    file = Path(conf)
    if file.is_file():
        print('Loading file')
        with open(str(file)) as stream:
            c = yaml.safe_load(stream)
        print('processing config ' + conf)

        try:
            run_file(c)
            print('finished config ' + conf)
        except (KeyboardInterrupt, SystemExit):
            print('manually aborted config ' + conf)
            raise
        except Exception:
            print('error for config ', file)
            traceback.print_exc()
            sys.exit(1)

        exit()

    print('File not found: ' + conf)


def run_file(conf):
    preprocessor = load_preprocessor(conf)

    raw = conf['data']['folder'] + conf['data']['prefix']
    if 'sample_percentage' in conf['data']:
        data = preprocessor.load_data(raw, sample_percentage=conf['data']['sample_percentage'])
    else:
        data = preprocessor.load_data(raw)
    if type(data) == tuple:
        data = data[0]
    # session-aware window: the filters run per slice, after splitting
    if not (conf['mode'] == 'session_aware' and conf['type'] == 'window'):
        data = preprocessor.filter_data(data, **conf['filter'])

    out = conf['output']['folder'] + conf['data']['prefix']
    ensure_dir(out)
    if conf['type'] == 'single':
        preprocessor.split_data(data, out, **conf['params'])
    elif conf['type'] == 'window':
        preprocessor.slice_data(data, out, **conf['params'])
    else:
        print('preprocessing type not supported')


def load_preprocessor(conf):
    return importlib.import_module('preprocessing.' + conf['mode'] + '.preprocess_' + conf['preprocessor'])


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def cli():
    '''Entry point: session-rec-preprocess <preprocess_config.yml>'''
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('Preprocessing configuration expected.')
        sys.exit(2)


if __name__ == '__main__':
    cli()
