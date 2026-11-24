import gc
import glob
import importlib
import os
import random
import sys
import time
import traceback
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import yaml
from skopt import Optimizer

import evaluation.loader as dl


def main(conf, out=None):
    '''
    Run the experiments of a configuration.
        --------
        conf: string
            A config file, or a folder of config files.
        out: string
            With a folder as conf: keep watching the folder for new configs and move finished ones to this folder.
    '''
    print('Checking {}'.format(conf))

    file = Path(conf)
    if file.is_file():
        print('Loading file')
        try:
            run_file(load_yaml(file))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            print('error for config ', conf)
            traceback.print_exc()
            sys.exit(1)
        exit()

    if file.is_dir():
        if out is not None:
            ensure_dir(out + '/out.txt')
            while True:
                print('waiting for configuration files in ', conf)
                configs = glob.glob(conf + '/' + '*.yml')
                if len(configs) > 0:
                    file = Path(configs[0])
                    try:
                        print('processing config', configs[0])
                        run_file(load_yaml(file))
                        print('finished config', configs[0])
                        os.rename(configs[0], out + '/' + file.name + str(time.time()) + '.done')
                    except (KeyboardInterrupt, SystemExit):
                        os.rename(configs[0], out + '/' + file.name + str(time.time()) + '.cancled')
                        raise
                    except Exception:
                        print('error for config ', configs[0])
                        os.rename(configs[0], out + '/' + file.name + str(time.time()) + '.error')
                        traceback.print_exc()
                time.sleep(5)
        else:
            print('processing folder ', conf)
            for conf in glob.glob(conf + '/' + '*.yml'):
                try:
                    print('processing config', conf)
                    run_file(load_yaml(Path(conf)))
                    print('finished config', conf)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    print('error for config ', conf)
                    traceback.print_exc()
            exit()


def load_yaml(file):
    with open(str(file)) as stream:
        return yaml.safe_load(stream)


def run_file(conf):
    if conf['type'] == 'single':
        run_single(conf)
    elif conf['type'] == 'window':
        run_window(conf)
    elif conf['type'] == 'opt':
        run_opt(conf)
    elif conf['type'] == 'bayopt':
        run_bayopt(conf)
    else:
        print(conf['type'] + ' not supported')


def load_data(conf, **kwargs):
    '''
    Load train and test set as configured in conf['data'] (csv by default, or data.type: hdf).
    kwargs (slice_num or train_eval) are passed to the loader together with the optional data.opts.
    '''
    data = conf['data']
    opts = {**kwargs, **data.get('opts', {})}
    if data.get('type') == 'hdf':
        return dl.load_data_session_hdf(data['folder'], data['prefix'], **opts)
    return dl.load_data_session(data['folder'], data['prefix'], **opts)


def run_single(conf, slice=None):
    '''
    Evaluate the algorithms on one split (or on one slice of a window config).
    '''
    print('run test single')

    algorithms = create_algorithms_dict(conf['algorithms'])
    metrics = create_metric_list(conf['metrics'])
    evaluation = load_evaluation(conf['evaluation'])

    train, test = load_data(conf, slice_num=slice)
    buys = pd.DataFrame()
    if conf['data'].get('type') != 'hdf' and 'buys' in conf['data'] and 'file_buys' in conf['data']:
        buys = dl.load_buys(conf['data']['folder'], conf['data']['file_buys'])

    for m in metrics:
        m.init(train)
        if hasattr(m, 'set_buys'):
            m.set_buys(buys, test)

    results = {}
    for k, a in algorithms.items():
        eval_algorithm(train, test, k, a, evaluation, metrics, results, conf, slice=slice, iteration=slice)

    print_results(results)
    write_results_csv(results, conf, iteration=slice)


def run_opt_iteration(conf, algorithms, iteration, globals):
    '''
    One optimisation step: evaluate the given algorithm instances on the validation split and track the best
    value of the first metric (conf['optimize']) per algorithm class in globals.
    '''
    print('run test opt single')

    for a in algorithms.values():
        if type(a) not in globals:
            globals[type(a)] = {'key': '', 'best': -1}

    metrics = create_metric(conf['optimize']) + create_metric_list(conf['metrics'])
    evaluation = load_evaluation(conf['evaluation'])

    train, test = load_data(conf, train_eval=conf['data'].get('train_eval', True))

    for m in metrics:
        m.init(train)

    results = {}
    for k, a in algorithms.items():
        eval_algorithm(train, test, k, a, evaluation, metrics, results, conf, iteration=iteration, out=False)

    write_results_csv(results, conf, iteration=iteration)

    for k, a in algorithms.items():
        aclass = type(a)
        current_value = results[k][0][1]
        if globals[aclass]['best'] < current_value:
            print('found new best configuration')
            print(k)
            print('improvement from {} to {}'.format(globals[aclass]['best'], current_value))
            globals[aclass]['best'] = current_value
            globals[aclass]['key'] = k
        globals['current'] = current_value

    globals['results'].append(results)

    del algorithms, metrics, evaluation, results
    gc.collect()


def run_window(conf):
    print('run test window')

    slices = list(range(conf['data']['slices']))
    for i in conf['data'].get('skip', []):
        slices.remove(i)

    for i in slices:
        print('start run for slice ', str(i))
        run_single(conf, slice=i)


def opt_range(conf):
    optimize = conf.get('optimize', {})
    return optimize.get('iterations_skip', 0), optimize.get('iterations', 100)


def run_opt(conf):
    '''
    Random search: every iteration draws new values for each algorithm's params_opt.
    '''
    start, iterations = opt_range(conf)
    print('run opt with {} iterations starting at {}'.format(iterations, start))

    globals = {'results': []}
    for i in range(start, iterations):
        print('start random test ', str(i))
        run_opt_iteration(conf, create_algorithms_dict(conf['algorithms']), i, globals)

    write_results_csv(merge_results(globals), conf)


def run_bayopt(conf):
    '''
    Bayesian optimisation (scikit-optimize) of each algorithm's params_opt.
    '''
    start, iterations = opt_range(conf)
    print('run opt with {} iterations starting at {}'.format(iterations, start))

    globals = {'results': []}
    for entry in conf['algorithms']:
        space_dict = generate_space(entry)
        opt = Optimizer([values for values in space_dict.values()],
                        n_initial_points=conf.get('optimize', {}).get('initial_points', 10))

        for i in range(start, iterations):
            print('start bayesian test ', str(i))
            suggested = opt.ask()
            params = {k: v for k, v in zip(space_dict.keys(), suggested)}
            run_opt_iteration(conf, create_algorithm_dict(entry, params), i, globals)
            # skopt minimises
            opt.tell(suggested, -1 * globals['current'])

    write_results_csv(merge_results(globals), conf)


def merge_results(globals):
    merged = {}
    for results in globals['results']:
        merged.update(results)
    return merged


def eval_algorithm(train, test, key, algorithm, eval, metrics, results, conf, slice=None, iteration=None, out=True):
    '''
    Fit one algorithm, evaluate it with the evaluation module eval and store its metric values in results[key].
    '''
    ts = time.time()
    print('fit ', key)

    if hasattr(algorithm, 'init'):
        algorithm.init(train, test, slice=slice)

    for m in metrics:
        if hasattr(m, 'start'):
            m.start(algorithm)

    algorithm.fit(train, test)
    print(key, ' time: ', (time.time() - ts))

    if 'results' in conf and 'pickle_models' in conf['results']:
        try:
            save_model(key, algorithm, conf)
        except Exception:
            print('could not save model for ' + key)

    for m in metrics:
        if hasattr(m, 'start'):
            m.stop(algorithm)

    results[key] = eval.evaluate_sessions(algorithm, metrics, test, train)
    if out:
        write_results_csv({key: results[key]}, conf, extra=key, iteration=iteration)

    algorithm.clear()


def write_results_csv(results, conf, iteration=None, extra=None):
    '''
    Write results[algorithm_key] = [(metric_name, value[, DataFrame])] to
    <results.folder>/test_<type>_<key>_<data.name>[.<iteration>].csv; a DataFrame (e.g. from the Saver metric)
    goes to a separate file next to it.
    '''
    if 'results' in conf and 'folder' in conf['results']:

        export_csv = conf['results']['folder'] + 'test_' + conf['type'] + '_' + conf['key'] + '_' + conf['data']['name']
        if iteration is not None:
            export_csv += '.' + str(iteration)
        export_csv += '.csv'

        ensure_dir(export_csv)

        with open(export_csv, 'w+') as file:
            file.write('Metrics;')
            for k, l in results.items():
                for e in l:
                    file.write(e[0])
                    file.write(';')
                break
            file.write('\n')

            for k, l in results.items():
                file.write(k)
                file.write(';')
                for e in l:
                    file.write(str(e[1]))
                    file.write(';')
                    if len(e) > 2:
                        if type(e[2]) == pd.DataFrame:
                            name = export_csv.replace('.csv', '-') + e[0].replace(':', '').replace(' ', '') + '.csv'
                            e[2].to_csv(name, sep=";", index=False)
                file.write('\n')


def save_model(key, algorithm, conf):
    '''
    Pickle a fitted algorithm (results.pickle_models) for reuse with filemodel.FileModel.
    '''
    file_name = Path(conf['results']['folder'] + '/' + conf['key'] + '_' + conf['data']['name'] + '_' + key + '.pkl')
    ensure_dir(file_name)
    with open(file_name, 'wb') as file:
        dill.dump(algorithm, file)


def print_results(res):
    for k, l in res.items():
        for e in l:
            print(k, ':', e[0], ' ', e[1])


def load_evaluation(module):
    return importlib.import_module('evaluation.' + module)


def algorithm_key(algorithm, params, a_keys=None):
    '''
    Result key of an algorithm: its configured key plus every parameter (hybrids: the keys of their components).
    '''
    key = algorithm['key'] if 'key' in algorithm else algorithm['class']
    if 'params' in algorithm:
        for k, val in params.items():
            if k == 'algorithms' and a_keys is not None:
                for pKey in a_keys:
                    key += '-' + pKey
            elif k != 'file':
                key += '-' + str(k) + "=" + str(val)
                key = key.replace(',', '_')
    return key


def instantiate(algorithm, params, algorithms):
    '''
    Create the instance(s) of one config entry in algorithms; hybrids first get their component instances.
    '''
    Class = load_class('algorithms.' + algorithm['class'])

    a_keys = None
    if 'params' in algorithm and 'algorithms' in algorithm['params']:
        hybrid_algorithms = create_algorithms_dict(algorithm['params']['algorithms'])
        params['algorithms'] = list(hybrid_algorithms.values())
        a_keys = list(hybrid_algorithms.keys())

    key = algorithm_key(algorithm, params, a_keys)

    if 'params_var' in algorithm:
        for k, var in algorithm['params_var'].items():
            for val in var:
                params[k] = val
                kv = k
                for v in val:
                    kv += '-' + str(v)
                algorithms[key + kv] = Class(**params)
    else:
        algorithms[key] = Class(**params)


def create_algorithms_dict(entries):
    '''
    Create the algorithm instances of the config's algorithms list (with random params_opt values, if any).
    '''
    algorithms = {}
    for algorithm in entries:
        params = {**algorithm.get('params', {}), **generate_random_params(algorithm)}
        instantiate(algorithm, params, algorithms)
    return algorithms


def create_algorithm_dict(entry, additional_params={}):
    '''
    Create the instance(s) of one algorithms entry with additional_params on top of its params.
    '''
    algorithms = {}
    instantiate(entry, {**entry.get('params', {}), **additional_params}, algorithms)
    return algorithms


def params_space(value):
    '''
    Candidate values of a params_opt list: plain values, lists of values and {from, to, in} ranges, flattened.
    '''
    space = []
    for entry in value:
        if type(entry) == list:
            space += entry
        elif type(entry) == dict:
            space += list(create_linspace(entry))
        else:
            space += [entry]
    return space


def generate_random_params(algorithm):
    params = {}

    for key, value in algorithm.get('params_opt', {}).items():
        if type(value) == list:
            chosen = random.choice(params_space(value))
        elif type(value) == dict:
            if 'space' in value:
                space = []
                if value['space'] == 'weight':
                    space.append(create_weightspace(value))
                elif value['space'] == 'recLen':
                    space.append(create_linspace(value))
            else:
                space = create_linspace(value)
            chosen = random.choice(space)
            chosen = float(chosen) if 'type' in value and value['type'] == 'float' else chosen
        else:
            print('not the right type')

        params[key] = chosen

    return params


def generate_space(algorithm):
    params = {}

    for key, value in algorithm.get('params_opt', {}).items():
        if type(value) == list:
            space = params_space(value)
        elif type(value) == dict:
            if 'space' in value:
                if value['space'] == 'weight':
                    space = [create_weightspace(value)]
                elif value['space'] == 'recLen':
                    space = [create_linspace(value)]
            elif value['type'] == 'float':
                space = (float(value['from']), float(value['to']))
            else:
                space = (int(value['from']), int(value['to']))
        else:
            print('not the right type')

        params[key] = space

    return params


def create_weightspace(value):
    '''
    Random hybrid weights: num values rounded to 2 decimals that sum to 1.
    '''
    num = value['num']
    space = []
    sum = 0
    rand = 1
    for i in range(num - 1):
        while (sum + rand) >= 1:
            rand = round(np.random.rand(), 2)
        space.append(rand)
        sum += rand
        rand = 1

    space.append(round(1 - sum, 2))
    return space


def create_linspace(value):
    return np.linspace(value['from'], value['to'], num=value['in']).astype(value.get('type', 'float32'))


def create_metric_list(entries):
    metrics = []
    for metric in entries:
        metrics += create_metric(metric)
    return metrics


def create_metric(metric):
    '''
    One metric instance per configured list length (or one without a length).
    '''
    Class = load_class('evaluation.metrics.' + metric['class'])
    if 'length' in metric:
        return [Class(list_length) for list_length in metric['length']]
    return [Class()]


def load_class(path):
    '''
    Load a class by dotted path, e.g. algorithms.knn.vsknn.VMContextKNN.
    '''
    module_name, class_name = path.rsplit('.', 1)
    return getattr(importlib.import_module(module_name), class_name)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def cli():
    '''Entry point: session-rec <config.yml | config_folder> [out_folder]'''
    if len(sys.argv) > 1:
        main(sys.argv[1], out=sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print('File or folder expected.')
        sys.exit(2)


if __name__ == '__main__':
    cli()
