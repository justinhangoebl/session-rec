"""Write LASTFM configs for the neural rows of Latifi et al. 2021, Table 6, on the port.

Parameters: conf/save/lastfm/session_aware/window/exp/ (checked against the paper's Table A.14, all match), translated
to the port classes. GRU4Rec uses the official PyTorch implementation: the paper's LASTFM values (bpr-max, linear,
lr 0.04, momentum 0.1, no dropout, unconstrained embedding, batch 100) plus the old Theano defaults for everything the
paper does not list (10 epochs, n_sample 2048, sample_alpha 0.75, bpreg 1.0, no logq).

One config per (model, slice, seed): the runner's data.skip selects the slice, so slices run in parallel.
usage: python reference/latifi/make_neural_configs.py -> conf/latifi_lastfm/neural/<model>_s<slice>_seed<seed>.yml
"""
import os

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SEEDS = [1, 2, 3]
SLICES = 5
# key: (class, params, seed parameter name, device or None for a GPU)
MODELS = {
    'gru4rec': ('gru4rec_torch.adapter.GRU4RecTorch',
                dict(loss='bpr-max', elu_param=0, layers=[100], n_epochs=10, batch_size=100, dropout_p_embed=0.0,
                     dropout_p_hidden=0.0, learning_rate=0.04, momentum=0.1, n_sample=2048, sample_alpha=0.75,
                     bpreg=1.0, logq=0.0, embedding=0, constrained_embedding=False), 'seed', None),
    'narm': ('narm_torch.narm.NARM', dict(epochs=20, lr=0.007, hidden_units=100, factors=100), 'seed', None),
    'hgru4rec': ('hgru4rec_torch.hgru4rec.HGRU4Rec',
                 dict(final_act='linear', dropout_p_hidden_usr=0.7, dropout_p_hidden_ses=0.1, dropout_p_init=0.3,
                      momentum=0.5, learning_rate=0.09, user_propagation_mode='all', batch_size=50),
                 'torch_seed', 'cpu'),
    'iirnn': ('IIRNN_torch.ii_rnn.IIRNN',
              dict(learning_rate=0.001, dropout_pkeep=0.6, embedding_size=100, use_last_hidden_state=True,
                   max_session_representation=20, max_epoch=100), 'seed', None),
    'ncsf': ('NCFS_torch.ncfs.NCFS', dict(window_sz=1, max_nb_his_sess=0, att_alpha=1), 'seed', None),
    'nsar': ('nsar_torch.nsar.NSAR',
             dict(num_epoch=20, batch_size=64, keep_pr=0.25, learning_rate=0.003, hidden_units=100), 'seed', None),
    'shan': ('shan_torch.shan.SHAN', dict(iter=100, global_dimension=100, lambda_uv=0.01, lambda_a=10), 'seed', 'cpu'),
}
L = [3, 5, 10, 15, 20]
METRICS = ([{'class': 'accuracy_multiple.' + m, 'length': L} for m in ('Precision', 'Recall', 'MAP', 'NDCG')]
           + [{'class': 'accuracy.' + m, 'length': L} for m in ('HitRate', 'MRR')]
           + [{'class': 'coverage.Coverage', 'length': [20]}, {'class': 'popularity.Popularity', 'length': [20]},
              {'class': 'time_memory_usage.Time_usage_training'}, {'class': 'time_memory_usage.Time_usage_testing'}])

out = os.path.join(ROOT, 'conf/latifi_lastfm/neural')
os.makedirs(out, exist_ok=True)
n = 0
for key, (cls, params, seed_name, device) in MODELS.items():
    for seed in SEEDS:
        for s in range(SLICES):
            p = dict(params, **{seed_name: seed, 'device': device or 'cuda:0'})  # GPU chosen by CUDA_VISIBLE_DEVICES
            conf = {
                'type': 'window', 'key': key, 'evaluation': 'evaluation_user_based',
                'data': {'name': 'lastfm', 'folder': os.path.join(ROOT, 'data/lastfm/window_port') + '/',
                         'prefix': 'userid-timestamp-artid-artname-traid-traname', 'type': 'hdf', 'slices': SLICES,
                         'skip': [i for i in range(SLICES) if i != s]},
                'results': {'folder': os.path.join(ROOT, 'results/latifi_lastfm/neural', key, f'seed{seed}') + '/'},
                'metrics': METRICS,
                'algorithms': [{'class': cls, 'params': p, 'key': key}],
            }
            with open(os.path.join(out, f'{key}_s{s}_seed{seed}.yml'), 'w') as f:
                yaml.safe_dump(conf, f, sort_keys=False)
            n += 1
print(f'{n} configs written to conf/latifi_lastfm/neural/')
