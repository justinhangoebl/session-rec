import sys, os
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, torch
import evaluation.loader as dl
from algorithms.gru4rec_torch.adapter import GRU4RecTorch
import importlib.util
spec = importlib.util.spec_from_file_location('official_eval', os.environ['G4R_OFFICIAL'] + '/evaluation.py')
# official evaluation.py does `from gru4rec_pytorch import SessionDataIterator` -> make it resolve to the vendored module
import algorithms.gru4rec_torch.gru4rec_pytorch as vend
sys.modules['gru4rec_pytorch'] = vend
oe = importlib.util.module_from_spec(spec); spec.loader.exec_module(oe)

train, test = dl.load_data_session('data/rsc15/prepared/', 'yoochoose-clicks-100k')
params = dict(loss='cross-entropy', constrained_embedding=True, embedding=0, elu_param=0, layers=[100], n_epochs=10, batch_size=32,
              dropout_p_embed=0.0, dropout_p_hidden=0.4, learning_rate=0.2, momentum=0.2, n_sample=2048, sample_alpha=0.5,
              bpreg=0.0, logq=0.0, seed=1, device='cuda:0')
a = GRU4RecTorch(**params); a.fit(train)
rec, mrr = oe.batch_eval(a.gru, test, cutoff=[5, 10, 20], batch_size=512, mode='standard')
print('OFFICIAL batch_eval  (same model):', {k: round(float(v), 4) for k, v in rec.items()}, {k: round(float(v), 4) for k, v in mrr.items()})
# our adapter, event by event, same evaluation maths (rank of the true next item)
test = test.sort_values(['SessionId', 'Time']); hits = {5: 0, 10: 0, 20: 0}; rr = {5: 0.0, 10: 0.0, 20: 0.0}; n = 0
for sid, g in test.groupby('SessionId', sort=False):
    items = g.ItemId.values
    for prev, nxt in zip(items[:-1], items[1:]):
        p = a.predict_next(sid, prev)
        if p is None or nxt not in p.index: continue
        rank = int((p.values > p[nxt]).sum()) + 1; n += 1
        for c in hits:
            if rank <= c: hits[c] += 1; rr[c] += 1.0 / rank
print('ADAPTER predict_next (same model):', {c: round(hits[c] / n, 4) for c in hits}, {c: round(rr[c] / n, 4) for c in rr}, 'events', n)
