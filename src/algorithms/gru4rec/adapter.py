"""Adapter that lets the official GRU4Rec PyTorch implementation run inside this repo's evaluation loop.

The official class (gru4rec_pytorch.GRU4Rec, vendored unchanged) trains on a DataFrame and evaluates in batches.
This runner calls fit(train, test) and then predict_next(session_id, item_id, items_to_predict) once per event,
so the adapter keeps the GRU hidden state of the *current* session and resets it when the session id changes.

YAML:
  - class: gru4rec.adapter.GRU4RecTorch
    params: {loss: cross-entropy, layers: [100], n_epochs: 10, batch_size: 32, dropout_p_hidden: 0.4,
             learning_rate: 0.2, momentum: 0.2, n_sample: 2048, sample_alpha: 0.5, bpreg: 0.0, logq: 1.0,
             constrained_embedding: True, embedding: 0, elu_param: 0, seed: 1, device: 'cuda:0'}
All params except seed/device go to the official constructor unchanged (the PyTorch version only has the
'cross-entropy' and 'bpr-max' losses, not 'top1-max').
"""
import numpy as np
import pandas as pd
import torch

from algorithms.gru4rec.gru4rec_pytorch import GRU4Rec


class GRU4RecTorch:
    def __init__(self, seed=None, device=None, **params):
        self.seed = seed
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.gru = GRU4Rec(device=self.device, **params)
        self.model = None
        self.current_session = None

    def fit(self, train, test=None):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        self.gru.fit(train)  # official defaults (compatibility_mode=True), as in the official run.py
        if self.gru.error_during_train:
            raise RuntimeError('GRU4Rec training diverged (NaN loss)')
        self.model = self.gru.model
        self.model.eval()
        # the negative-sample cache (~80 MB of random item ids) is only needed for training; keep it out of saved models
        if hasattr(self.gru.data_iterator, 'sample_cache'):
            del self.gru.data_iterator.sample_cache
        self.itemidmap = self.gru.data_iterator.itemidmap
        self.item_ids = self.itemidmap.index.values
        self.current_session = None

    def _reset_hidden(self):
        self.H = [torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=self.device)
                  for size in self.gru.layers]

    @torch.no_grad()
    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, mode_type='view', timestamp=0):
        """Scores for all training items (or the given subset) after seeing input_item_id in session_id.
        Like the original GRU4Rec in this repo, returns None for an input item that was not in the training data
        (the evaluation loop expects test events to be filtered to known items beforehand)."""
        if input_item_id not in self.itemidmap.index:
            return None
        if session_id != self.current_session:
            self._reset_hidden()
            self.current_session = session_id
        x = torch.tensor([self.itemidmap[input_item_id]], dtype=torch.long, device=self.device)
        scores = self.model.forward(x, self.H, None, training=False)[0].cpu().numpy()
        preds = pd.Series(data=scores, index=self.item_ids)
        if predict_for_item_ids is not None:
            preds = preds[preds.index.isin(predict_for_item_ids)]
        return preds

    @torch.no_grad()
    def export(self):
        '''Item embeddings: row i belongs to item_ids[i]. item_embeddings is the output embedding Wy (also the input
        embedding with constrained_embedding); input_item_embeddings the separate input embedding E, if any.'''
        rows = torch.as_tensor(self.itemidmap.values.astype('int64'), device=self.device)
        out = {'item_ids': self.itemidmap.index.values,
               'item_embeddings': self.model.Wy.weight[rows].detach().float().cpu(),
               'item_bias': self.model.By.weight[rows, 0].detach().float().cpu()}
        if getattr(self.model, 'embedding', 0) and not self.model.constrained_embedding:
            out['input_item_embeddings'] = self.model.E.weight[rows].detach().float().cpu()
        return out

    # ---------------------------------------------------------------- candidate pools (evaluation/candidates.py)

    candidate_batch_requests = 200000  # many sessions per batch: the GRU steps through them in parallel

    def candidate_item_ids(self):
        ids = np.empty(len(self.itemidmap), dtype=self.itemidmap.index.dtype)
        ids[self.itemidmap.values.astype(np.int64)] = self.itemidmap.index.values
        return ids

    @torch.no_grad()
    def score_sessions(self, sessions, histories=None, chunk=4096):
        '''Scores after every prefix of every session (column indices, -1 = unknown item: the state is kept).'''
        self.model.eval()
        lens = np.array([len(s) for s in sessions])
        offsets = np.r_[0, np.cumsum(lens - 1)][:-1]
        order = np.argsort(-lens, kind='stable')  # longest first: the active sessions of step t are a prefix
        lens_o, off_o = lens[order], offsets[order]
        L = int(lens.max()) if len(lens) else 0
        X = np.full((len(sessions), max(L, 1)), -1, dtype=np.int64)
        for r, si in enumerate(order):
            X[r, :lens[si]] = sessions[si]
        X = torch.as_tensor(X, device=self.device)
        for c in range(0, len(order), chunk):  # independent groups of sessions
            sl = slice(c, c + chunk)
            H = [torch.zeros((len(lens_o[sl]), size), device=self.device) for size in self.gru.layers]
            for t in range(int(lens_o[sl].max()) - 1):
                na = int((lens_o[sl] > t + 1).sum())
                x = X[c:c + na, t]
                valid = (x >= 0).unsqueeze(1)
                old = [h[:na] for h in H]
                Hn = [h.clone() for h in old]
                scores = self.model.forward(x.clamp(min=0), Hn, None, training=False)
                H = [torch.where(valid, hn, ho) for hn, ho in zip(Hn, old)]
                yield off_o[c:c + na] + t, scores

    def clear(self):
        self.model = None
        self.gru.model = None
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def support_users(self):
        return False
