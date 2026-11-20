"""Adapter that lets the official GRU4Rec PyTorch implementation run inside this repo's evaluation loop.

The official class (gru4rec_pytorch.GRU4Rec, vendored unchanged) trains on a DataFrame and evaluates in batches.
This runner calls fit(train, test) and then predict_next(session_id, item_id, items_to_predict) once per event,
so the adapter keeps the GRU hidden state of the *current* session and resets it when the session id changes.

YAML:
  - class: gru4rec_torch.adapter.GRU4RecTorch
    params: {loss: cross-entropy, layers: [100], n_epochs: 10, batch_size: 32, dropout_p_hidden: 0.4,
             learning_rate: 0.2, momentum: 0.2, n_sample: 2048, sample_alpha: 0.5, bpreg: 0.0, logq: 1.0,
             constrained_embedding: True, embedding: 0, elu_param: 0, seed: 1, device: 'cuda:0'}
All params except seed/device go to the official constructor unchanged. See official_paramfiles/ for the authors' best
parameters (note: the PyTorch version only has the 'cross-entropy' and 'bpr-max' losses, not 'top1-max').
"""
import numpy as np
import pandas as pd
import torch

from algorithms.gru4rec_torch.gru4rec_pytorch import GRU4Rec


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
        self.itemidmap = self.gru.data_iterator.itemidmap          # ItemId -> model row
        self.item_ids = self.itemidmap.index.values                # model row -> ItemId
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

    def clear(self):
        self.model = None
        self.gru.model = None
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def support_users(self):
        return False
