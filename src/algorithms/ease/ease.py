'''
EASE (Steck, WWW 2019): closed-form item-item model B = argmin ||X - XB||^2 + l2 ||B||^2, diag(B) = 0, i.e.
P = (X^T X + l2 I)^-1, B = -P / diag(P), diag(B) = 0. The score of item j for a query x is (x B)_j.

Training rows (rows): 'session' (one binary row per training session) or 'user' (one row per user, all of their
training items). Queries: the items of the current session so far (weight 1 each) and, with use_history, the
items of the user's training sessions (history_weight in total, spread evenly over them, so a long history does
not drown the prefix) - "history + prefix". use_history makes the
model session-aware (support_users), so it needs evaluation_user_based and data with UserId.

Parameters: l2 (regularisation), rows, use_history, history_weight, recency_decay (None: every prefix item weighs
1, as in EASE; d in (0, 1]: the item k steps before the last one weighs d^k), max_items (keep only the max_items most
popular items; None = all), dtype of the solve ('float32'; on LASTFM float64 gives the same metrics at
10x the time on this GPU; B is stored as float32). Memory: 2 dense n_items^2 matrices on the device during the fit
(max_items 80000: 26 GB each; beyond 4 n^2 of free device memory the inverse runs on the cpu), device ('cuda:0' / 'cpu').

Saving: B is a dense n_items x n_items matrix (10 GB for 50k items), so the runner does not pickle the object
(pickle_model = False); the .pt file of evaluation/model_store.py holds export() = {item_ids, B, ...} and
EASE.load(path) rebuilds a working model from it.
'''
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


class EASE:
    pickle_model = False  # the .pt export holds the full model

    def __init__(self, l2=500.0, rows='session', use_history=False, history_weight=1.0, recency_decay=None, max_items=None,
                 dtype='float32', device=None, session_key='SessionId', item_key='ItemId', user_key='UserId'):
        assert rows in ('session', 'user'), rows
        self.l2 = float(l2)
        self.rows = rows
        self.use_history = bool(use_history)
        self.history_weight = float(history_weight)
        self.recency_decay = None if recency_decay is None else float(recency_decay)
        self.max_items = max_items
        self.dtype = getattr(torch, dtype)
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.session_key, self.item_key, self.user_key = session_key, item_key, user_key
        self.B = None

    def fit(self, data, test=None):
        t0 = time.time()
        items = data[self.item_key].value_counts()
        if self.max_items is not None:
            items = items.iloc[:int(self.max_items)]
        itemids = data[self.item_key].unique()
        itemids = itemids[np.isin(itemids, items.index)]  # training order, as the evaluation's item list
        self.itemidmap = pd.Series(np.arange(len(itemids)), index=itemids)
        data = data[data[self.item_key].isin(itemids)]

        row_key = self.session_key if self.rows == 'session' else self.user_key
        row_idx, _ = pd.factorize(data[row_key])
        col_idx = self.itemidmap[data[self.item_key].values].values
        X = sp.csr_matrix((np.ones(len(data), dtype=np.float32), (row_idx, col_idx)),
                          shape=(row_idx.max() + 1, len(itemids)))
        X.data[:] = 1.0  # binary: repeated plays of an item in a row count once
        G = self._gram(X)

        # P = (G + l2 I)^-1 via Cholesky (G is SPD): inverted on the cpu when 4 n^2 does not fit on the device
        G.diagonal().add_(self.l2)
        n = G.shape[0]
        if self.device.type == 'cuda' and 4 * n * n * G.element_size() > torch.cuda.mem_get_info(self.device)[0]:
            # cholesky + cholesky_inverse peak at ~4 n x n buffers besides G: invert on the cpu, then move B back
            print('EASE: {} items, inverting on the cpu'.format(n))
            G = G.cpu()
        L = torch.linalg.cholesky(G)
        del G
        P = torch.cholesky_inverse(L)
        del L
        P.div_(-P.diagonal().clone())  # B_ij = -P_ij / P_jj
        P.diagonal().zero_()
        self.B = P.float().to(self.device)
        del P
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        if self.use_history:
            hist = data.groupby(self.user_key)[self.item_key].unique()
            self.history = {u: self.itemidmap[v].values for u, v in hist.items()}
        else:
            self.history = {}
        print('EASE: {} rows x {} items, l2 {}, fit {:.1f}s on {}'.format(X.shape[0], X.shape[1], self.l2,
                                                                          time.time() - t0, self.device))
        self.session_id = None
        self._pred_ids = None

    def _gram(self, X, max_chunk_bytes=2 ** 31):
        '''G = X^T X on the device, in dense row chunks of the binary X (TF32 is exact for 0/1 entries).'''
        n = X.shape[1]
        G = torch.zeros((n, n), dtype=self.dtype, device=self.device)
        rows = max(256, max_chunk_bytes // (4 * n))
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.dtype == torch.float32
        try:
            for s in range(0, X.shape[0], rows):
                sub = X[s:s + rows].tocoo()
                Xc = torch.zeros((sub.shape[0], n), dtype=self.dtype, device=self.device)
                Xc[torch.as_tensor(sub.row, dtype=torch.long, device=self.device),
                   torch.as_tensor(sub.col, dtype=torch.long, device=self.device)] = 1
                G.addmm_(Xc.T, Xc)
                del Xc
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        return G

    def support_users(self):
        return self.use_history

    def predict_with_training_data(self):
        return False

    @torch.no_grad()
    def predict_next(self, session_id, input_item_id, *args, timestamp=0, skip=False, mode_type='view'):
        '''session-based: (session_id, item, items_to_predict); session-aware: (session_id, item, user, items).'''
        if self.support_users():
            input_user_id, predict_for_item_ids = args
        else:
            input_user_id, (predict_for_item_ids,) = None, args
        if session_id != self.session_id:
            self.session_id = session_id
            self.session_items = []
            self.session_hist = self.history.get(input_user_id, np.empty(0, dtype=np.int64))
        idx = self.itemidmap.get(input_item_id)
        if idx is not None:
            if idx in self.session_items:  # a repeated item counts once, at its latest position
                self.session_items.remove(idx)
            self.session_items.append(int(idx))
        if skip:
            return None

        scores = self.score(self.session_items, self.session_hist)
        if self._pred_ids is not predict_for_item_ids:
            self._pred_ids = predict_for_item_ids
            pred_idx = self.itemidmap.reindex(predict_for_item_ids).values
            self._pred_known = ~np.isnan(pred_idx)
            self._pred_rows = torch.as_tensor(pred_idx[self._pred_known].astype(np.int64), device=self.device)
        out = np.full(len(predict_for_item_ids), -np.inf, dtype=np.float32)
        out[self._pred_known] = scores[self._pred_rows].cpu().numpy()
        return pd.Series(out, index=predict_for_item_ids)

    @torch.no_grad()
    def score(self, prefix_idx, history_idx=()):
        '''Scores of all items (model order) for a query of prefix items (weight 1) + history items.'''
        x = torch.zeros(self.B.shape[0], device=self.device)
        if len(history_idx) and self.use_history:
            x[torch.as_tensor(np.array(history_idx, dtype=np.int64), device=self.device)] = self.history_weight / len(history_idx)
        if len(prefix_idx):
            w = 1.0 if self.recency_decay is None else torch.as_tensor(
                self.recency_decay ** np.arange(len(prefix_idx) - 1, -1, -1), dtype=torch.float32, device=self.device)
            x[torch.as_tensor(np.asarray(prefix_idx, dtype=np.int64), device=self.device)] = w
        return x @ self.B

    # ---------------------------------------------------------------- candidate pools (evaluation/candidates.py)

    @property
    def uses_history(self):
        return self.use_history

    def candidate_item_ids(self):
        return self.itemidmap.index.values

    @torch.no_grad()
    def score_sessions(self, sessions, histories=None, chunk=1024):
        '''Scores after every prefix of every session (column indices, -1 = unknown), as predict_next / score.'''
        n = self.B.shape[0]
        # with a recency decay, items far back weigh < 1e-12: keep the last max_prefix distinct items only
        max_prefix = None if self.recency_decay is None or self.recency_decay >= 1 else \
            max(1, int(np.ceil(np.log(1e-12) / np.log(max(self.recency_decay, 1e-6)))))
        queries = []  # (session, prefix item list)
        for si, seq in enumerate(sessions):
            order = []
            for t in range(len(seq) - 1):
                c = int(seq[t])
                if c >= 0:
                    if c in order:
                        order.remove(c)
                    order.append(c)
                    if max_prefix is not None and len(order) > max_prefix:
                        order.pop(0)
                queries.append((si, list(order)))
        for s in range(0, len(queries), chunk):
            part = queries[s:s + chunk]
            X = torch.zeros((len(part), n), device=self.device)
            if histories is not None and self.use_history:
                for r, (si, _) in enumerate(part):
                    h = histories[si]
                    if len(h):
                        X[r, torch.as_tensor(h, device=self.device)] = self.history_weight / len(h)
            rows, cols, vals = [], [], []
            for r, (_, order) in enumerate(part):
                m = len(order)
                rows.extend([r] * m)
                cols.extend(order)
                vals.extend([1.0] * m if self.recency_decay is None else
                            (self.recency_decay ** np.arange(m - 1, -1, -1)).tolist())
            if rows:
                X[torch.as_tensor(rows, device=self.device), torch.as_tensor(cols, device=self.device)] = \
                    torch.as_tensor(vals, dtype=torch.float32, device=self.device)
            yield np.arange(s, s + len(part)), X @ self.B

    @torch.no_grad()
    def export(self):
        '''item_ids (original ids, row/column i <-> item_ids[i]), B [n_items, n_items] and the user histories.'''
        return {'item_ids': self.itemidmap.index.values, 'B': self.B.cpu(),
                'history': {u: v for u, v in self.history.items()},
                'params': {'l2': self.l2, 'rows': self.rows, 'use_history': self.use_history,
                           'history_weight': self.history_weight, 'recency_decay': self.recency_decay,
                           'max_items': self.max_items}}

    @classmethod
    def load(cls, path, device=None):
        '''Rebuild a fitted EASE from the .pt file written by the runner.'''
        ex = torch.load(path, weights_only=False)['export']
        m = cls(device=device, **ex['params'])
        m.itemidmap = pd.Series(np.arange(len(ex['item_ids'])), index=ex['item_ids'])
        m.B = ex['B'].to(m.device)
        m.history = ex['history']
        m.session_id, m._pred_ids = None, None
        return m

    def clear(self):
        self.B = None
        self.history = {}
        self._pred_ids = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
