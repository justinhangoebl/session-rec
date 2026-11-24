"""PyTorch port of session-rec's HGRU4Rec (Theano), kept as close to the original as possible.

Quadrana et al., Personalizing Session-based Recommendations with Hierarchical RNNs, RecSys 2017; session-rec's
version. Ported for the configuration space session-rec uses: one-hot item input (item_embedding=None), one session
and one user layer, n_sample=0, adagrad (+ momentum), losses top1 / bpr / cross-entropy.
Reproduced exactly (they change results):
  * initialisation order and distributions from a numpy RandomState(seed) (U(+-sqrt(6/(fan_in+fan_out))))
  * Theano gate layout (gates in the upper slices, reset = first half), user GRU updated only at session start and
    reset for new users, session GRU initialised from tanh(h_u Ws_init + Bs_init); propagation 'all' adds the user
    bias to the session gates only
  * inverted dropout on h_u, h_s_init and h_s in training
  * the original top1/bpr/xe losses on the B x B in-batch score matrix (B = batch_size)
  * optimizer: adagrad scaling g / sqrt(acc + 1e-6) then momentum; sampled rows (Ws_in[0][X], Wsy[Y], By[Y]) use
    set_subtensor for accumulators / velocities (last duplicate wins) and inc_subtensor for the parameters
  * user-parallel mini-batching (iterate) copied verbatim
Extra parameters: device (default cpu: batch_size 5 is faster there), torch_seed (dropout RNG). `seed` seeds the
init RandomState exactly as the original (default 42, so the initial weights are identical to the original's).
"""
import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class HGRU4Rec:
    def __init__(self, session_layers=100, user_layers=100, n_epochs=10, batch_size=50, learning_rate=0.05, momentum=0.0,
                 adapt='adagrad', decay=0.9, grad_cap=0, sigma=0, dropout_p_hidden_usr=0.0, dropout_p_hidden_ses=0.0,
                 dropout_p_init=0.0, init_as_normal=False, reset_after_session=True, loss='top1', hidden_act='tanh',
                 final_act=None, train_random_order=False, lmbd=0.0, session_key='SessionId', item_key='ItemId',
                 time_key='Time', user_key='UserId', n_sample=0, sample_alpha=0.75, item_embedding=None,
                 init_item_embeddings=None, user_propagation_mode='init', user_to_output=False,
                 user_to_session_act='tanh', seed=42, torch_seed=None, device='cpu'):
        assert adapt == 'adagrad' and grad_cap == 0 and n_sample == 0 and item_embedding is None and not user_to_output, \
            'only the configuration space used by session-rec is ported'
        self.session_layers, self.user_layers = int(session_layers), int(user_layers)
        self.n_epochs, self.batch_size, self.learning_rate, self.momentum = n_epochs, batch_size, learning_rate, momentum
        self.sigma, self.init_as_normal, self.lmbd = sigma, init_as_normal, lmbd
        self.dropout_p_hidden_usr, self.dropout_p_hidden_ses, self.dropout_p_init = dropout_p_hidden_usr, dropout_p_hidden_ses, dropout_p_init
        self.session_key, self.item_key, self.time_key, self.user_key = session_key, item_key, time_key, user_key
        self.user_propagation_mode = user_propagation_mode
        self.loss, self.final_act = loss, final_act
        self.hidden_act = torch.relu if hidden_act == 'relu' else torch.tanh
        self.s_init_act = torch.relu if user_to_session_act == 'relu' else torch.tanh
        self.seed, self.torch_seed = seed, torch_seed
        self.rng = np.random.RandomState(seed=seed)
        self.device = torch.device(device)

    def init_matrix(self, shape):
        sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (shape[0] + shape[1]))
        if self.init_as_normal:
            return np.asarray(self.rng.randn(*shape) * sigma, dtype=np.float32)
        return np.asarray(self.rng.rand(*shape) * sigma * 2 - sigma, dtype=np.float32)

    def init_params(self):
        Hs, Hu, n = self.session_layers, self.user_layers, self.n_items
        t = lambda a: torch.tensor(a, device=self.device)
        p = {}
        p['Ws_in'] = t(np.hstack([self.init_matrix((n, Hs)) for _ in range(3)]))
        p['Ws_hh'] = t(self.init_matrix((Hs, Hs)))
        p['Ws_rz'] = t(np.hstack([self.init_matrix((Hs, Hs)) for _ in range(2)]))
        p['Bs_h'] = torch.zeros(3 * Hs, device=self.device)
        p['Wsy'] = t(self.init_matrix((n, Hs)))
        p['By'] = torch.zeros(n, 1, device=self.device)
        p['Wu_in'] = t(np.hstack([self.init_matrix((Hs, Hu)) for _ in range(3)]))
        p['Wu_hh'] = t(self.init_matrix((Hu, Hu)))
        p['Wu_rz'] = t(np.hstack([self.init_matrix((Hu, Hu)) for _ in range(2)]))
        p['Bu_h'] = torch.zeros(3 * Hu, device=self.device)
        p['Ws_init'] = t(self.init_matrix((Hu, Hs)))
        p['Bs_init'] = torch.zeros(Hs, device=self.device)
        if self.user_propagation_mode == 'all':
            p['Wu_to_s'] = t(np.hstack([self.init_matrix((Hu, Hs)) for _ in range(3)]))
        self.p = p
        self.dense_names = [k for k in ['Ws_hh', 'Ws_rz', 'Bs_h', 'Ws_init', 'Bs_init', 'Wu_in', 'Wu_hh', 'Wu_rz', 'Bu_h', 'Wu_to_s'] if k in p]
        self.acc = {k: torch.zeros_like(v) for k, v in p.items()}
        self.vel = {k: torch.zeros_like(v) for k, v in p.items()}

    def reset_states(self, batch):
        self.Hs = torch.zeros(batch, self.session_layers, device=self.device)
        self.Hu = torch.zeros(batch, self.user_layers, device=self.device)

    @staticmethod
    def dropout(X, drop_p):
        if drop_p > 0:
            retain = 1 - drop_p
            X = X * torch.bernoulli(torch.full_like(X, retain)) / retain
        return X

    def model(self, p, Sx, Sstart, Ustart, Hs, Hu, Ssy, SBy, train):
        Hs_n, Hu_n = self.session_layers, self.user_layers
        dp = lambda x, q: self.dropout(x, q) if train else x
        S, U = Sstart[:, None], Ustart[:, None]
        user_in = Hs @ p['Wu_in'] + p['Bu_h']
        rz_u = torch.sigmoid(user_in[:, Hu_n:] + Hu @ p['Wu_rz'])
        h_u = self.hidden_act((Hu * rz_u[:, :Hu_n]) @ p['Wu_hh'] + user_in[:, :Hu_n])
        z = rz_u[:, Hu_n:]
        h_u = (1.0 - z) * Hu + z * h_u
        h_u = dp(h_u, self.dropout_p_hidden_usr)
        h_u = Hu * (1 - S) + h_u * S
        h_u = torch.zeros_like(h_u) * U + h_u * (1 - U)
        session_in = Sx + p['Bs_h']
        h_s_init = dp(self.s_init_act(h_u @ p['Ws_init'] + p['Bs_init']), self.dropout_p_init)
        h_s = Hs * (1 - S) + h_s_init * S
        h_s = h_s * (1 - U) + torch.zeros_like(h_s) * U
        gates = session_in[:, Hs_n:] + h_s @ p['Ws_rz']
        if self.user_propagation_mode == 'all':
            gates = (h_u @ p['Wu_to_s'])[:, Hs_n:] + gates
        rz_s = torch.sigmoid(gates)
        h_s_cand = self.hidden_act((h_s * rz_s[:, :Hs_n]) @ p['Ws_hh'] + session_in[:, :Hs_n])
        z = rz_s[:, Hs_n:]
        h_s = (1.0 - z) * h_s + z * h_s_cand
        h_s = dp(h_s, self.dropout_p_hidden_ses)
        preact = h_s @ Ssy.T + SBy.flatten()
        return h_s, h_u, self.final_activation(preact)

    def final_activation(self, X):
        if self.loss == 'cross-entropy':
            if self.final_act == 'tanh':
                X = torch.tanh(X)
            e_x = torch.exp(X - X.max(dim=1, keepdim=True).values)
            return e_x / e_x.sum(dim=1, keepdim=True)
        if self.final_act == 'linear':
            return X
        if self.final_act == 'relu':
            return torch.relu(X)
        return torch.tanh(X)

    def loss_function(self, yhat):
        if self.loss == 'cross-entropy':
            return torch.mean(-torch.log(torch.diag(yhat) + 1e-24))
        if self.loss == 'bpr':
            return torch.mean(-torch.log(torch.sigmoid(torch.diag(yhat) - yhat.T)))
        yhatT = yhat.T
        return torch.mean(torch.mean(torch.sigmoid(-torch.diag(yhat) + yhatT) + torch.sigmoid(yhatT ** 2), dim=0)
                          - torch.sigmoid(torch.diag(yhat) ** 2) / self.batch_size)

    # the original's RMSprop() is actually adagrad (accumulated squared gradients)
    @staticmethod
    def _last_occurrence(idx):
        rev = idx.flip(0)
        uniq, inv = torch.unique(rev, return_inverse=True)
        pos = torch.full((len(uniq),), len(idx), dtype=torch.long, device=idx.device)
        pos.scatter_reduce_(0, inv, torch.arange(len(idx), device=idx.device), reduce='amin', include_self=True)
        return uniq, len(idx) - 1 - pos

    def _sparse_update(self, name, idx, g):
        lr, mom = np.float32(self.learning_rate), self.momentum
        acc_new = self.acc[name][idx] + g ** 2
        g = g / torch.sqrt(acc_new + 1e-6)
        delta = lr * (g + self.lmbd * self.p[name][idx]) if self.lmbd > 0 else lr * g
        uniq, last = self._last_occurrence(idx)
        self.acc[name][uniq] = acc_new[last]
        if mom > 0:
            velocity2 = mom * self.vel[name][idx] - delta
            self.vel[name][uniq] = velocity2[last]
            self.p[name].index_add_(0, idx, velocity2)
        else:
            self.p[name].index_add_(0, idx, -delta)

    def _dense_update(self, name, g):
        lr, mom = np.float32(self.learning_rate), self.momentum
        self.acc[name] += g ** 2
        g = g / torch.sqrt(self.acc[name] + 1e-6)
        if mom > 0:
            self.vel[name] = mom * self.vel[name] - lr * (g + self.lmbd * self.p[name])
            self.p[name] += self.vel[name]
        else:
            self.p[name] = self.p[name] * np.float32(1.0 - self.learning_rate * self.lmbd) - lr * g

    def train_step(self, in_idx, sstart, ustart, y):
        dev = self.device
        X = torch.tensor(in_idx, dtype=torch.long, device=dev); Y = torch.tensor(y, dtype=torch.long, device=dev)
        S = torch.tensor(sstart, device=dev); U = torch.tensor(ustart, device=dev)
        p = {k: (v.clone().requires_grad_(True) if k in self.dense_names else v) for k, v in self.p.items()}
        Sx = self.p['Ws_in'][X].clone().requires_grad_(True)
        Ssy = self.p['Wsy'][Y].clone().requires_grad_(True)
        SBy = self.p['By'][Y].clone().requires_grad_(True)
        h_s, h_u, yhat = self.model(p, Sx, S, U, self.Hs, self.Hu, Ssy, SBy, train=True)
        cost = self.loss_function(yhat)
        grads = torch.autograd.grad(cost, [p[k] for k in self.dense_names] + [Sx, Ssy, SBy])
        with torch.no_grad():
            for k, g in zip(self.dense_names, grads[:len(self.dense_names)]):
                self._dense_update(k, g)
            gx, gy, gb = grads[len(self.dense_names):]
            self._sparse_update('Ws_in', X, gx); self._sparse_update('Wsy', Y, gy); self._sparse_update('By', Y, gb)
        self.Hs, self.Hu = h_s.detach(), h_u.detach()
        return cost.item()

    def preprocess_data(self, data):
        data.sort_values([self.user_key, self.session_key, self.time_key], inplace=True)
        data.reset_index(drop=True, inplace=True)
        offset_session = np.r_[0, data.groupby([self.user_key, self.session_key], sort=False).size().cumsum()[:-1]]
        user_indptr = np.r_[0, data.groupby(self.user_key, sort=False)[self.session_key].nunique().cumsum()[:-1]]
        return user_indptr, offset_session

    def fit(self, train_data, test_data=None, valid_data=None, **kwargs):
        torch.manual_seed(self.torch_seed if self.torch_seed is not None else self.seed)
        self.predict = None
        self.error_during_train = False
        itemids = train_data[self.item_key].unique()
        self.n_items = len(itemids)
        self.init_params()
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        train_data = pd.merge(train_data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                              on=self.item_key, how='inner')
        user_indptr, offset_sessions = self.preprocess_data(train_data)
        for epoch in range(self.n_epochs):
            train_cost = self.iterate(train_data, offset_sessions, user_indptr)
            if np.isnan(train_cost):
                return
            logger.info('Epoch {} - train cost: {:.4f}'.format(epoch, train_cost))
            print('Epoch {} - train cost: {:.4f}'.format(epoch, train_cost))

    def iterate(self, data, offset_sessions, user_indptr):
        self.reset_states(self.batch_size)
        n_users = len(user_indptr)
        offset_users = offset_sessions[user_indptr]
        user_idx_arr = np.arange(n_users - 1)
        user_iters = np.arange(self.batch_size)
        user_maxiter = user_iters.max()
        user_start = offset_users[user_idx_arr[user_iters]]
        user_end = offset_users[user_idx_arr[user_iters] + 1]
        session_iters = user_indptr[user_iters]
        session_start = offset_sessions[session_iters]
        session_end = offset_sessions[session_iters + 1]
        sstart = np.zeros((self.batch_size,), dtype=np.float32)
        ustart = np.zeros((self.batch_size,), dtype=np.float32)
        finished = False
        c = []
        item_idx = data.ItemIdx.values
        while not finished:
            session_minlen = (session_end - session_start).min()
            out_idx = item_idx[session_start]
            for i in range(session_minlen - 1):
                in_idx = out_idx
                out_idx = item_idx[session_start + i + 1]
                cost = self.train_step(in_idx, sstart, ustart, out_idx)
                sstart = np.zeros_like(sstart, dtype=np.float32)
                ustart = np.zeros_like(ustart, dtype=np.float32)
                c.append(cost)
                if np.isnan(cost):
                    logger.error('NaN error!'); self.error_during_train = True
                    return np.nan
            session_start = session_start + session_minlen - 1
            session_start_mask = np.arange(len(session_iters))[(session_end - session_start) <= 1]
            sstart[session_start_mask] = 1
            for idx in session_start_mask:
                session_iters[idx] += 1
                if session_iters[idx] + 1 >= len(offset_sessions):
                    finished = True
                    break
                session_start[idx] = offset_sessions[session_iters[idx]]
                session_end[idx] = offset_sessions[session_iters[idx] + 1]
            user_change_mask = np.arange(len(user_iters))[(user_end - session_start <= 0)]
            ustart[user_change_mask] = 1
            for idx in user_change_mask:
                user_maxiter += 1
                if user_maxiter + 1 >= len(offset_users):
                    finished = True
                    break
                user_iters[idx] = user_maxiter
                user_start[idx] = offset_users[user_maxiter]
                user_end[idx] = offset_users[user_maxiter + 1]
                session_iters[idx] = user_indptr[user_maxiter]
                session_start[idx] = offset_sessions[session_iters[idx]]
                session_end[idx] = offset_sessions[session_iters[idx] + 1]
        return np.mean(c)

    @torch.no_grad()
    def predict_next_batch(self, session_ids, input_item_ids, input_user_ids, predict_for_item_ids=None, batch=100):
        if self.error_during_train:
            raise Exception
        if self.predict is None or self.predict_batch != batch:
            self.reset_states(batch)
            self.current_session = np.ones(batch) * -1
            self.current_users = np.ones(batch) * -1
            self.predict_batch = batch
            self.predict = True
        session_change = (session_ids != self.current_session).astype(np.float32)
        self.current_session = session_ids.copy()
        user_change = (input_user_ids != self.current_users).astype(np.float32)
        self.current_users = input_user_ids.copy()
        dev = self.device
        X = torch.tensor(self.itemidmap[input_item_ids].values, dtype=torch.long, device=dev)
        if predict_for_item_ids is not None:
            Y = torch.tensor(self.itemidmap[predict_for_item_ids].values, dtype=torch.long, device=dev)
            Ssy, SBy, index = self.p['Wsy'][Y], self.p['By'][Y], predict_for_item_ids
        else:
            Ssy, SBy, index = self.p['Wsy'], self.p['By'], self.itemidmap.index
        h_s, h_u, yhat = self.model(self.p, self.p['Ws_in'][X], torch.tensor(session_change, device=dev),
                                    torch.tensor(user_change, device=dev), self.Hs, self.Hu, Ssy, SBy, train=False)
        self.Hs, self.Hu = h_s, h_u
        return pd.DataFrame(data=yhat.cpu().numpy().T, index=index)

    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids=None, skip=False, mode_type='view', timestamp=0):
        if input_item_id not in self.itemidmap.index:
            return None
        return self.predict_next_batch(np.array([session_id]), np.array([input_item_id]), np.array([input_user_id]),
                                       predict_for_item_ids, 1)[0]

    def clear(self):
        self.p = None

    def support_users(self):
        return True

    def predict_with_training_data(self):
        return True
