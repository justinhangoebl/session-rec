"""PyTorch port of algorithms/NCFS/ (NCFS, Keras 2.3.1 on TF1), kept as close to the original as possible.

Hu et al., Neural Cross-Session Filtering (IJCAI 2020 / session-rec's version). Keras 2.3.1 behaviour checked in
its installed source and reproduced:
  * Embedding init U(-0.05, 0.05); shared masked (mask_zero) item embedding for history and context
  * ISR attention: e = x.W + b, e / sqrt(1 + alpha e^2), exp, masked, normalised (+1e-32); W glorot with Keras'
    1-D fans (sqrt(d) each), b = 0; the attention output mask any(mask) reaches the history GRU through
    TimeDistributed, so empty history sessions are skipped
  * Keras GRU(relu): recurrent_activation sigmoid, reset_after False, kernel glorot, recurrent orthogonal
    (numpy SVD), bias 0; masked steps keep the previous state
  * BatchNormalization (eps 1e-3, momentum 0.99): batch statistics in training, plain EMA of the mean and of the
    variance times n / (n - 1.001); moving statistics at prediction
  * gate Dense(sigmoid) on [ctx, his]; loss = Keras sparse CE on softmax probabilities
    (-log_softmax(log(clip(p, 1e-7, 1-1e-7)))[0]); Keras Adam (lr 1e-3, eps 1e-7)
  * training samples, history window and the session/user bookkeeping (incl. its quirks: the last session of the
    last user is never stored, interleaved users attribute a session to the next row's user) copied verbatim
Extra parameters: seed (numpy, torch), device.
"""
import math

import numpy as np
import pandas as pd
import torch

from algorithms.STAMP_torch.stamp import TFAdam


def keras_glorot(*shape):
    if len(shape) == 2:
        fan_in, fan_out = shape
    else:
        fan_in = fan_out = math.sqrt(np.prod(shape))
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return torch.nn.Parameter((torch.rand(*shape) * 2 - 1) * limit)


def keras_orthogonal(rows, cols):
    a = np.random.normal(0.0, 1.0, (rows, cols))
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == (rows, cols) else v
    return torch.nn.Parameter(torch.tensor(q[:rows, :cols], dtype=torch.float32))


class KerasBN(torch.nn.Module):
    def __init__(self, d, momentum=0.99, eps=1e-3):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(d)); self.beta = torch.nn.Parameter(torch.zeros(d))
        self.register_buffer('moving_mean', torch.zeros(d)); self.register_buffer('moving_variance', torch.ones(d))
        self.momentum, self.eps = momentum, eps

    def forward(self, x):
        if self.training:
            mean = x.mean(0); var = x.var(0, unbiased=False)
            with torch.no_grad():
                n = float(x.shape[0])
                self.moving_mean -= (self.moving_mean - mean) * (1 - self.momentum)
                self.moving_variance -= (self.moving_variance - var * (n / (n - (1.0 + self.eps)))) * (1 - self.momentum)
        else:
            mean, var = self.moving_mean, self.moving_variance
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class CrossSessModel(torch.nn.Module):
    def __init__(self, num_items, d, max_nb_sess, att_alpha):
        super().__init__()
        P = torch.nn.Parameter
        self.item_embedding = P((torch.rand(num_items + 1, d) * 2 - 1) * 0.05)
        self.his_W = keras_glorot(d); self.his_b = P(torch.zeros(1))
        # Keras GRU(d): kernel (d, 3d) glorot, recurrent (d, 3d) orthogonal, bias zeros; gate order z, r, h
        self.gru_kernel = keras_glorot(d, 3 * d); self.gru_recurrent = keras_orthogonal(d, 3 * d); self.gru_bias = P(torch.zeros(3 * d))
        self.ctx_W = keras_glorot(d); self.ctx_b = P(torch.zeros(1))
        self.use_his = max_nb_sess > 0
        if self.use_his:
            self.bn_his = KerasBN(d); self.bn_ctx = KerasBN(d)
            self.gate_kernel = keras_glorot(2 * d, d); self.gate_bias = P(torch.zeros(d))
        self.target_embedding = P((torch.rand(num_items + 1, d) * 2 - 1) * 0.05)
        self.d, self.alpha = d, att_alpha

    def attention(self, x, mask, W, b):
        e = x @ W + b
        e = e / torch.sqrt(1 + self.alpha * e ** 2)
        a = torch.exp(e) * mask
        a = a / (a.sum(1, keepdim=True) + 1e-32)
        return (x * a[..., None]).sum(1)

    def gru(self, xs, step_mask):
        d = self.d; B, S, _ = xs.shape
        Uz, Ur, Uh = self.gru_recurrent[:, :d], self.gru_recurrent[:, d:2 * d], self.gru_recurrent[:, 2 * d:]
        xw = xs @ self.gru_kernel + self.gru_bias
        h = torch.zeros(B, d, device=xs.device)
        for t in range(S):
            x_z, x_r, x_h = xw[:, t, :d], xw[:, t, d:2 * d], xw[:, t, 2 * d:]
            z = torch.sigmoid(x_z + h @ Uz); r = torch.sigmoid(x_r + h @ Ur)
            hh = torch.relu(x_h + (r * h) @ Uh)
            hn = z * h + (1 - z) * hh
            h = torch.where(step_mask[:, t:t + 1], hn, h)
        return h

    def context(self, sess_input, his_input):
        B, S, L = his_input.shape
        E = self.item_embedding
        his_mask = (his_input != 0)
        enc = self.attention(E[his_input].reshape(B * S, L, -1), his_mask.reshape(B * S, L).float(), self.his_W, self.his_b).reshape(B, S, -1)
        his_sess_embed = self.gru(enc, his_mask.any(-1))
        ctx = self.attention(E[sess_input], (sess_input != 0).float(), self.ctx_W, self.ctx_b)
        if self.use_his:
            his_sess_embed = self.bn_his(his_sess_embed); ctx = self.bn_ctx(ctx)
            gate = torch.sigmoid(torch.cat([ctx, his_sess_embed], 1) @ self.gate_kernel + self.gate_bias)
            ctx = gate * his_sess_embed + (1 - gate) * ctx
        return ctx

    def forward(self, sess_input, his_input, target_input):
        ctx = self.context(sess_input, his_input)
        return (self.target_embedding[target_input] * ctx[:, None, :]).sum(-1)


class NCFS:
    def __init__(self, window_sz=8, max_nb_his_sess=2, mini_batch_sz=200, neg_samples=100, max_epoch=10,
                 max_session_len=None, embeding_len=100, dropout=0, att_alpha=0.01, session_key='SessionId',
                 item_key='ItemId', time_key='Time', user_key='UserId', seed=None, device=None):
        assert dropout == 0, 'GRU dropout is not ported (session-rec configs use 0)'
        self.window_sz, self.max_nb_his_sess, self.mini_batch_sz = window_sz, max_nb_his_sess, mini_batch_sz
        self.neg_samples, self.max_epoch, self.max_session_len = neg_samples, max_epoch, max_session_len
        self.embeding_len, self.att_alpha = embeding_len, att_alpha
        self.session_key, self.item_key, self.time_key, self.user_key = session_key, item_key, time_key, user_key
        self.session, self.session_items = -1, []
        self.seed = seed
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def prepare_item_maps(self, train_data):
        item_dict, reversed_item_dict, item_ctr = {}, {}, 1
        for itemId in train_data[self.item_key].values:
            if itemId not in item_dict:
                item_dict[itemId] = item_ctr; reversed_item_dict[item_ctr] = itemId; item_ctr += 1
        return item_dict, reversed_item_dict

    def init_model(self, train_data, test_data):
        if self.max_session_len is None:
            self.max_session_len = max(train_data.groupby(self.session_key).size().max(),
                                       test_data.groupby(self.session_key).size().max())
        self.nb_item = len(self.item_dict)
        self.sessionSet = dict()
        session_prev, user_prev = -1, -1
        for userId, sessionId, item in zip(train_data[self.user_key].values, train_data[self.session_key].values,
                                           train_data[self.item_key].values):
            itemId = self.item_dict[item]
            if userId not in self.sessionSet.keys():
                if user_prev != -1:
                    self.sessionSet[user_prev].append(sess)
                self.sessionSet[userId] = []
                sess = []
            elif sessionId != session_prev:
                if session_prev != -1:
                    self.sessionSet[userId].append(sess)
                sess = []
            sess.append(itemId)
            user_prev, session_prev = userId, sessionId
        self.nb_trn_spl = sum(len(v[i]) for v in self.sessionSet.values() for i in range(1, len(v)))
        self.nb_batch = math.ceil(self.nb_trn_spl / self.mini_batch_sz)
        self.model = CrossSessModel(self.nb_item, self.embeding_len, self.max_nb_his_sess, self.att_alpha).to(self.device)
        self.opt = TFAdam(self.model.parameters(), lr=0.001, eps=1e-7)

    def _train_on_batch(self, sess_input, his_input, target_input):
        self.model.train()
        dev = self.device
        scores = self.model(torch.tensor(sess_input, dtype=torch.long, device=dev), torch.tensor(his_input, dtype=torch.long, device=dev),
                            torch.tensor(target_input, dtype=torch.long, device=dev))
        p = torch.softmax(scores, dim=1).clamp(1e-7, 1 - 1e-7)
        loss = -torch.log_softmax(torch.log(p), dim=1)[:, 0].mean()       # labels are all 0
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    def fit(self, train_data, test_data, valid_data=None, **kwargs):
        if self.seed is not None:
            np.random.seed(self.seed); torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        self.item_dict, self.reversed_item_dict = self.prepare_item_maps(train_data)
        self.init_model(train_data, test_data)
        spl_idx = spl_cnt = 0
        W, S, L, K = self.window_sz, self.max_nb_his_sess, self.max_session_len, self.neg_samples
        for ep in range(self.max_epoch):
            curr_batch = 0; losses = []
            batch_sz = min(self.mini_batch_sz, self.nb_trn_spl)
            his_input = np.zeros([batch_sz, S, L], dtype=np.int32); sess_input = np.zeros([batch_sz, 2 * W], dtype=np.int32)
            target_input = np.zeros([batch_sz, K + 1], dtype=np.int32)
            for user_id, v in self.sessionSet.items():
                for i in range(1, len(v)):
                    sess = v[i]
                    for c in range(len(sess)):
                        target_input[spl_idx, 0] = sess[c]
                        target_input[spl_idx, 1:] = np.random.randint(0, self.nb_item, K)
                        ctx_sess = np.concatenate([sess[max(0, c - W):c], sess[(c + 1):min(c + 1 + W, len(sess))]])
                        sess_input[spl_idx, -len(ctx_sess):] = ctx_sess
                        for j in range(min(i, S)):
                            his_sess = v[i - 1 - j][:L]
                            his_input[spl_idx, -(j + 1), :len(his_sess)] = his_sess
                        spl_idx += 1
                        if spl_idx == batch_sz:
                            spl_idx = 0; curr_batch += 1; spl_cnt += batch_sz
                            batch_sz = min(self.mini_batch_sz, self.nb_trn_spl - spl_cnt)
                            losses.append(self._train_on_batch(sess_input, his_input, target_input))
                            his_input = np.zeros([batch_sz, S, L], dtype=np.int32); sess_input = np.zeros([batch_sz, 2 * W], dtype=np.int32)
                            target_input = np.zeros([batch_sz, K + 1], dtype=np.int32)
            spl_cnt = 0
            print('epoch %d loss %g' % (ep, np.mean(losses)))
        self.model.eval()
        self.all_items = torch.arange(0, self.nb_item + 1, device=self.device)[None]
        self.predicted_item_ids = [int(self.reversed_item_dict[idx]) for idx in range(1, self.nb_item + 1)]

    @torch.no_grad()
    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids=None, skip=False, mode_type='view', timestamp=0):
        if self.session != session_id:
            self.session = session_id; self.session_items = list()
        self.session_items.append(self.item_dict[input_item_id])
        W, S, L = self.window_sz, self.max_nb_his_sess, self.max_session_len
        his_input = np.zeros([1, S, L], dtype=np.int32); sess_input = np.zeros([1, 2 * W], dtype=np.int32)
        user_sess = self.sessionSet[input_user_id]
        c = len(self.session_items)
        ctx_sess = np.array(self.session_items[max(0, c - W):c])
        sess_input[0, -len(ctx_sess):] = ctx_sess
        for j in range(min(len(user_sess) - 1, S)):
            his_sess = user_sess[len(user_sess) - 1 - j][:L]
            his_input[0, -(j + 1), :len(his_sess)] = his_sess
        dev = self.device
        scores = self.model(torch.tensor(sess_input, dtype=torch.long, device=dev), torch.tensor(his_input, dtype=torch.long, device=dev),
                            self.all_items)[0].cpu().numpy()
        return pd.Series(data=scores[1:], index=self.predicted_item_ids)

    def clear(self):
        self.model = None

    def support_users(self):
        return True

    def predict_with_training_data(self):
        return False
