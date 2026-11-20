"""PyTorch port of algorithms/nsar/ (NSAR / UserGru, TensorFlow 1), kept as close to the original as possible.

Phuong et al., Neural Session-Aware Recommendation, 2019; session-rec's TF1 version (combination 'adaptive',
fusion 'post' are the defaults and the only path used by the configs; 'linear' is also supported).
Reproduced exactly (they change results):
  * data loader: sessions padded with zero rows to max_length+1, rows [user, item, hour, weekday, half-month]
  * tf.get_variable default init = glorot-uniform for every variable, including 1-D biases (U(+-sqrt(3/n)))
    and scalars (U(+-sqrt(3)))
  * TF1 GRUCell (gate bias 1.0, reset before the candidate matmul) through dynamic_rnn (zero outputs past length)
  * TF inverted dropout on the item and user embeddings with keep_pr during training, keep 1 at test
  * adaptive post-fusion: tanh(FF) of GRU state and user embedding, 2-way softmax gates over x.Va + ba
  * loss: mean softmax cross-entropy over ALL positions incl. padding (target class 0), as the original
  * Adam (TF1), shuffled session batches; predict_next replays the precomputed test session by position
Extra parameters: seed (numpy, torch), device.
"""
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from algorithms.STAMP_torch.stamp import TFAdam


def tf_default_init(*shape):
    """tf.get_variable default: glorot_uniform with TF's fan computation for rank 0/1/2."""
    if len(shape) == 0:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    else:
        fan_in, fan_out = shape[0], shape[1]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return torch.nn.Parameter((torch.rand(tuple(shape)) * 2 - 1) * limit)


class DataLoader:
    def __init__(self, max_length, batch_size):
        self._max_length, self._batch_size = max_length, batch_size
        self._batch_index = -1

    def load_data(self, data, user2id, item2id):
        self._data, self._num_events, self.sid_to_index = [], 0, dict()
        for sess_id in data.SessionId.unique():
            session_pd = data.loc[data['SessionId'] == sess_id]
            session = [[user2id[int(row[0])], item2id[int(row[1])]] + list(self.extract_time_context_utc(row[2]))
                       for row in session_pd[['UserId', 'ItemId', 'Time']].values]
            self._num_events += len(session)
            if len(session) < self._max_length + 1:
                for _ in range(self._max_length - len(session) + 1):
                    session.append([0] * 5)
            self.sid_to_index[sess_id] = [len(self._data)]
            self._data.append(session)
        self._data = np.array(self._data, dtype=np.int32)
        self._num_batch = int(float(len(self._data) - 1) / self._batch_size) + 1

    def next_epoch(self, shuffle=False):
        if shuffle:
            np.random.shuffle(self._data)
        self._batch_index = 0

    def next_batch(self):
        start_idx = self._batch_index * self._batch_size
        self._batch_index += 1
        if self._batch_index == self._num_batch:
            self._batch_index = -1
        return self._data[start_idx: start_idx + self._batch_size]

    def has_next(self):
        return self._batch_index != -1

    @staticmethod
    def extract_time_context_utc(utc):
        dt = datetime.utcfromtimestamp(float(utc))
        month = dt.month
        if month == 12:
            day_of_month = datetime(day=1, month=1, year=dt.year + 1) - datetime(day=1, month=month, year=dt.year)
        else:
            day_of_month = datetime(day=1, month=month + 1, year=dt.year) - datetime(day=1, month=month, year=dt.year)
        half_month_ped = month * 2 - 1 if dt.day < day_of_month.days / 2 else month * 2
        return dt.hour, dt.weekday(), half_month_ped

    def data_from_sid(self, sid):
        return self._data[self.sid_to_index[sid]]


class UserGruModel(torch.nn.Module):
    def __init__(self, num_users, num_items, max_length, entity_embedding, context_embedding, hidden_units, combination):
        super().__init__()
        H, E = hidden_units, entity_embedding
        self.H, self.T, self.combination = H, max_length, combination
        self.Ei = tf_default_init(num_items + 1, E); self.Eu = tf_default_init(num_users + 1, E)
        self.Ed = tf_default_init(8, context_embedding); self.Em = tf_default_init(25, context_embedding)  # unused by 'adaptive'
        # TF1 GRUCell(H) on input depth E
        self.gates_kernel = tf_default_init(E + H, 2 * H); self.gates_bias = torch.nn.Parameter(torch.ones(2 * H))
        self.cand_kernel = tf_default_init(E + H, H); self.cand_bias = torch.nn.Parameter(torch.zeros(H))
        if combination == 'adaptive':
            self.w_a_item = tf_default_init(H, H); self.b_a_item = tf_default_init(H)
            self.w_a_user = tf_default_init(E, H); self.b_a_user = tf_default_init(H)
            self.Va_i = tf_default_init(H); self.Va_u = tf_default_init(H)
            self.ba_i = tf_default_init(); self.ba_u = tf_default_init()
            fc_in = H + E
        elif combination == 'linear':
            fc_in = H + E
        else:
            raise NotImplementedError(combination)
        self.w_fc = tf_default_init(fc_in, num_items + 1); self.b_fc = tf_default_init(num_items + 1)

    def gru_cell(self, x, h):
        value = torch.sigmoid(torch.cat([x, h], 1) @ self.gates_kernel + self.gates_bias)
        r, u = value[:, :self.H], value[:, self.H:]
        c = torch.tanh(torch.cat([x, r * h], 1) @ self.cand_kernel + self.cand_bias)
        return u * h + (1 - u) * c

    def forward(self, user, item, next_items, keep_pr):
        """returns logits (B*T, num_items+1)"""
        B, T = item.shape
        e_i, e_u = self.Ei[item], self.Eu[user]
        if keep_pr < 1:
            e_i = torch.nn.functional.dropout(e_i, p=1 - keep_pr, training=True)
            e_u = torch.nn.functional.dropout(e_u, p=1 - keep_pr, training=True)
        length = (next_items != 0).sum(1)
        h = torch.zeros(B, self.H, device=item.device); outs = []
        for t in range(T):
            hn = self.gru_cell(e_i[:, t], h)
            m = (t < length)[:, None]
            h = torch.where(m, hn, h)
            outs.append(torch.where(m, hn, torch.zeros_like(hn)))
        output_states = torch.stack(outs, 1)                                 # (B, T, H)
        if self.combination == 'linear':
            final_state = torch.cat([output_states, e_u], -1).reshape(-1, self.H + e_u.shape[-1])
        else:
            it = torch.tanh(output_states.reshape(-1, self.H) @ self.w_a_item + self.b_a_item).reshape(B, T, -1)
            us = torch.tanh(e_u.reshape(-1, e_u.shape[-1]) @ self.w_a_user + self.b_a_user).reshape(B, T, -1)
            a_i = (it * self.Va_i).sum(2) + self.ba_i
            a_u = (us * self.Va_u).sum(2) + self.ba_u
            sum_exp = torch.exp(a_i) + torch.exp(a_u)
            al_i, al_u = torch.exp(a_i) / sum_exp, torch.exp(a_u) / sum_exp
            final_state = torch.cat([al_i[..., None] * it, al_u[..., None] * us], -1).reshape(-1, 2 * self.H)
        return final_state @ self.w_fc + self.b_fc


class NSAR:
    def __init__(self, cell='gru', name='baseline', context_embedding=5, hidden_units=100, num_layers=1,
                 combination='adaptive', fusion_type='post', learning_rate=0.001, num_epoch=20, display_every=500,
                 eval_every=1, keep_pr=0.25, batch_size=64, mode='train', session_key='SessionId', item_key='ItemId',
                 user_key='UserId', time_key='Time', seed=None, device=None):
        assert cell == 'gru' and num_layers == 1 and fusion_type == 'post', 'only the configuration used by session-rec is ported'
        self.session_key, self.item_key, self.time_key, self.user_key = session_key, item_key, time_key, user_key
        self.combination, self.context_embedding, self.hidden_units = combination, context_embedding, hidden_units
        self.entity_embedding = hidden_units
        self.learning_rate, self.keep_pr, self.num_epoch, self.batch_size = learning_rate, keep_pr, num_epoch, batch_size
        self.seed = seed
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def _t(self, a):
        return torch.tensor(np.ascontiguousarray(a), dtype=torch.long, device=self.device)

    def fit(self, train_data, test_data=None):
        if self.seed is not None:
            np.random.seed(self.seed); torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        max_train_length = train_data.groupby([self.session_key])[self.item_key].count().max()
        max_test_length = test_data.groupby([self.session_key])[self.item_key].count().max()
        self.max_length = max(max_train_length, max_test_length)
        self.itemids = train_data[self.item_key].unique()
        assert len(np.setdiff1d(test_data[self.item_key].unique(), self.itemids, assume_unique=True)) == 0
        self.userids = train_data[self.user_key].unique()
        assert len(np.setdiff1d(test_data[self.user_key].unique(), self.userids, assume_unique=True)) == 0
        self.item2id = dict(zip(self.itemids, range(1, len(self.itemids) + 1)))
        self.user2id = dict(zip(self.userids, range(1, len(self.userids) + 1)))
        self.model = UserGruModel(len(self.user2id), len(self.itemids), self.max_length, self.entity_embedding,
                                  self.context_embedding, self.hidden_units, self.combination).to(self.device)
        self.train_loader = DataLoader(self.max_length, self.batch_size)
        self.train_loader.load_data(train_data, self.user2id, self.item2id)
        opt = TFAdam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epoch):
            start = time.time(); losses = []
            self.train_loader.next_epoch(shuffle=True)
            while self.train_loader.has_next():
                b = self.train_loader.next_batch()
                next_items = self._t(b[:, 1:, 1])
                logits = self.model(self._t(b[:, :-1, 0]), self._t(b[:, :-1, 1]), next_items, self.keep_pr)
                loss = torch.nn.functional.cross_entropy(logits, next_items.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
                if math.isnan(losses[-1]):
                    print('loss is nan'); break
            print('++ Epoch: {} - Loss: {:.5f} - Time: {:.5f} ++'.format(epoch, np.mean(losses), time.time() - start))
        self.model.eval()
        self.test_loader = DataLoader(self.max_length, self.batch_size)
        self.test_loader.load_data(test_data, self.user2id, self.item2id)
        self.current_session = -1
        self.index = list(self.item2id.keys())

    @torch.no_grad()
    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids=None, timestamp=None):
        if self.current_session != session_id:
            self.current_session = session_id; self.pos = 0
            s = self.test_loader.data_from_sid(session_id)
            logits = self.model(self._t(s[:, :-1, 0]), self._t(s[:, :-1, 1]), self._t(s[:, 1:, 1]), 1.0)
            self.pr = torch.softmax(logits, dim=1).cpu().numpy()
            assert len(self.pr) != 1
        else:
            self.pos += 1
        return pd.Series(data=self.pr[self.pos, 1:], index=self.index)

    def support_users(self):
        return True

    def predict_with_training_data(self):
        return False

    def clear(self):
        self.current_session = -1
        self.model = None
