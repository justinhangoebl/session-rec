"""PyTorch port of session-rec's SR-GNN (TensorFlow 1), kept as close to the original as possible.

Wu et al., Session-based Recommendation with Graph Neural Networks, AAAI 2019; session-rec's TF1 version.
Data preparation and batching reuse the original numpy code (utils.py) unchanged.
Reproduced exactly (they change results):
  * TF1 GRUCell: gates sigmoid([x,h] Wg + bg) with bg initialised to 1.0, candidate tanh([x, r*h] Wc + bc),
    h' = u*h + (1-u)*c; kernels glorot-uniform (tf.get_variable default), model weights U(-1/sqrt(d), 1/sqrt(d))
  * L2 = l2 * sum(||v||^2 / 2) over ALL trainable variables (the original name filter never matches TF names)
  * Adam (TF1 formulation) with staircase exponential decay, decay_steps = lr_dc_step * n_train / batch_size
  * batch_predict=True: test scores computed in batches after training and returned in order by a counter
    (the original does the same, incl. the overlapping last batch); batch_predict=False: per-event graph
Extra parameters: seed (numpy, torch), device.
"""
import datetime
import math

import numpy as np
import pandas as pd
import torch

from algorithms.sgnn.utils import Data, prepare_data   # the original numpy utilities, logic unchanged
from algorithms.stamp.stamp import TFAdam


def glorot_uniform(fan_in, fan_out):
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return (torch.rand(fan_in, fan_out) * 2 - 1) * limit


class SRGNNModel(torch.nn.Module):
    def __init__(self, n_nodes, hidden_size, out_size, step, nonhybrid):
        super().__init__()
        self.stdv = 1.0 / math.sqrt(hidden_size)
        U = lambda *shape: torch.nn.Parameter((torch.rand(*shape) * 2 - 1) * self.stdv)
        P = torch.nn.Parameter
        d = out_size
        self.nasr_w1 = U(d, d); self.nasr_w2 = U(d, d); self.nasr_v = U(1, d)
        self.nasr_b = P(torch.zeros(d))
        self.embedding = U(n_nodes, hidden_size)
        self.W_in = U(d, d); self.b_in = U(d); self.W_out = U(d, d); self.b_out = U(d)
        # tf.nn.rnn_cell.GRUCell(out_size), input depth 2d, state depth d
        self.gates_kernel = P(glorot_uniform(3 * d, 2 * d)); self.gates_bias = P(torch.ones(2 * d))
        self.cand_kernel = P(glorot_uniform(3 * d, d)); self.cand_bias = P(torch.zeros(d))
        if not nonhybrid:
            self.B = U(2 * d, d)
        self.d, self.step, self.nonhybrid = d, step, nonhybrid

    def gru_cell(self, x, h):
        value = torch.sigmoid(torch.cat([x, h], 1) @ self.gates_kernel + self.gates_bias)
        r, u = value[:, :self.d], value[:, self.d:]
        c = torch.tanh(torch.cat([x, r * h], 1) @ self.cand_kernel + self.cand_bias)
        return u * h + (1 - u) * c

    def ggnn(self, item, adj_in, adj_out):
        B = item.shape[0]; d = self.d
        fin_state = self.embedding[item]
        for _ in range(self.step):
            fin_state = fin_state.reshape(B, -1, d)
            fin_in = (fin_state.reshape(-1, d) @ self.W_in + self.b_in).reshape(B, -1, d)
            fin_out = (fin_state.reshape(-1, d) @ self.W_out + self.b_out).reshape(B, -1, d)
            av = torch.cat([adj_in @ fin_in, adj_out @ fin_out], dim=-1)
            fin_state = self.gru_cell(av.reshape(-1, 2 * d), fin_state.reshape(-1, d))
        return fin_state.reshape(B, -1, d)

    def forward(self, item, adj_in, adj_out, alias, mask):
        re_embedding = self.ggnn(item, adj_in, adj_out)
        B = item.shape[0]; d = self.d
        rm = mask.sum(1)
        ar = torch.arange(B, device=item.device)
        last_id = alias[ar, rm.long() - 1]
        last_h = re_embedding[ar, last_id]
        seq_h = re_embedding[ar[:, None], alias]
        last = (last_h @ self.nasr_w1).reshape(B, 1, -1)
        seq = (seq_h.reshape(-1, d) @ self.nasr_w2).reshape(B, -1, d)
        m = torch.sigmoid(last + seq + self.nasr_b)
        coef = (m.reshape(-1, d) @ self.nasr_v.T) * mask.reshape(-1, 1)
        b = self.embedding[1:]
        if not self.nonhybrid:
            ma = torch.cat([(coef.reshape(B, -1, 1) * seq_h).sum(1), last.reshape(-1, d)], -1)
            return (ma @ self.B) @ b.T
        ma = (coef.reshape(B, -1, 1) * seq_h).sum(1)
        return ma @ b.T


class GGNN:
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, max_sess_length=100, lr=0.001, l2=0.00001,
                 step=1, lr_dc=0.1, lr_dc_step=3, nonhybrid=True, epoch_n=10, batch_predict=False, seed=None, device=None):
        self.hidden_size, self.out_size, self.batch_size = hidden_size, out_size, batch_size
        self.nonhybrid = nonhybrid
        self.L2, self.step, self.lr_dc_step, self.lr_dc, self.lr = l2, step, lr_dc_step, lr_dc, lr
        self.epoch_n, self.max_sess_length = epoch_n, max_sess_length
        self.session, self.session_items = -1, []
        self.batch_predict, self.test_idx = batch_predict, 0
        self.seed = seed
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def _tensors(self, adj_in, adj_out, alias, item, mask):
        dev = self.device
        return (torch.tensor(np.asarray(item), dtype=torch.long, device=dev),
                torch.tensor(np.asarray(adj_in), dtype=torch.float32, device=dev),
                torch.tensor(np.asarray(adj_out), dtype=torch.float32, device=dev),
                torch.tensor(np.asarray(alias), dtype=torch.long, device=dev),
                torch.tensor(np.asarray(mask), dtype=torch.float32, device=dev))

    def fit(self, train, test=None, sample_store=10000000):
        if self.seed is not None:
            np.random.seed(self.seed); torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        self.n_nodes = len(train.ItemId.unique()) + 1
        train_data, test_data, self.item_dict, self.reversed_item_dict = prepare_data(train, test)
        self.train_data = Data(train_data, sub_graph=True, method='ggnn', shuffle=True)
        self.test_data = Data(test_data, sub_graph=True, method='ggnn', shuffle=False)
        self.decay = self.lr_dc_step * len(self.train_data.inputs) / self.batch_size
        self.model = SRGNNModel(self.n_nodes, self.hidden_size, self.out_size, self.step, self.nonhybrid).to(self.device)
        opt = TFAdam(self.model.parameters(), lr=self.lr)
        global_step = 0
        for epoch in range(self.epoch_n):
            self.model.train()
            slices = self.train_data.generate_batch(self.batch_size)
            print('start training: ', datetime.datetime.now())
            for i in slices:
                adj_in, adj_out, alias, item, mask, targets = self.train_data.get_slice(i)
                item_t, ain, aout, alias_t, mask_t = self._tensors(adj_in, adj_out, alias, item, mask)
                logits = self.model(item_t, ain, aout, alias_t, mask_t)
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor(np.asarray(targets) - 1, dtype=torch.long, device=self.device))
                loss = loss + self.L2 * sum((p ** 2).sum() / 2 for p in self.model.parameters())
                for g in opt.param_groups:   # tf.train.exponential_decay(lr, global_step, decay, lr_dc, staircase=True)
                    g['lr'] = self.lr * self.lr_dc ** math.floor(global_step / self.decay)
                opt.zero_grad(); loss.backward(); opt.step(); global_step += 1
            print('train_loss:\t%.4f\tEpoch:\t%d' % (loss.item(), epoch))
        self.model.eval()
        if self.batch_predict:
            # the original computes these scores in its last epoch's test pass (shuffle=False, no dropout)
            all_scores = []
            with torch.no_grad():
                for i in self.test_data.generate_batch(self.batch_size):
                    adj_in, adj_out, alias, item, mask, targets = self.test_data.get_slice(i)
                    all_scores.append(self.model(*self._tensors(adj_in, adj_out, alias, item, mask)).cpu().numpy())
            self.all_scores = np.concatenate(all_scores)
            self.predicted_item_ids = [int(self.reversed_item_dict[idx + 1]) for idx in range(self.all_scores.shape[1])]

    @torch.no_grad()
    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, mode_type='view', timestamp=0):
        if self.session != session_id:
            self.session = session_id; self.session_items = list()
        self.session_items.append(self.item_dict[input_item_id])
        if self.batch_predict:
            scores = self.all_scores[self.test_idx]; self.test_idx += 1
        else:
            adj_in, adj_out, alias, item, mask, targets = self.test_data.get_slice_by_session_items(self.session_items, self.test_data.len_max)
            scores = self.model(*self._tensors(adj_in, adj_out, alias, item, mask))[0].cpu().numpy()
            self.predicted_item_ids = [int(self.reversed_item_dict[idx + 1]) for idx in range(len(scores))]
        return pd.Series(data=scores, index=self.predicted_item_ids)

    def clear(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def support_users(self):
        return False
