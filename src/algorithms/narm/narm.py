"""PyTorch port of session-rec's NARM (Theano), kept as close to the original as possible.

Li et al., Neural Attentive Session-based Recommendation, CIKM 2017; session-rec's Theano version.
Faithful details (they change results, so they are reproduced exactly):
  * Theano hard_sigmoid  = clip(0.2 x + 0.5, 0, 1)  (not torch.nn.functional.hardsigmoid)
  * custom GRU (hard-sigmoid gates, mask keeps the previous state on padded steps), orthogonal recurrent init
  * non-inverted dropout: training multiplies by a Bernoulli(keep) mask, inference multiplies by keep
  * attention softmax over the masked similarity, renormalised over the real steps
  * bilinear decoder tied to the item embedding: softmax(proj . (Wemb . bili)^T), padding row 0 included
  * training: all session prefixes, first 19 items kept, 10% random validation split, Adam, keep the parameters
    with the best validation MRR@20, early stopping as in the original
  * predict_next: re-encodes the whole session so far, drops the padding row
Extra parameters: seed (numpy + torch), device ('cuda:0' / 'cpu'; default cuda if available).
"""
import sys
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from evaluation import tracking


def hard_sigmoid(x):
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


class NARMModel(torch.nn.Module):
    def __init__(self, n_items, dim_proj, hidden_units):
        super().__init__()
        H = hidden_units

        def init_weights(shape):  # original: randn * sqrt(2 / shape[0]), numpy global RNG
            return torch.tensor((np.random.randn(*shape) * np.sqrt(2. / shape[0])).astype('float32'))

        def ortho_weight(ndim):
            u, s, v = np.linalg.svd(np.random.randn(ndim, ndim))
            return torch.tensor(u.astype('float32'))

        P = torch.nn.Parameter
        # same creation order as the original init_params
        self.Wemb = P(init_weights((n_items, dim_proj)))
        self.gru_Wxrz = P(torch.cat([init_weights((dim_proj, H)), init_weights((dim_proj, H)), init_weights((dim_proj, H))], 1))
        self.gru_Urz = P(torch.cat([ortho_weight(H), ortho_weight(H)], 1))
        self.gru_Uh = P(ortho_weight(H))
        self.gru_b = P(torch.zeros(3 * H))
        self.W_encoder = P(init_weights((H, H)))
        self.W_decoder = P(init_weights((H, H)))
        self.bl_vector = P(init_weights((1, H)))
        self.bili = P(init_weights((dim_proj, 2 * H)))
        self.H = H

    @staticmethod
    def dropout(x, p, training):
        keep = 1. - p
        if training:
            return x * torch.bernoulli(torch.full_like(x, keep))
        return x * keep

    def forward(self, x, mask, training=False):
        """x, mask: (T, n). Returns softmax probabilities (n, n_items)."""
        H = self.H
        T_, n = x.shape
        emb = self.Wemb[x.reshape(-1)].reshape(T_, n, -1)
        emb = self.dropout(emb, 0.25, training)
        state_below = emb @ self.gru_Wxrz + self.gru_b
        h = torch.zeros(n, H, device=x.device)
        hs = []
        for t in range(T_):
            x_ = state_below[t]
            preact = h @ self.gru_Urz + x_[:, 0:2 * H]
            z = hard_sigmoid(preact[:, 0:H])
            r = hard_sigmoid(preact[:, H:2 * H])
            hh = torch.tanh((h * r) @ self.gru_Uh + x_[:, 2 * H:3 * H])
            hh = (1.0 - z) * h + z * hh
            m_ = mask[t][:, None]
            h = m_ * hh + (1. - m_) * h
            hs.append(h)
        proj = torch.stack(hs)
        last_h = proj[-1]
        tmp = hard_sigmoid(proj @ self.W_encoder.T + (last_h @ self.W_decoder.T)[None])
        sim_matrix = (tmp @ self.bl_vector.T)[..., 0]
        att = torch.softmax(sim_matrix.T * mask.T, dim=1) * mask.T
        weight = att / att.sum(dim=1)[:, None]
        attention_proj = (proj * weight.T[:, :, None]).sum(dim=0)
        out = torch.cat([attention_proj, last_h], dim=1)
        out = self.dropout(out, 0.5, training)
        ytem = self.Wemb @ self.bili
        return torch.softmax(out @ ytem.T, dim=1)


class NARM:
    def __init__(self, factors=100, hidden_units=100, epochs=30, lr=0.001, session_key='SessionId', item_key='ItemId',
                 seed=None, device=None):
        self.factors = factors
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.lr = lr
        self.session_key = session_key
        self.item_key = item_key
        self.session = -1
        self.session_items = list()
        self.seed = seed
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def fit(self, data, test=None):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        nis = data[self.item_key].nunique()
        self.itemmap = pd.Series(index=data[self.item_key].unique(), data=range(1, nis + 1))
        data = data.merge(self.itemmap.to_frame('ItemIdx'), how='inner', right_index=True, left_on=self.item_key)
        data.sort_values(['SessionId', 'Time'], inplace=True)
        self.traindata = self.create_training_data(data)
        self.train_gru(self.factors, self.hidden_units, max_epochs=self.epochs, lrate=self.lr, n_items=nis + 1)

    def train_gru(self, dim_proj=50, hidden_units=100, patience=5, max_epochs=30, lrate=0.001, n_items=37484,
                  batch_size=512, valid_batch_size=512):
        train, valid = self.load_data()                      # (shuffles first, as the original)
        self.model = NARMModel(n_items, dim_proj, hidden_units).to(self.device)
        tracking.watch(self.model)
        opt = torch.optim.Adam(self.model.parameters(), lr=lrate, betas=(0.9, 0.999), eps=1e-8)
        print('%d train examples' % len(train[0])); print('%d valid examples' % len(valid[0]))
        history_vali, best_p, bad_count, estop = [], None, 0, False
        for eidx in range(max_epochs):
            start_time = time.time(); epoch_loss = []
            kf = self.get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
            kf_valid = self.get_minibatches_idx(len(valid[0]), valid_batch_size, shuffle=True)
            for _, train_index in kf:
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]
                x, mask, y = self.prepare_data(x, y)
                pred = self.model(x, mask, training=True)
                cost = -torch.log(pred[torch.arange(pred.shape[0], device=self.device), y] + 1e-8).mean()
                opt.zero_grad(); cost.backward(); opt.step()
                loss = cost.item(); epoch_loss.append(loss)
                if np.isnan(loss) or np.isinf(loss):
                    print('bad loss detected: ', loss); return 1., 1., 1.
            valid_evaluation = self.pred_evaluation(valid, kf_valid)
            if best_p is None or valid_evaluation[1] >= np.array(history_vali).max():
                best_p = OrderedDict((k, v.detach().clone()) for k, v in self.model.state_dict().items())
                print('Best perfomance updated!'); bad_count = 0
            print('Epoch', eidx, 'loss', np.mean(epoch_loss), 'Valid Recall@20:', valid_evaluation[0], '   Valid Mrr@20:', valid_evaluation[1])
            tracking.log({'train/epoch': eidx + 1, 'train/loss': float(np.mean(epoch_loss)),
                          'train/valid_recall@20': float(valid_evaluation[0]), 'train/valid_mrr@20': float(valid_evaluation[1])})
            if len(history_vali) > 10 and valid_evaluation[1] <= np.array(history_vali).max():
                bad_count += 1
                if bad_count > patience:
                    print('Early Stop!'); estop = True
            history_vali.append(valid_evaluation[1])
            print('This epoch took %.1fs' % (time.time() - start_time), file=sys.stderr)
            if estop:
                break
        self.model.load_state_dict(best_p)
        self.model.eval()

    def create_training_data(self, data):
        index_session = data.columns.get_loc(self.session_key)
        index_item = data.columns.get_loc('ItemIdx')
        out_seqs, labs, session, session_items = [], [], -1, []
        for row in data.itertuples(index=False):
            if row[index_session] != session:
                session = row[index_session]; session_items = list()
            session_items.append(row[index_item])
            if len(session_items) > 1:
                out_seqs += [session_items[:-1]]; labs += [session_items[-1]]
        return out_seqs, labs

    def prepare_data(self, seqs, labels):
        lengths = [len(s) for s in seqs]
        n_samples = len(seqs); maxlen = np.max(lengths)
        x = np.zeros((maxlen, n_samples)).astype('int64')
        for idx, s in enumerate(seqs):
            x[:lengths[idx], idx] = s
        x_mask = (x != 0).astype('float32')
        return (torch.tensor(x, device=self.device), torch.tensor(x_mask, device=self.device),
                torch.tensor(np.asarray(labels, dtype='int64'), device=self.device))

    def load_data(self, valid_portion=0.1, maxlen=19):
        train_set_x, train_set_y = [], []
        for x, y in zip(*self.traindata):
            train_set_x.append(x if len(x) < maxlen else x[:maxlen]); train_set_y.append(y)
        n_samples = len(train_set_x)
        sidx = np.arange(n_samples, dtype='int32'); np.random.shuffle(sidx)
        n_train = int(np.round(n_samples * (1. - valid_portion)))
        valid = ([train_set_x[s] for s in sidx[n_train:]], [train_set_y[s] for s in sidx[n_train:]])
        train = ([train_set_x[s] for s in sidx[:n_train]], [train_set_y[s] for s in sidx[:n_train]])
        return train, valid

    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
        idx_list = np.arange(n, dtype='int32')
        if shuffle:
            np.random.shuffle(idx_list)
        minibatches, minibatch_start = [], 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:minibatch_start + minibatch_size]); minibatch_start += minibatch_size
        if minibatch_start != n:
            minibatches.append(idx_list[minibatch_start:])
        return zip(range(len(minibatches)), minibatches)

    @torch.no_grad()
    def pred_evaluation(self, data, iterator):
        recall, mrr, count = 0.0, 0.0, 0
        for _, valid_index in iterator:
            x, mask, y = self.prepare_data([data[0][t] for t in valid_index], np.array(data[1])[valid_index])
            preds = self.model(x, mask, training=False)
            target = preds[torch.arange(preds.shape[0], device=self.device), y]
            ranks = (preds > target[:, None]).sum(dim=1) + 1
            ok = ranks <= 20
            recall += ok.sum().item(); mrr += (1.0 / ranks[ok].float()).sum().item(); count += len(ranks)
        return recall / count, mrr / count

    @torch.no_grad()
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, timestamp=0, skip=False, mode_type='view'):
        if self.session != session_id:
            self.session = session_id; self.session_items = list()
        if mode_type == 'view':
            self.session_items.append(input_item_id)
        if skip:
            return
        x = [self.itemmap[self.session_items].values]
        x, mask, _ = self.prepare_data(x, [0])
        preds = self.model(x, mask, training=False)
        return pd.Series(data=preds[0][1:].cpu().numpy(), index=self.itemmap.index)

    @torch.no_grad()
    def export(self):
        '''Item embeddings (Wemb, also used for scoring): row i belongs to item_ids[i] (padding row dropped).'''
        rows = torch.as_tensor(self.itemmap.values.astype('int64'), device=self.device)
        return {'item_ids': self.itemmap.index.values, 'item_embeddings': self.model.Wemb[rows].detach().float().cpu()}

    def clear(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def support_users(self):
        return False
