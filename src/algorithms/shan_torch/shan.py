"""PyTorch port of algorithms/shan/shan.py (SHAN, TensorFlow 1), kept as close to the original as possible.

Ying et al., Sequential Recommender System based on Hierarchical Attention Network, IJCAI 2018; session-rec's version.
Reproduced exactly (they change results), including the original's quirks:
  * training records: per user, every session after the first; the held-out item is random.choice(session) and is
    removed; 10 negatives via np.random.randint not in the user's items so far
  * train_pre_sessions gets the first session of EVERY user (also users without a later session), so it drifts out
    of alignment with the training records exactly as in the original
  * embeddings N(0, 0.01), attention weights / biases U(+-sqrt(3/d)); two-level attention with relu + softmax
  * loss mean(-log sigmoid(pos - neg)) + lambda_uv * l2(user, item, neg rows) + lambda_a * l2(W1, W2), l2 = sum(x^2)/2
  * plain SGD lr 0.1, batch size 1, records in fixed order, `iter` epochs
  * predict_next: current session = test session[:position] (empty at the first event, as the original),
    long-term part = all training items of the user
Extra parameters: seed (numpy, python random, torch), device (default cpu: batch size 1 is faster there).
"""
import copy
import datetime
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


class _DataGeneration:
    def __init__(self, train_data, test_data, neg_number, data_key):
        self.item_key, self.time_key = data_key['item_key'], data_key['time_key']
        self.user_key, self.session_key = data_key['user_key'], data_key['session_key']
        self.train_users, self.train_sessions, self.train_items, self.train_neg_items, self.train_pre_sessions = [], [], [], [], []
        self.neg_number = neg_number
        self.train_batch_id = 0
        self.records_number = 0
        self.load_data(train_data, test_data)

    def load_data(self, train_data, test_data):
        self.user_purchased_item = dict()
        self.itemids = train_data[self.item_key].unique()
        assert len(np.setdiff1d(test_data[self.item_key].unique(), self.itemids, assume_unique=True)) == 0
        self.userids = np.union1d(train_data[self.user_key].unique(), test_data[self.user_key].unique())
        self.sessionids = np.union1d(train_data[self.session_key].unique(), test_data[self.session_key].unique())
        self.user_number, self.item_number = len(self.userids), len(self.itemids)
        self.item2id = dict(zip(self.itemids, range(0, len(self.itemids))))
        self.user2id = dict(zip(self.userids, range(0, len(self.userids))))
        self.session2id = dict(zip(self.sessionids, range(0, len(self.sessionids))))
        self.id2item = {v: k for k, v in self.item2id.items()}
        for user_id in train_data.UserId.unique():
            uid = self.user2id[user_id]
            user_pd = train_data.loc[train_data['UserId'] == user_id]
            mapped = [self.item2id[iid] for iid in user_pd['ItemId']]
            sessions = []
            for sess in user_pd.SessionId.unique():
                mask = (user_pd['SessionId'] == sess).values
                sessions.append([m for m, keep in zip(mapped, mask) if keep])
            size = len(sessions)
            the_first_session = sessions[0]
            self.train_pre_sessions.append(the_first_session)
            self.user_purchased_item[uid] = copy.deepcopy(the_first_session)
            for j in range(1, size):
                self.train_users.append(uid)
                current_session = sessions[j]
                self.train_neg_items.append(self.gen_neg(uid))
                if j != 1:
                    self.train_pre_sessions.append(copy.deepcopy(self.user_purchased_item[uid]))
                self.user_purchased_item[uid].extend(copy.deepcopy(current_session))
                item = random.choice(current_session)
                self.train_items.append(item)
                current_session.remove(item)
                self.train_sessions.append(current_session)
                self.records_number += 1
        self.test_candidate_items = list(range(self.item_number))
        self.test_sid_to_data = dict()
        for sess_id in test_data.SessionId.unique():
            session_pd = test_data.loc[test_data['SessionId'] == sess_id]
            self.test_sid_to_data[self.session2id[sess_id]] = [self.item2id[int(row)] for row in session_pd['ItemId'].values]

    def gen_neg(self, user_id):
        neg_item_set = list()
        while len(neg_item_set) < self.neg_number:
            neg_item = np.random.randint(self.item_number)
            if neg_item not in self.user_purchased_item[user_id]:
                neg_item_set.append(neg_item)
        return neg_item_set

    def gen_train_batch_data(self, batch_size):
        if self.train_batch_id == self.records_number:
            self.train_batch_id = 0
        i = self.train_batch_id
        out = (self.train_users[i:i + batch_size], self.train_items[i:i + batch_size], self.train_sessions[i],
               self.train_neg_items[i], self.train_pre_sessions[i])
        self.train_batch_id = i + batch_size
        return out


class SHANModel(torch.nn.Module):
    def __init__(self, user_number, item_number, d):
        super().__init__()
        lim = np.sqrt(3 / d)
        U = lambda *s: torch.nn.Parameter((torch.rand(*s) * 2 - 1) * lim)
        self.user_embedding_matrix = torch.nn.Parameter(torch.randn(user_number, d) * 0.01)
        self.item_embedding_matrix = torch.nn.Parameter(torch.randn(item_number, d) * 0.01)
        self.the_first_w = U(d, d); self.the_second_w = U(d, d)
        self.the_first_bias = U(d); self.the_second_bias = U(d)

    def hybrid(self, user_id, current_session, pre_sessions):
        # sparse row lookups: plain SGD then updates only the touched rows (as TF1's IndexedSlices did), which is
        # exactly the dense update but avoids rewriting the full user and item matrices at every batch-size-1 step
        u = F.embedding(user_id, self.user_embedding_matrix, sparse=True)  # (1, d)
        pre = F.embedding(pre_sessions, self.item_embedding_matrix, sparse=True)
        w1 = torch.softmax((torch.relu(pre @ self.the_first_w + self.the_first_bias) @ u.T).T, dim=1)
        long_user = (pre * w1.T).sum(0)
        cat = torch.cat([F.embedding(current_session, self.item_embedding_matrix, sparse=True), long_user[None]], 0)
        w2 = torch.softmax((torch.relu(cat @ self.the_second_w + self.the_second_bias) @ u.T).T, dim=1)
        return (cat * w2.T).sum(0), u


class SHAN:
    def __init__(self, iter=100, global_dimension=50, neg_number=10, lambda_a=0.01, lambda_uv=0.01,
                 session_key='SessionId', item_key='ItemId', user_key='UserId', time_key='Time', seed=None, device='cpu'):
        self.lambda_u_v, self.lambda_a, self.iter = lambda_uv, lambda_a, iter
        self.global_dimensions, self.neg_number = global_dimension, neg_number
        self.session_key, self.item_key, self.time_key, self.user_key = session_key, item_key, time_key, user_key
        self.seed = seed
        self.device = torch.device(device)
        self.current_session_test = -1

    def fit(self, train_data, test_data=None):
        if self.seed is not None:
            np.random.seed(self.seed); random.seed(self.seed); torch.manual_seed(self.seed)
        data_key = dict(item_key=self.item_key, time_key=self.time_key, user_key=self.user_key, session_key=self.session_key)
        self.dg = _DataGeneration(train_data, test_data, self.neg_number, data_key)
        self.model = SHANModel(self.dg.user_number, self.dg.item_number, self.global_dimensions).to(self.device)
        opt = torch.optim.SGD(self.model.parameters(), lr=0.1)
        l2 = lambda t: (t ** 2).sum() / 2
        dev = self.device
        # pre-build tensors once (the record order never changes)
        recs = []
        for _ in range(self.dg.records_number):
            bu, bi, bs, bn, bp = self.dg.gen_train_batch_data(1)
            recs.append(tuple(torch.tensor(x, dtype=torch.long, device=dev) for x in (bu, bi, bs, bn, bp)))
        self.dg.train_batch_id = 0
        for it in range(self.iter):
            print(datetime.datetime.now(), 'iteration:', it)
            all_loss = 0.0
            for bu, bi, bs, bn, bp in recs:
                h, u = self.model.hybrid(bu, bs, bp)
                item_e = F.embedding(bi, self.model.item_embedding_matrix, sparse=True)
                neg_e = F.embedding(bn, self.model.item_embedding_matrix, sparse=True)
                pos = h[None] @ item_e.T; neg = h[None] @ neg_e.T
                loss = torch.mean(-torch.log(torch.sigmoid(pos - neg)))
                loss = loss + (self.lambda_a * l2(self.model.the_first_w) + self.lambda_a * l2(self.model.the_second_w)) + \
                    (self.lambda_u_v * l2(u) + self.lambda_u_v * l2(item_e) + self.lambda_u_v * l2(neg_e))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                all_loss += loss.item()
            print('loss = ' + str(all_loss))
        self.model.eval()
        self.all_items = torch.arange(self.dg.item_number, device=dev)
        self.index_all = [self.dg.id2item[x] for x in range(self.dg.item_number)]

    @torch.no_grad()
    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids=None, timestamp=None):
        dev = self.device
        if self.current_session_test != session_id:
            self.current_session_test = session_id
            self.batch_session = self.dg.test_sid_to_data[self.dg.session2id[session_id]]
            self.batch_pre_session = self.dg.user_purchased_item[self.dg.user2id[input_user_id]]
            self.current_pos = 0
        else:
            self.current_pos += 1
        h, _ = self.model.hybrid(torch.tensor([self.dg.user2id[input_user_id]], device=dev),
                                 torch.tensor(self.batch_session[:self.current_pos], dtype=torch.long, device=dev),
                                 torch.tensor(self.batch_pre_session, dtype=torch.long, device=dev))
        scores = (h[None] @ self.model.item_embedding_matrix.T)[0].cpu().numpy()
        return pd.Series(data=scores, index=self.index_all)

    def support_users(self):
        return True

    def predict_with_training_data(self):
        return False

    def clear(self):
        self.model = None
