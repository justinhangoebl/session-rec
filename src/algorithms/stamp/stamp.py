"""PyTorch port of session-rec's STAMP (TensorFlow 1), kept as close to the original as possible.

Liu et al., STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation, KDD 2018;
session-rec's TF1 version. Reproduced exactly (they change results):
  * item index built over train AND test sessions (test-only items stay randomly initialised but are scored)
  * embedding N(0, emb_stddev), padding row 0 masked to zero; lookups clipped to L2 norm <= 1 *on the fly*
    (tf.nn.embedding_lookup(max_norm=1); torch's max_norm would rewrite the weights instead)
  * attention: act(asp W_a + ctx W_c + mean W_o) . w_ca with the 'none' normaliser (no softmax), attention output
    plus the mean pool, then tanh(. W1) * tanh(last W2), scores against the masked embedding rows 1..N
  * loss = SUM of the per-example softmax cross-entropies (TF differentiates the loss vector)
  * TF1 Adam (lr_t = lr sqrt(1-b2^t)/(1-b1^t), eps outside the sqrt of the raw v), per-variable clip_by_norm(110);
    learning-rate decay only if update_lr=True (default False, as in the original)
  * training batches: sessions bucketed by length, random bucket order, shuffled inside the bucket, chunks of
    batch_size, one optimisation step per prefix position
  * predict_next replays the stored test sample of the session at step s (the original ignores input_item_id)
Extra parameters: seed (numpy, python random, torch), device.
"""
import random
import time

import numpy as np
import pandas as pd
import torch
from evaluation import tracking


class _Sample:
    __slots__ = ('id', 'session_id', 'click_items', 'items_idxes', 'in_idxes', 'out_idxes')


def _load_data(data, item2idx, idx_cnt, pad_idx, session_key, item_key, time_key):
    data.sort_values([session_key, time_key], inplace=True)       # in place, as the original
    samples, now_id, last_id, click_items = [], 0, None, []

    def close(session_id, items, idx_cnt, now_id):
        idxes = []
        for item in items:
            if item not in item2idx:
                if idx_cnt == pad_idx:
                    idx_cnt += 1
                item2idx[item] = idx_cnt
                idx_cnt += 1
            idxes.append(item2idx[item])
        s = _Sample(); s.id = now_id; s.session_id = session_id; s.click_items = items
        s.items_idxes = idxes; s.in_idxes = idxes[:-1]; s.out_idxes = idxes[1:]
        return s, idx_cnt

    for s_id, item_id in zip(list(data[session_key].values), list(data[item_key].values)):
        if last_id is None:
            last_id = s_id
        if s_id != last_id:
            s, idx_cnt = close(last_id, click_items, idx_cnt, now_id); samples.append(s)
            last_id = s_id; click_items = []; now_id += 1
        else:
            last_id = s_id
        click_items.append(item_id)
    s, idx_cnt = close(last_id, click_items, idx_cnt, now_id); samples.append(s)
    return samples, idx_cnt


def _batches(samples, shuffle):
    """Equal-length batcher (util/batcher/equal_len/batcher_p.py)."""
    len_dic = {}
    for sample in samples:
        len_dic.setdefault(len(sample.click_items), []).append(sample)
    key_list = list(len_dic.keys())
    order = np.random.permutation(len(key_list)) if shuffle else range(len(key_list))
    for k in order:
        samplelist = len_dic[key_list[k]]
        random.shuffle(samplelist)
        yield [s.id for s in samplelist], [list(s.in_idxes) for s in samplelist], [list(s.out_idxes) for s in samplelist]


class TFAdam(torch.optim.Optimizer):
    """tf.compat.v1.train.AdamOptimizer with per-variable tf.clip_by_norm applied to each gradient first."""

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, max_grad_norm=None):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, max_grad_norm=max_grad_norm))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if group['max_grad_norm'] is not None:
                    norm = g.norm()
                    if norm > group['max_grad_norm']:
                        g = g * (group['max_grad_norm'] / norm)
                st = self.state[p]
                if not st:
                    st['t'] = 0; st['m'] = torch.zeros_like(p); st['v'] = torch.zeros_like(p)
                st['t'] += 1; t = st['t']
                st['m'].mul_(b1).add_(g, alpha=1 - b1)
                st['v'].mul_(b2).addcmul_(g, g, value=1 - b2)
                lr_t = group['lr'] * (1 - b2 ** t) ** 0.5 / (1 - b1 ** t)
                p.addcdiv_(st['m'], st['v'].sqrt().add_(group['eps']), value=-lr_t)


class STAMPModel(torch.nn.Module):
    def __init__(self, pre_embedding, edim, stddev, active):
        super().__init__()
        P = torch.nn.Parameter
        normal = lambda *shape: torch.randn(*shape) * stddev          # tf.random_normal(stddev)
        self.embe_dict = P(torch.tensor(pre_embedding, dtype=torch.float32))
        mask = torch.ones_like(self.embe_dict); mask[0] = 0
        self.register_buffer('pe_mask', mask)
        self.w_asp = P(normal(edim, edim)); self.w_ctx = P(normal(edim, edim)); self.w_out = P(normal(edim, edim))
        self.w_ca = P(normal(edim, 1))
        self.w1 = P(normal(edim, edim)); self.w2 = P(normal(edim, edim))
        self.act = {'tanh': torch.tanh, 'relu': torch.relu, 'sigmoid': torch.sigmoid}.get(active, torch.tanh)

    @staticmethod
    def lookup(table, idx, max_norm=1.0):
        e = table[idx]
        norm = e.norm(dim=-1, keepdim=True)
        return e * (max_norm / torch.clamp(norm, min=max_norm))      # tf clip_by_norm per looked-up row

    def forward(self, inputs, last_inputs, seq_len):
        E = self.embe_dict * self.pe_mask
        ctx = self.lookup(E, inputs)
        last = self.lookup(E, last_inputs)
        pool_out = ctx.sum(dim=1) / seq_len[:, None]
        res = self.act(last[:, None, :] @ self.w_asp + ctx @ self.w_ctx + pool_out[:, None, :] @ self.w_out)
        alpha = (res @ self.w_ca)[..., 0]                              # 'none' normaliser (bitmap of ones)
        attout = (alpha[:, None, :] @ ctx)[:, 0, :] + pool_out
        attout = torch.tanh(attout @ self.w1)
        lastv = torch.tanh(last @ self.w2)
        return (attout * lastv) @ E[1:].T


class Seq2SeqAttNN:
    def __init__(self, model="stamp", dataset="rsc15_64", recsys_threshold_acc=0.68, cikm_threshold_acc=0.62,
                 n_epochs=10, hidden_size=100, active="sigmoid", decay_steps=1, decay_rate=0.1, batch_size=512,
                 init_lr=0.003, stddev=0.05, emb_stddev=0.002, edim=100, max_grad_norm=110, pad_idx=0, emb_up=True,
                 update_lr=False, model_save_path=None, is_print=True, cell="gru", cut_off=20, reload=True,
                 class_num=3, is_train=False, is_save=False, model_path=None, session_key='SessionId',
                 item_key='ItemId', time_key='Time', seed=None, device=None):
        self.nepoch = n_epochs; self.hidden_size = hidden_size; self.active = active
        self.decay_steps = decay_steps; self.decay_rate = decay_rate; self.update_lr = update_lr
        self.batch_size = batch_size; self.init_lr = init_lr; self.stddev = stddev; self.emb_stddev = emb_stddev
        self.edim = edim; self.max_grad_norm = max_grad_norm; self.pad_idx = pad_idx; self.emb_up = emb_up
        self.session_key = session_key; self.item_key = item_key; self.time_key = time_key
        self.s = 0; self.old_session_id = 0
        self.seed = seed
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def fit(self, train, test):
        if self.seed is not None:
            np.random.seed(self.seed); random.seed(self.seed); torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        item2idx = {'<pad>': self.pad_idx}
        self.train_samples, idx_cnt = _load_data(train, item2idx, 0, self.pad_idx, self.session_key, self.item_key, self.time_key)
        self.test_samples, idx_cnt = _load_data(test, item2idx, idx_cnt, self.pad_idx, self.session_key, self.item_key, self.time_key)
        self.n_items = len(item2idx) - 1
        pre_embedding = np.random.normal(0, self.emb_stddev, [len(item2idx), self.hidden_size])
        pre_embedding[self.pad_idx] = [0.0] * self.hidden_size
        mapping = dict(item2idx); mapping.pop('<pad>')
        self.item_index = list(mapping.keys())
        self.test_by_session = {}
        for s in self.test_samples:
            self.test_by_session.setdefault(s.session_id, s)
        self.model = STAMPModel(pre_embedding, self.edim, self.stddev, self.active).to(self.device)
        tracking.watch(self.model)
        if not self.emb_up:
            self.model.embe_dict.requires_grad_(False)
        self.opt = TFAdam([p for p in self.model.parameters() if p.requires_grad], lr=self.init_lr,
                          max_grad_norm=self.max_grad_norm)
        self.global_step = 0
        self.train()
        self.model.eval()

    def _lr(self):
        if not self.update_lr:
            return self.init_lr
        # tf.train.exponential_decay(init_lr, global_step / 28, decay_steps, decay_rate, staircase=True)
        return self.init_lr * self.decay_rate ** np.floor((self.global_step / 28) / self.decay_steps)

    def _step(self, batch_in, batch_last, batch_out, seq_l):
        dev = self.device
        logits = self.model(torch.tensor(batch_in, device=dev), torch.tensor(batch_last, device=dev),
                            torch.tensor(seq_l, dtype=torch.float32, device=dev))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor(batch_out, device=dev), reduction='none')
        for g in self.opt.param_groups:
            g['lr'] = self._lr()
        self.opt.zero_grad(); loss.sum().backward(); self.opt.step()
        self.global_step += 1
        return loss.detach().cpu().numpy()

    def train(self):
        for epoch in range(self.nepoch):
            c = []; start_time = time.time()
            for _, in_data, out_data in _batches(self.train_samples, shuffle=True):
                n = len(in_data)
                chunks = [(i, i + self.batch_size) for i in range(0, n - n % self.batch_size, self.batch_size)] if n > self.batch_size else [(0, n)]
                if n > self.batch_size and n % self.batch_size > 0:
                    chunks.append((n - n % self.batch_size, n))
                for a, b in chunks:
                    tmp_in, tmp_out = in_data[a:b], out_data[a:b]
                    for s in range(len(tmp_in[0])):
                        c += list(self._step([x[:s + 1] for x in tmp_in], [x[s] for x in tmp_in],
                                             [y[s] - 1 for y in tmp_out], [s + 1] * len(tmp_in)))
            avgc = np.mean(c)
            if np.isnan(avgc):
                print('Epoch {}: NaN error!'.format(epoch)); self.error_during_train = True; return
            print('Epoch{}\tloss: {:.6f}\ttime: {}s'.format(epoch, avgc, time.time() - start_time))
            tracking.log({'train/epoch': epoch + 1, 'train/loss': float(avgc)})

    @torch.no_grad()
    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, mode_type='view', timestamp=0):
        sample = self.test_by_session.get(session_id)
        if self.old_session_id != session_id:
            self.s = 0
        if skip:
            self.s = self.s + 1; self.old_session_id = session_id
            return
        if sample is None or len(sample.in_idxes) <= self.s:
            return None                       # the original returns None too (no batch for this session)
        dev = self.device
        x = sample.in_idxes
        preds = self.model(torch.tensor([x[:self.s + 1]], device=dev), torch.tensor([x[self.s]], device=dev),
                           torch.tensor([self.s + 1], dtype=torch.float32, device=dev))
        self.s = self.s + 1; self.old_session_id = session_id
        return pd.Series(data=preds[0].cpu().numpy(), index=self.item_index)

    def clear(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def support_users(self):
        return False
