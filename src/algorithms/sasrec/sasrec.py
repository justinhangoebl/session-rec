'''
SASRec (Kang & McAuley, ICDM 2018) for session-based recommendation: a causal self-attention encoder over the items
of the current session, trained on next-item prediction for every position.

Parameters: hidden_units, max_seq_length (sessions longer than this are cut into windows for training; prediction
uses the last max_seq_length items), num_blocks, num_heads, dropout_rate, batch_size, learning_rate, l2_emb (L2 on
the item embeddings), n_epochs, loss ('bce' with n_sample negatives per position | 'cross-entropy' over all items),
seed, device ('cuda:0' / 'cpu'; default cuda if available), sequence:

    'session'  (default) one sequence per session: trained on sessions, predicts from the session prefix only
    'user'     one sequence per user: all training events of the user in time order (sessions concatenated, cut into
               windows of max_seq_length); predicts from the user's latest events before the session + the prefix
               (session-aware, as in the Yambda paper; the candidate pools pass the rolling user history)
'''
import logging
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation import tracking

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_units)
        self.attn = nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout_rate)

        self.ln2 = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # Pre-LN causal self-attention; attn_mask: [batch * heads, L, L], True = not allowed to attend
        norm_x = self.ln1(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=attn_mask, need_weights=False)
        attn_output = torch.where(valid_mask, attn_output, 0.0)
        x = x + self.attn_dropout(attn_output)

        # Pre-LN point-wise FFN
        x = x + self.ffn(self.ln2(x))
        return torch.where(valid_mask, x, 0.0)


class SASRecModel(nn.Module):
    def __init__(self, num_items: int, hidden_units: int = 50, max_seq_length: int = 50, num_blocks: int = 2,
                 num_heads: int = 1, dropout_rate: float = 0.2):
        super().__init__()
        self.num_items = num_items
        self.hidden_units = hidden_units
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads

        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_length, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList([TransformerBlock(hidden_units, num_heads, dropout_rate) for _ in range(num_blocks)])
        self.last_ln = nn.LayerNorm(hidden_units)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.item_emb.weight[1:])
        nn.init.zeros_(self.item_emb.weight[0])
        nn.init.xavier_uniform_(self.pos_emb.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def attention_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        '''
        Causal + key padding mask, [batch * heads, L, L] (True = masked). Every position may attend to itself, so the
        rows of the left padding are never fully masked (a fully masked softmax row gives NaN values and gradients).
        '''
        seq_len = padding_mask.size(1)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=padding_mask.device, dtype=torch.bool), diagonal=1)
        eye = torch.eye(seq_len, device=padding_mask.device, dtype=torch.bool)
        mask = causal.unsqueeze(0) | (padding_mask.unsqueeze(1) & ~eye.unsqueeze(0))
        return mask.repeat_interleave(self.num_heads, dim=0)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # input_seq: [batch_size, seq_len], left-padded with 0, valid items in 1..num_items
        seq_len = input_seq.size(1)
        padding_mask = input_seq == 0
        valid_mask = (~padding_mask).unsqueeze(-1)
        attn_mask = self.attention_mask(padding_mask)

        positions = torch.arange(seq_len, dtype=torch.long, device=input_seq.device).unsqueeze(0)
        seq_emb = self.item_emb(input_seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        seq_emb = self.emb_dropout(seq_emb)
        seq_emb = torch.where(valid_mask, seq_emb, 0.0)

        for block in self.blocks:
            seq_emb = block(seq_emb, attn_mask, valid_mask)

        seq_emb = self.last_ln(seq_emb)
        return torch.where(valid_mask, seq_emb, 0.0)

    def predict(self, seq_output: torch.Tensor, item_indices: torch.Tensor = None) -> torch.Tensor:
        # Left-padded input: index -1 holds the state after the last interaction
        last_hidden = seq_output[:, -1, :]
        if item_indices is None:
            return torch.matmul(last_hidden, self.item_emb.weight[1:].T)  # [batch, num_items], item idx 1..n
        item_emb = self.item_emb(item_indices)
        return torch.matmul(last_hidden.unsqueeze(1), item_emb.transpose(-1, -2)).squeeze(1)


class SASRec:
    def __init__(self, hidden_units=50, max_seq_length=50, num_blocks=2, num_heads=1, dropout_rate=0.2,
                 batch_size=128, learning_rate=0.001, l2_emb=0.0, n_epochs=10, loss='bce', n_sample=1, seed=None,
                 device=None, sequence='session', session_key='SessionId', item_key='ItemId', time_key='Time',
                 user_key='UserId'):
        assert sequence in ('session', 'user'), sequence
        self.sequence = sequence
        self.user_key = user_key
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.hidden_units = hidden_units
        self.max_seq_length = max_seq_length
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_emb = l2_emb
        self.n_epochs = n_epochs
        self.loss = loss
        self.set_loss_function(loss)
        self.n_sample = n_sample
        self.seed = seed
        self.session_key, self.item_key, self.time_key = session_key, item_key, time_key
        self.model = None

    def set_loss_function(self, loss: str):
        if loss == 'bce':
            self.loss_function = self.bce_loss_with_neg_sampling
        elif loss == 'cross-entropy':
            self.loss_function = self.ce_loss
        else:
            raise NotImplementedError(f'Unsupported loss: {loss}')

    # ------------------------------------------------------------------ training data

    def _prepare_sequences(self, data: pd.DataFrame):
        '''
        One left-padded (input, target) pair of length max_seq_length per session; sessions longer than
        max_seq_length + 1 are cut into several windows from the end.
        '''
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        # 1-indexed so 0 stays reserved for padding
        self.itemidmap = pd.Series(data=np.arange(1, self.n_items + 1, dtype=np.int64), index=itemids, name='ItemIdx')

        # Sessions in the order of their start time, events in time order (stable: ties keep the file order)
        with_user = self.sequence == 'user' or (self.uses_times and self.user_key in data.columns)
        if self.uses_times:
            self._fit_time_scales(data)
        if self.sequence == 'user':  # one sequence per user: all of their events in time order
            data = data[[self.user_key, self.item_key, self.time_key]].copy()
            data = data.sort_values([self.user_key, self.time_key], kind='mergesort')
            group_key = self.user_key
        else:
            data = data[[self.session_key, self.item_key, self.time_key] + ([self.user_key] if with_user else [])].copy()
            data['start'] = data.groupby(self.session_key)[self.time_key].transform('min')
            data = data.sort_values(['start', self.session_key, self.time_key], kind='mergesort')
            group_key = self.session_key

        mapped_items = self.itemidmap[data[self.item_key].values].values
        if self.sequence == 'user':  # the latest max_seq_length training items per user, for predict_next
            users = data[self.user_key].values
            ends = np.r_[np.flatnonzero(users[1:] != users[:-1]), len(users) - 1]
            self.user_history = {users[e]: mapped_items[max(0, e + 1 - self.max_seq_length):e + 1].copy()
                                 for e in ends}
            times_all = data[self.time_key].values.astype(np.int64)
            self.user_history_times = {users[e]: times_all[max(0, e + 1 - self.max_seq_length):e + 1].copy()
                                       for e in ends}
        session_change = np.r_[True, data[group_key].values[1:] != data[group_key].values[:-1]]
        offsets = np.r_[np.flatnonzero(session_change), len(data)]

        times = data[self.time_key].values.astype(np.int64) if self.uses_times else None
        row_users = data[self.user_key].values if (self.uses_times and with_user) else None
        max_len = self.max_seq_length
        inputs, targets, in_times, scales = [], [], [], []
        for s, e in zip(offsets[:-1], offsets[1:]):
            seq = mapped_items[s:e]
            end = len(seq)
            while end > 1:
                start = max(0, end - 1 - max_len)
                x = np.zeros(max_len, dtype=np.int64)
                y = np.zeros(max_len, dtype=np.int64)
                n = end - 1 - start
                x[max_len - n:] = seq[start:end - 1]
                y[max_len - n:] = seq[start + 1:end]
                inputs.append(x)
                targets.append(y)
                if self.uses_times:
                    t = np.zeros(max_len, dtype=np.int64)
                    t[max_len - n:] = times[s + start:s + end - 1]
                    in_times.append(t)
                    scales.append(self._time_scale(row_users[s] if row_users is not None else None))
                end -= max_len

        extras = []
        if self.uses_times:
            extras = [torch.from_numpy(np.stack(in_times)), torch.tensor(scales, dtype=torch.float32)]
        return torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(targets)), extras

    # ------------------------------------------------------------------ hooks (TiSASRec overrides them)

    uses_times = False  # the model reads the event times (TiSASRec)

    def _fit_time_scales(self, data):
        pass

    def _time_scale(self, user):
        return 1.0

    def _build_model(self):
        return SASRecModel(num_items=self.n_items, hidden_units=self.hidden_units,
                           max_seq_length=self.max_seq_length, num_blocks=self.num_blocks,
                           num_heads=self.num_heads, dropout_rate=self.dropout_rate)

    def _forward(self, x, *extras):
        '''Sequence output [batch, L, d] for left-padded item rows x (extras: event times, time scales).'''
        return self.model(x)

    # ------------------------------------------------------------------ losses

    def bce_loss_with_neg_sampling(self, seq_output: torch.Tensor, pos_targets: torch.Tensor) -> torch.Tensor:
        valid = pos_targets > 0
        pos_emb = self.model.item_emb(pos_targets)
        pos_logits = (seq_output * pos_emb).sum(dim=-1)

        neg_targets = torch.randint(1, self.n_items + 1, (*pos_targets.shape, self.n_sample), device=seq_output.device)
        # resample negatives that hit the positive item (one pass is enough to make collisions negligible)
        collisions = neg_targets == pos_targets.unsqueeze(-1)
        neg_targets = torch.where(collisions, torch.randint_like(neg_targets, 1, self.n_items + 1), neg_targets)

        neg_emb = self.model.item_emb(neg_targets)
        neg_logits = torch.einsum('btd,btsd->bts', seq_output, neg_emb)

        pos_loss = F.binary_cross_entropy_with_logits(pos_logits[valid], torch.ones_like(pos_logits[valid]))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits[valid], torch.zeros_like(neg_logits[valid]))
        return pos_loss + neg_loss

    def ce_loss(self, seq_output: torch.Tensor, pos_targets: torch.Tensor) -> torch.Tensor:
        valid = pos_targets > 0
        logits = torch.matmul(seq_output[valid], self.model.item_emb.weight[1:].T)
        return F.cross_entropy(logits, pos_targets[valid] - 1)

    # ------------------------------------------------------------------ session-rec interface

    def fit(self, data: pd.DataFrame, test: pd.DataFrame = None):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        self.error_during_train = False
        t0 = time.time()
        input_seqs, target_seqs, extras = self._prepare_sequences(data)
        logger.info(type(self).__name__ + ': %d training sequences of length %d, %d items, device %s (%.1fs)',
                    len(input_seqs), self.max_seq_length, self.n_items, self.device, time.time() - t0)

        # The whole training set lives on the device; batches are slices of a random permutation
        input_seqs = input_seqs.to(self.device)
        target_seqs = target_seqs.to(self.device)
        extras = [e.to(self.device) for e in extras]
        n = len(input_seqs)

        self.model = self._build_model().to(self.device)
        tracking.watch(self.model)

        emb_params = [self.model.item_emb.weight]
        other_params = [p for name, p in self.model.named_parameters() if name != 'item_emb.weight']
        opt = torch.optim.Adam([{'params': emb_params, 'weight_decay': self.l2_emb},
                                {'params': other_params, 'weight_decay': 0.0}],
                               lr=self.learning_rate, betas=(0.9, 0.98))

        # TF32 matmuls (Ampere and later): the full-softmax cross-entropy over all items dominates the cost
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            self._train_epochs(input_seqs, target_seqs, extras, opt, n)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32  # inference (and the logits of the pools) in full fp32

        self.model.eval()
        self.session_id = None
        self.session_items = []
        self._pred_ids = None

    def _train_epochs(self, input_seqs, target_seqs, extras, opt, n):
        for epoch in range(self.n_epochs):
            t0 = time.time()
            self.model.train()
            loss_sum = torch.zeros((), device=self.device)
            event_sum = torch.zeros((), device=self.device)
            n_batches = 0
            for idx in torch.randperm(n, device=self.device).split(self.batch_size):
                batch_x, batch_y = input_seqs[idx], target_seqs[idx]
                opt.zero_grad(set_to_none=True)
                loss = self.loss_function(self._forward(batch_x, *[e[idx] for e in extras]), batch_y)
                loss.backward()
                opt.step()

                n_events = (batch_y > 0).sum()
                loss_sum += loss.detach() * n_events
                event_sum += n_events
                n_batches += 1

            avg_loss = float(loss_sum / event_sum.clamp(min=1))
            dt = time.time() - t0
            if np.isnan(avg_loss):
                logger.error('Epoch %d: NaN error!', epoch + 1)
                self.error_during_train = True
                return
            events = float(event_sum)
            logger.info('Epoch %d --> loss: %.6f \t(%.2fs) \t[%.2f mb/s | %.0f e/s]',
                        epoch + 1, avg_loss, dt, n_batches / dt, events / dt)
            tracking.log({'train/epoch': epoch + 1, 'train/loss': avg_loss, 'train/epoch_time_s': dt,
                          'train/events_per_s': events / dt})

    @torch.no_grad()
    def predict_next(self, session_id, input_item_id, *args, timestamp=0, skip=False, mode_type='view'):
        '''Scores of predict_for_item_ids for the next item of the session, given its items so far (sequence 'user':
        after the user's latest training items). session-based: (session_id, item, items_to_predict);
        session-aware (sequence 'user'): (session_id, item, user, items_to_predict).'''
        if self.support_users():
            input_user_id, predict_for_item_ids = args
        else:
            input_user_id, (predict_for_item_ids,) = None, args
        if session_id != self.session_id:
            self.session_id = session_id
            self.session_user = input_user_id
            self.session_items = [int(i) for i in getattr(self, 'user_history', {}).get(input_user_id, [])]
            self.session_times = [int(t) for t in getattr(self, 'user_history_times', {}).get(input_user_id, [])]
        idx = self.itemidmap.get(input_item_id)
        if idx is not None:
            self.session_items.append(int(idx))
            self.session_times.append(int(timestamp))
        if skip:
            return None

        if self._pred_ids is not predict_for_item_ids:
            self._pred_ids = predict_for_item_ids
            pred_idx = self.itemidmap.reindex(predict_for_item_ids).values  # NaN: not a training item
            self._pred_known = ~np.isnan(pred_idx)
            self._pred_rows = torch.as_tensor(pred_idx[self._pred_known].astype(np.int64) - 1, device=self.device)

        seq = self.session_items[-self.max_seq_length:]
        scores = np.full(len(predict_for_item_ids), -np.inf, dtype=np.float32)
        if seq:
            x = torch.zeros((1, self.max_seq_length), dtype=torch.long, device=self.device)
            x[0, self.max_seq_length - len(seq):] = torch.as_tensor(seq, device=self.device)
            extras = []
            if self.uses_times:
                t = torch.zeros((1, self.max_seq_length), dtype=torch.long, device=self.device)
                t[0, self.max_seq_length - len(seq):] = torch.as_tensor(self.session_times[-len(seq):], device=self.device)
                extras = [t, torch.tensor([self._time_scale(self.session_user)], device=self.device)]
            all_scores = self.model.predict(self._forward(x, *extras))[0]
            scores[self._pred_known] = all_scores[self._pred_rows].float().cpu().numpy()
        return pd.Series(data=scores, index=predict_for_item_ids)

    # ---------------------------------------------------------------- candidate pools (evaluation/candidates.py)

    def candidate_item_ids(self):
        return self.itemidmap.index.values  # column j <-> embedding row j + 1

    @property
    def uses_history(self):
        return getattr(self, 'sequence', 'session') == 'user'

    @property
    def history_kind(self):
        return 'sequence'  # the latest events in time order, not a set

    @torch.no_grad()
    def score_sessions(self, sessions, histories=None, times=None, history_times=None, users=None, chunk=1024):
        '''Scores after every prefix of every session (column indices, -1 = unknown), as predict_next; with
        sequence 'user' the input is the user's history (column indices in time order) followed by the prefix.
        Models with uses_times also get the event times of the sessions / histories and the users.'''
        self.model.eval()
        L = self.max_seq_length
        x, tx, sc = [], [], []
        for si, seq in enumerate(sessions):
            items, item_times = [], []
            if histories is not None and self.uses_history:
                h = np.asarray(histories[si], dtype=np.int64)
                keep = h >= 0
                items = list(h[keep][-L:] + 1)
                if self.uses_times:
                    item_times = list(np.asarray(history_times[si], dtype=np.int64)[keep][-L:])
            known = np.asarray(seq[:-1], dtype=np.int64) + 1  # embedding rows; 0 = unknown -> skipped
            scale = self._time_scale(users[si] if users is not None else None) if self.uses_times else None
            for t in range(len(known)):
                if known[t] > 0:
                    items.append(known[t])
                    if self.uses_times:
                        item_times.append(times[si][t])
                row = np.zeros(L, dtype=np.int64)
                last = items[-L:]
                if last:
                    row[L - len(last):] = last
                x.append(row)
                if self.uses_times:
                    trow = np.zeros(L, dtype=np.int64)
                    if last:
                        trow[L - len(last):] = item_times[-len(last):]
                    tx.append(trow)
                    sc.append(scale)
        x = torch.as_tensor(np.stack(x), device=self.device) if x else torch.zeros((0, L), dtype=torch.long)
        if self.uses_times and len(tx):
            tx = torch.as_tensor(np.stack(tx), device=self.device)
            sc = torch.as_tensor(np.array(sc, dtype=np.float32), device=self.device)
        for s in range(0, len(x), chunk):
            xb = x[s:s + chunk]
            extras = [tx[s:s + chunk], sc[s:s + chunk]] if self.uses_times else []
            yield np.arange(s, s + len(xb)), self.model.predict(self._forward(xb, *extras))

    def support_users(self):
        return getattr(self, 'sequence', 'session') == 'user'

    def predict_with_training_data(self):
        return False

    @torch.no_grad()
    def export(self):
        '''Item embeddings for downstream use: row i of item_embeddings belongs to item_ids[i] (padding row dropped).'''
        rows = self.itemidmap.values
        return {'item_ids': self.itemidmap.index.values,
                'item_embeddings': self.model.item_emb.weight[rows].detach().float().cpu(),
                'position_embeddings': self.model.pos_emb.weight.detach().float().cpu()}

    def to(self, device):
        self.device = torch.device(device)
        if self.model is not None:
            self.model = self.model.to(self.device).eval()
        self._pred_ids = None
        return self

    def savemodel(self, path: str):
        torch.save(self, path)

    @classmethod
    def loadmodel(cls, path: str, device='cuda:0'):
        sasrec = torch.load(path, map_location=device, weights_only=False)
        return sasrec.to(device)

    def clear(self):
        self.model = None
        self.itemidmap = None
        self._pred_ids = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
