'''
TiSASRec (Li, Wang & McAuley, WSDM 2020): SASRec with time-interval aware self-attention. Besides the item and the
absolute position, attention sees the time interval between every pair of events i, j of the sequence:

    e_ij = q_i . (k_j + P^K_j + R^K_{r_ij}) / sqrt(d_h),      z_i = sum_j a_ij (v_j + P^V_j + R^V_{r_ij})

with separate absolute position embeddings P^K, P^V for keys and values and interval embeddings R^K, R^V for the
bucketed interval r_ij in 0..time_span. The paper's intervals are personalised and linear:
r_ij = min(floor(|t_i - t_j| / s_u), time_span), s_u = the user's smallest non-zero gap in the training data.

The interval terms are computed without a [batch, L, L, d] tensor: q . R^K over the time_span + 1 buckets, gathered
by r_ij, and the attention weights summed per bucket (scatter_add) times R^V.

Parameters (besides those of SASRec, sasrec.SASRec): time_span (k, number of interval buckets - 1, paper: 256),
interval ('linear' as in the paper | 'log': r = min(floor(log_resolution * log2(1 + |dt| / s_u)), time_span), which
keeps gaps from seconds to months apart; with linear buckets and Yambda's 5 s resolution everything beyond ~20 min
saturates), log_resolution (buckets per doubling, default 8), time_scale ('user' as in the paper, or a number of
seconds for every user). sequence defaults to 'user' (the paper trains on user sequences); 'session' also works.
Loss and training as in SASRec (cross-entropy over all items, or 'bce' with sampled negatives as in the paper).
'''
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from algorithms.sasrec.sasrec import SASRec


class TimeAwareBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        assert hidden_units % num_heads == 0
        self.h, self.dh = num_heads, hidden_units // num_heads
        self.ln1 = nn.LayerNorm(hidden_units)
        self.q = nn.Linear(hidden_units, hidden_units)
        self.k = nn.Linear(hidden_units, hidden_units)
        self.v = nn.Linear(hidden_units, hidden_units)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)
        self.ln2 = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU(), nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_units, hidden_units), nn.Dropout(dropout_rate))

    def _heads(self, x):  # [B, L, d] -> [B, h, L, dh]
        return x.view(x.size(0), x.size(1), self.h, self.dh).transpose(1, 2)

    def forward(self, x, pos_k, pos_v, time_k, time_v, buckets, mask, valid_mask):
        '''pos_k / pos_v: [L, d]; time_k / time_v: [k + 1, d]; buckets: [B, L, L] long; mask: [B, L, L] True = masked.'''
        B, L, _ = x.shape
        n = self.ln1(x)
        q = self._heads(self.q(n))                                     # [B, h, L, dh]
        k = self._heads(self.k(n) + pos_k.unsqueeze(0))
        v = self._heads(self.v(n) + pos_v.unsqueeze(0))
        tk = time_k.view(-1, self.h, self.dh).transpose(0, 1)          # [h, k+1, dh]
        tv = time_v.view(-1, self.h, self.dh).transpose(0, 1)

        scores = q @ k.transpose(-1, -2)                               # [B, h, L, L]
        qt = q @ tk.transpose(-1, -2).unsqueeze(0)                     # [B, h, L, k+1]
        idx = buckets.unsqueeze(1).expand(B, self.h, L, L)
        scores = scores + qt.gather(-1, idx)
        scores = scores / math.sqrt(self.dh)
        scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))

        out = attn @ v                                                 # [B, h, L, dh]
        per_bucket = torch.zeros(B, self.h, L, tk.size(1), device=x.device, dtype=attn.dtype)
        per_bucket.scatter_add_(-1, idx, attn)                         # sum of a_ij per interval bucket
        out = out + per_bucket @ tv.unsqueeze(0)
        out = out.transpose(1, 2).reshape(B, L, -1)
        out = torch.where(valid_mask, out, 0.0)
        x = x + self.out_dropout(out)
        x = x + self.ffn(self.ln2(x))
        return torch.where(valid_mask, x, 0.0)


class TiSASRecModel(nn.Module):
    def __init__(self, num_items, hidden_units=50, max_seq_length=50, num_blocks=2, num_heads=1, dropout_rate=0.2,
                 time_span=256, interval='linear', log_resolution=8):
        super().__init__()
        self.num_items, self.hidden_units, self.max_seq_length = num_items, hidden_units, max_seq_length
        self.time_span, self.interval, self.log_resolution = time_span, interval, log_resolution
        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_k = nn.Embedding(max_seq_length, hidden_units)
        self.pos_v = nn.Embedding(max_seq_length, hidden_units)
        self.time_k = nn.Embedding(time_span + 1, hidden_units)
        self.time_v = nn.Embedding(time_span + 1, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([TimeAwareBlock(hidden_units, num_heads, dropout_rate) for _ in range(num_blocks)])
        self.last_ln = nn.LayerNorm(hidden_units)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.item_emb.weight[1:])
        nn.init.zeros_(self.item_emb.weight[0])
        for emb in (self.pos_k, self.pos_v, self.time_k, self.time_v):
            nn.init.xavier_uniform_(emb.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def buckets(self, times, scale):
        '''times [B, L] (int64 seconds), scale [B] -> interval buckets [B, L, L] in 0..time_span.'''
        dt = (times.unsqueeze(2) - times.unsqueeze(1)).abs().double() / scale.double().view(-1, 1, 1)
        if self.interval == 'log':
            r = torch.floor(self.log_resolution * torch.log2(1 + dt))
        else:
            r = torch.floor(dt)
        return r.clamp(max=self.time_span).long()

    def forward(self, input_seq, times, scale):
        B, L = input_seq.shape
        padding_mask = input_seq == 0
        valid_mask = (~padding_mask).unsqueeze(-1)
        causal = torch.triu(torch.ones(L, L, device=input_seq.device, dtype=torch.bool), diagonal=1)
        eye = torch.eye(L, device=input_seq.device, dtype=torch.bool)
        mask = causal.unsqueeze(0) | (padding_mask.unsqueeze(1) & ~eye.unsqueeze(0))  # self is never masked
        buckets = self.buckets(times, scale)

        x = self.item_emb(input_seq) * math.sqrt(self.hidden_units)
        x = torch.where(valid_mask, self.emb_dropout(x), 0.0)
        pos = torch.arange(L, device=input_seq.device)
        pos_k, pos_v = self.emb_dropout(self.pos_k(pos)), self.emb_dropout(self.pos_v(pos))
        time_k, time_v = self.emb_dropout(self.time_k.weight), self.emb_dropout(self.time_v.weight)
        for block in self.blocks:
            x = block(x, pos_k, pos_v, time_k, time_v, buckets, mask, valid_mask)
        x = self.last_ln(x)
        return torch.where(valid_mask, x, 0.0)

    def predict(self, seq_output, item_indices=None):
        last_hidden = seq_output[:, -1, :]
        if item_indices is None:
            return last_hidden @ self.item_emb.weight[1:].T
        return (last_hidden.unsqueeze(1) * self.item_emb(item_indices)).sum(-1)


class TiSASRec(SASRec):
    uses_times = True

    def __init__(self, time_span=256, interval='linear', log_resolution=8, time_scale='user', sequence='user',
                 **params):
        assert interval in ('linear', 'log'), interval
        super().__init__(sequence=sequence, **params)
        self.time_span, self.interval, self.log_resolution = int(time_span), interval, float(log_resolution)
        self.time_scale = time_scale

    def _fit_time_scales(self, data):
        '''Per user: the smallest non-zero gap between two of their training events (paper); else a constant.'''
        if self.time_scale != 'user':
            self.user_scale, self.default_scale = {}, float(self.time_scale)
            return
        d = data[[self.user_key, self.time_key]].sort_values([self.user_key, self.time_key], kind='mergesort')
        u, t = d[self.user_key].values, d[self.time_key].values.astype(np.int64)
        gap = np.diff(t).astype(np.float64)
        ok = (u[1:] == u[:-1]) & (gap > 0)
        scales = pd.Series(gap[ok], index=u[1:][ok]).groupby(level=0).min()
        self.user_scale = scales.to_dict()
        self.default_scale = float(scales.median()) if len(scales) else 1.0

    def support_users(self):
        return True  # the personal time scale needs the user, also with session sequences

    def _time_scale(self, user):
        return float(self.user_scale.get(user, self.default_scale))

    def _build_model(self):
        return TiSASRecModel(num_items=self.n_items, hidden_units=self.hidden_units,
                             max_seq_length=self.max_seq_length, num_blocks=self.num_blocks,
                             num_heads=self.num_heads, dropout_rate=self.dropout_rate, time_span=self.time_span,
                             interval=self.interval, log_resolution=self.log_resolution)

    def _forward(self, x, times, scale):
        return self.model(x, times, scale)

    @torch.no_grad()
    def export(self):
        rows = self.itemidmap.values
        return {'item_ids': self.itemidmap.index.values,
                'item_embeddings': self.model.item_emb.weight[rows].detach().float().cpu(),
                'time_k': self.model.time_k.weight.detach().float().cpu(),
                'time_v': self.model.time_v.weight.detach().float().cpu()}
