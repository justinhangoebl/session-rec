'''
Candidate pools for the Sinkhorn / OT stage (research plan sections 4.1-4.3): a frozen backbone scores every request
of a data part (T1 for calibration and the duals, T2 for serving) and the top-N items with their logits are stored.

A request is a session prefix: after event t of a session (t = 0 .. len-2) the model scores all items for the next
one, given the prefix (and, for history models such as EASE, the user's history). The target is event t+1.

Output, one folder per part (<model folder>/candidates/<part>/):

    requests.parquet   one row per request (row r <-> row r of the arrays):
                       SessionId, UserId, Time (of the last prefix event), pos (t), ItemId (last prefix item),
                       target (next ItemId), target_rank (1 = top; n_items + 1 if the model does not know it),
                       target_score (logit of the target), lse (logsumexp of all logits: log p(target) =
                       target_score - lse), and the other columns of the target event as target_<column>
                       (Yambda: target_is_organic, target_played_ratio_pct, ...)
    items.npy          [R, N] original ItemIds of the top-N, best first
    scores.npy         [R, N] float32 logits of items.npy
    (np.load(..., mmap_mode='r') reads them lazily)

metrics.json holds next-item HitRate / MRR / NDCG @ 1, 5, 10, 20 and Coverage@20 per part (one target per request;
session-rec's NDCG / Recall of evaluation_user_based count all remaining items of the session instead).

User history (models with uses_history: EASE with use_history gets the set of items, SASRec with sequence 'user' the
latest max_seq_length events in time order): 'rolling' (default) = every event of the user
before the session starts (training data, the earlier parts and the earlier sessions of this part; the model stays
frozen, only its query grows), 'train' = the training data only.

Models provide:
    candidate_item_ids()                      original ItemIds of the score columns
    score_sessions(sessions, histories)       sessions: list of int64 arrays of column indices (-1: unknown item),
                                              histories: list of column index arrays (or None); yields
                                              (request indices in session-major order, scores [r, n_items] on GPU)
    uses_history (attribute, optional)
'''
import json
import os
import time

import numpy as np
import pandas as pd
import torch

SESSION_KEY, ITEM_KEY, TIME_KEY, USER_KEY = 'SessionId', 'ItemId', 'Time', 'UserId'
CUTOFFS = (1, 5, 10, 20)


def supported(model):
    return hasattr(model, 'score_sessions') and hasattr(model, 'candidate_item_ids')


def _sorted(part):
    part = part.copy()
    part['_start'] = part.groupby(SESSION_KEY)[TIME_KEY].transform('min')
    part = part.sort_values(['_start', SESSION_KEY, TIME_KEY], kind='mergesort').drop(columns='_start')
    return part.reset_index(drop=True)


class _History:
    '''Per user: the column indices of their events in time order, to cut at a session start.'''

    def __init__(self, frames, colidx):
        df = pd.concat([f[[USER_KEY, ITEM_KEY, TIME_KEY]] for f in frames if f is not None and len(f)])
        df = df.sort_values([USER_KEY, TIME_KEY], kind='mergesort')
        cols = colidx(df[ITEM_KEY].values)
        users = df[USER_KEY].values
        times = df[TIME_KEY].values.astype(np.int64)
        bounds = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1, len(df)]
        self.data = {users[s]: (times[s:e], cols[s:e]) for s, e in zip(bounds[:-1], bounds[1:])}

    def before(self, user, t, kind='set', last=None, with_times=False):
        '''The user's items before time t: 'set' = unique column indices, 'sequence' = the last `last` events in
        time order (repeats kept).'''
        entry = self.data.get(user)
        if entry is None:
            empty = np.empty(0, dtype=np.int64)
            return (empty, empty) if with_times else empty
        times, cols = entry
        n = np.searchsorted(times, t, side='left')
        h, ht = cols[:n], times[:n]
        keep = h >= 0
        h, ht = h[keep], ht[keep]
        if kind == 'sequence':
            if last:
                h, ht = h[-last:], ht[-last:]
            return (h, ht) if with_times else h
        return np.unique(h)


def score_part(model, part, out_dir, k=200, history_frames=None, batch_requests=2048, desc=''):
    '''Score every request of part (a DataFrame with SessionId, ItemId, Time, UserId, ...); returns the metrics.'''
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    part = _sorted(part)
    item_ids = np.asarray(model.candidate_item_ids())
    n_items = len(item_ids)
    lookup = pd.Series(np.arange(n_items, dtype=np.int64), index=item_ids)

    def colidx(ids):
        return lookup.reindex(ids).fillna(-1).values.astype(np.int64)

    cols = colidx(part[ITEM_KEY].values)
    sess = part[SESSION_KEY].values
    bounds = np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1, len(part)]
    lengths = np.diff(bounds)
    keep = lengths >= 2
    starts, lengths = bounds[:-1][keep], lengths[keep]

    # request r: input row = starts[i] + t, target row = starts[i] + t + 1
    n_req = int((lengths - 1).sum())
    req_offsets = np.r_[0, np.cumsum(lengths - 1)]
    input_rows = np.concatenate([np.arange(s, s + l - 1) for s, l in zip(starts, lengths)]) if n_req else np.empty(0, int)
    target_rows = input_rows + 1
    target_cols = torch.as_tensor(cols[target_rows], device=_device(model))

    uses_history = getattr(model, 'uses_history', False) and USER_KEY in part.columns
    history = _History(list(history_frames or []) + [part], colidx) if uses_history else None
    history_kind = getattr(model, 'history_kind', 'set')
    uses_times = getattr(model, 'uses_times', False)
    history_last = getattr(model, 'max_seq_length', None)

    k = min(k, n_items)
    top_items = np.empty((n_req, k), dtype=np.int64)
    top_scores = np.empty((n_req, k), dtype=np.float32)
    t_rank = np.empty(n_req, dtype=np.int64)
    t_score = np.empty(n_req, dtype=np.float32)
    lse = np.empty(n_req, dtype=np.float32)
    covered = torch.zeros(n_items, dtype=torch.bool, device=_device(model))

    users = part[USER_KEY].values if USER_KEY in part.columns else None
    times = part[TIME_KEY].values.astype(np.int64)
    batch_requests = getattr(model, 'candidate_batch_requests', batch_requests)
    i = 0
    n_sessions = len(starts)
    last_print = time.time()
    while i < n_sessions:
        j = i
        total = 0
        while j < n_sessions and (j == i or total + lengths[j] - 1 <= batch_requests):
            total += lengths[j] - 1
            j += 1
        sessions = [cols[starts[s]:starts[s] + lengths[s]] for s in range(i, j)]
        histories, kwargs = None, {}
        if history is not None:
            hs = [history.before(users[starts[s]], times[starts[s]], kind=history_kind, last=history_last,
                                 with_times=uses_times) for s in range(i, j)]
            if uses_times:
                histories, kwargs['history_times'] = [h for h, _ in hs], [t for _, t in hs]
            else:
                histories = hs
        if uses_times:
            kwargs['times'] = [times[starts[s]:starts[s] + lengths[s]] for s in range(i, j)]
            kwargs['users'] = [users[starts[s]] if users is not None else None for s in range(i, j)]
        base = req_offsets[i]
        for local, scores in model.score_sessions(sessions, histories, **kwargs):
            rows = base + np.asarray(local)
            scores = scores.float()
            tc = target_cols[torch.as_tensor(rows, device=scores.device)]
            ts = scores.gather(1, tc.clamp(min=0).unsqueeze(1)).squeeze(1)
            ts = torch.where(tc >= 0, ts, torch.full_like(ts, -float('inf')))
            rank = (scores > ts.unsqueeze(1)).sum(1) + 1
            rank = torch.where(tc >= 0, rank, torch.full_like(rank, n_items + 1))
            s, idx = scores.topk(k, dim=1)
            covered[idx[:, :20].reshape(-1)] = True
            top_items[rows] = idx.cpu().numpy()
            top_scores[rows] = s.cpu().numpy()
            t_rank[rows] = rank.cpu().numpy()
            t_score[rows] = ts.cpu().numpy()
            lse[rows] = torch.logsumexp(scores, 1).cpu().numpy()
        i = j
        if time.time() - last_print > 60:
            last_print = time.time()
            print('candidates {}: {}/{} requests, {:.0f}s'.format(desc, req_offsets[i], n_req, time.time() - t_start),
                  flush=True)

    top_items = item_ids[top_items]
    if top_items.max(initial=0) < 2 ** 31 and top_items.min(initial=0) >= -2 ** 31:
        top_items = top_items.astype(np.int32)
    np.save(os.path.join(out_dir, 'items.npy'), top_items)
    np.save(os.path.join(out_dir, 'scores.npy'), top_scores)

    keys = [SESSION_KEY, USER_KEY, TIME_KEY, ITEM_KEY]
    req = part.loc[input_rows, [c for c in keys if c in part.columns]].reset_index(drop=True)
    req.insert(3, 'pos', input_rows - np.repeat(starts, lengths - 1))
    req['target'] = part[ITEM_KEY].values[target_rows]
    req['target_rank'] = t_rank
    req['target_score'] = t_score
    req['lse'] = lse
    for c in part.columns:
        if c not in keys:
            req['target_' + c] = part[c].values[target_rows]
    req.to_parquet(os.path.join(out_dir, 'requests.parquet'), index=False)

    metrics = {'requests': n_req, 'sessions': int(n_sessions), 'n_items': n_items, 'k': k}
    for c in CUTOFFS:
        hit = t_rank <= c
        metrics['HitRate@{}'.format(c)] = float(hit.mean()) if n_req else 0.0
        metrics['MRR@{}'.format(c)] = float(np.where(hit, 1.0 / t_rank, 0).mean()) if n_req else 0.0
        metrics['NDCG@{}'.format(c)] = float(np.where(hit, 1.0 / np.log2(t_rank + 1), 0).mean()) if n_req else 0.0
    metrics['Coverage@20'] = float(covered.float().mean())
    metrics['target_in_pool'] = float((t_rank <= k).mean()) if n_req else 0.0
    known = t_rank <= n_items  # targets the model can score (EASE with max_items: the most played items only)
    metrics['target_known'] = float(known.mean()) if n_req else 0.0
    metrics['NLL_full_softmax'] = float(np.mean((lse - t_score)[known])) if known.any() else 0.0  # known targets
    metrics['time_s'] = time.time() - t_start
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=1)
    print('candidates {}: {} requests, HR@20 {:.4f}, MRR@20 {:.4f}, NDCG@20 {:.4f}, {:.0f}s -> {}'.format(
        desc, n_req, metrics['HitRate@20'], metrics['MRR@20'], metrics['NDCG@20'], metrics['time_s'], out_dir),
        flush=True)
    return metrics


def score_parts(model, parts, train, out_dir, k=200, history='rolling', batch_requests=2048):
    '''
    parts: [(name, DataFrame)] in time order. With history 'rolling' each part's queries see the training data and
    all earlier parts. Returns {name: metrics}.
    '''
    assert history in ('rolling', 'train'), history
    out = {}
    earlier = [train]
    for name, df in parts:
        out[name] = score_part(model, df, os.path.join(out_dir, 'candidates', name), k=k,
                               history_frames=earlier if history == 'rolling' else [train],
                               batch_requests=batch_requests, desc=name)
        earlier = earlier + [df]
    with open(os.path.join(out_dir, 'candidates', 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=1)
    return out


def results_list(metrics):
    '''metrics dict of a part -> session-rec style [(name, value)] for results csv / wandb.'''
    return [(k + ': ', v) for k, v in metrics.items() if isinstance(v, (int, float))]


def _device(model):
    d = getattr(model, 'device', None)
    return torch.device(d) if d is not None else torch.device('cpu')
