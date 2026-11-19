"""Create a synthetic session-aware dataset from the rsc15 100k split by adding a UserId column.

Deterministic (fixed seed). Sessions are assigned to pseudo-users:
  - with prob P_OWNER a session goes to the 'owner' of its first item (item -> user via a fixed random map),
    so a user's sessions share some items (gives reminder/boost models a signal),
  - otherwise to a uniformly random user.
Test sessions of users without any train session are dropped (the evaluator needs train history).
usage: python reference/synth/make_users.py   (repo root)
"""
import numpy as np, pandas as pd, os, json
SEED, SESS_PER_USER, P_OWNER = 20261119, 6, 0.5
SRC, DST, PREFIX = 'data/rsc15/prepared/', 'data/rsc15_users/prepared/', 'yoochoose-clicks-100k'
tr = pd.read_csv(SRC + PREFIX + '_train_full.txt', sep='\t')
te = pd.read_csv(SRC + PREFIX + '_test.txt', sep='\t')
rng = np.random.RandomState(SEED)
sess = pd.concat([tr.assign(part='train'), te.assign(part='test')])
first = sess.sort_values(['SessionId', 'Time']).groupby('SessionId').ItemId.first()
n_users = max(1, sess.SessionId.nunique() // SESS_PER_USER)
owner = {i: rng.randint(n_users) for i in sorted(sess.ItemId.unique())}
user_of = {}
for s in sorted(first.index):                      # sorted -> deterministic draw order
    user_of[s] = owner[first[s]] if rng.rand() < P_OWNER else rng.randint(n_users)
tr['UserId'] = tr.SessionId.map(user_of); te['UserId'] = te.SessionId.map(user_of)
before = te.SessionId.nunique()
te = te[te.UserId.isin(set(tr.UserId))]
os.makedirs(DST, exist_ok=True)
tr[['SessionId', 'UserId', 'ItemId', 'Time']].to_csv(DST + PREFIX + '_train_full.txt', sep='\t', index=False)
te[['SessionId', 'UserId', 'ItemId', 'Time']].to_csv(DST + PREFIX + '_test.txt', sep='\t', index=False)
info = dict(seed=SEED, users=n_users, train_sessions=int(tr.SessionId.nunique()), test_sessions_before=int(before),
            test_sessions_kept=int(te.SessionId.nunique()), train_users=int(tr.UserId.nunique()), test_users=int(te.UserId.nunique()))
json.dump(info, open(DST + 'synth_info.json', 'w'), indent=1); print(info)
