'''
Yambda-50M listens (yandex/yambda, flat/50m/listens.parquet), session-aware, one global temporal split cut backward
from the last timestamp with 30-minute gaps (research plan section 3):

    T0   train the backbones, item vocabulary, long-term profiles      [t_min, ...)          hdf 'train', 'valid_train'
    T1a  model selection / early stopping                    3 days                         hdf 'valid_test'
    T1b  post-hoc fit: calibration, (eps, w_min, w_max), Sinkhorn duals   4 days            hdf 'calib'
    T2   held-out test, drift of the dual prices             last 7 days                    hdf 'test'

Each part ends 30 minutes (gap) before the next one starts. A session (30-minute inactivity gap per user, on the
full log) belongs to a part only if it lies completely inside it; sessions that straddle a boundary or sit in a gap
are dropped, so no session leaks across parts. Load T1a with train_eval: true (valid_train / valid_test).

Filters: items with at least min_item_support plays in T0, sessions with min..max_session_length plays, users with
at least min_user_sessions sessions in T0 (repeated until stable). T1a / T1b / T2 keep only T0 items, then sessions
with at least min_session_length plays (users unseen in T0 are kept: the session-based backbones can serve them).

Columns: SessionId, UserId (uid), ItemId (item_id), Time (seconds since the start of the log, 5 s resolution),
is_organic, played_ratio_pct, track_length_seconds. min_played_ratio (optional) keeps only listens with
played_ratio_pct >= the value (Yambda's "listen+" uses 50); the default keeps all listens.
'''
import os

import numpy as np
import pandas as pd

USER_KEY, ITEM_KEY, TIME_KEY, SESSION_KEY = 'UserId', 'ItemId', 'Time', 'SessionId'
DAY = 86400


def load_data(file):
    data = pd.read_parquet(file + '.parquet')
    data = data.rename(columns={'uid': USER_KEY, 'item_id': ITEM_KEY, 'timestamp': TIME_KEY})
    data[TIME_KEY] = data[TIME_KEY].astype(np.int64)
    data[USER_KEY] = data[USER_KEY].astype(np.int64)
    data[ITEM_KEY] = data[ITEM_KEY].astype(np.int64)
    data = data.sort_values([USER_KEY, TIME_KEY], kind='mergesort').reset_index(drop=True)
    _stats('Loaded', data)
    return data


def filter_data(data, min_played_ratio=None, session_gap=1800):
    '''Optional played-ratio filter, then sessions by an inactivity gap of session_gap seconds per user.'''
    if min_played_ratio is not None:
        data = data[data['played_ratio_pct'] >= min_played_ratio].reset_index(drop=True)
        print('kept listens with played_ratio_pct >= {}'.format(min_played_ratio))
    t, u = data[TIME_KEY].values, data[USER_KEY].values
    new = np.r_[True, (np.diff(t) > session_gap) | (u[1:] != u[:-1])]
    data[SESSION_KEY] = np.cumsum(new)
    cols = [SESSION_KEY, USER_KEY, ITEM_KEY, TIME_KEY] + [c for c in data.columns
                                                         if c not in (SESSION_KEY, USER_KEY, ITEM_KEY, TIME_KEY)]
    data = data[cols]
    _stats('Sessions', data)
    return data


def split_data(data, output_file, test_days=7, calib_days=4, valid_days=3, gap=1800, min_item_support=20,
               min_session_length=5, max_session_length=200, min_user_sessions=3):
    t_max = int(data[TIME_KEY].max())
    # [lo, hi) per part, backward from the end
    t2 = (t_max - test_days * DAY, t_max + 1)
    t1b = (t2[0] - gap - calib_days * DAY, t2[0] - gap)
    t1a = (t1b[0] - gap - valid_days * DAY, t1b[0] - gap)
    t0 = (int(data[TIME_KEY].min()), t1a[0] - gap)
    windows = [('T0', t0), ('T1a', t1a), ('T1b', t1b), ('T2', t2)]
    for name, (lo, hi) in windows:
        print('{:4s} [{:>10d}, {:>10d})  day {:7.2f} - {:7.2f}  ({:.2f} days)'.format(
            name, lo, hi, lo / DAY, hi / DAY, (hi - lo) / DAY))

    g = data.groupby(SESSION_KEY)[TIME_KEY]
    start, stop = g.transform('min').values, g.transform('max').values
    parts = {}
    inside = np.zeros(len(data), dtype=bool)
    for name, (lo, hi) in windows:
        m = (start >= lo) & (stop < hi)
        inside |= m
        parts[name] = data[m]
    dropped = data[~inside]
    print('dropped {} events / {} sessions that straddle a boundary or fall into a gap'.format(
        len(dropped), dropped[SESSION_KEY].nunique()))

    train = parts['T0']
    while True:
        n = len(train)
        train = train[train[ITEM_KEY].map(train[ITEM_KEY].value_counts()) >= min_item_support]
        slen = train[SESSION_KEY].map(train[SESSION_KEY].value_counts())
        train = train[(slen >= min_session_length) & (slen <= max_session_length)]
        train = train[train[USER_KEY].map(train.groupby(USER_KEY)[SESSION_KEY].nunique()) >= min_user_sessions]
        if len(train) == n:
            break
    items = train[ITEM_KEY].unique()

    out = {'T0': train}
    for name in ('T1a', 'T1b', 'T2'):
        p = parts[name]
        slen = p[SESSION_KEY].map(p[SESSION_KEY].value_counts())
        p = p[slen <= max_session_length]  # as in T0: the length limit applies to the whole session
        p = p[p[ITEM_KEY].isin(items)]
        slen = p[SESSION_KEY].map(p[SESSION_KEY].value_counts())
        out[name] = p[slen >= min_session_length]

    for name, part in out.items():
        _stats(name, part)
    for name in ('T1a', 'T1b', 'T2'):
        print('{} users seen in T0: {:.1%}'.format(
            name, np.isin(out[name][USER_KEY].unique(), train[USER_KEY].unique()).mean()))

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    file = output_file + '.hdf'
    if os.path.exists(file):
        os.remove(file)
    for key, name in [('train', 'T0'), ('valid_train', 'T0'), ('valid_test', 'T1a'), ('calib', 'T1b'),
                      ('test', 'T2')]:
        out[name].reset_index(drop=True).to_hdf(file, key=key)
    print('written', file)


def _stats(title, data):
    print('{}\n\tEvents: {}\n\tSessions: {}\n\tUsers: {}\n\tItems: {}\n\tSpan: day {:.2f} - {:.2f}\n'.format(
        title, len(data), data[SESSION_KEY].nunique() if SESSION_KEY in data else '-', data[USER_KEY].nunique(),
        data[ITEM_KEY].nunique(), data[TIME_KEY].min() / DAY, data[TIME_KEY].max() / DAY), flush=True)
