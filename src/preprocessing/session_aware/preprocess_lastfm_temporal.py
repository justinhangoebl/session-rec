'''
LASTFM 1K, session-aware, one global temporal split (research plan section 3):

    T0  train the backbones                      -> hdf keys 'train' and 'valid_train'
    T1  calibration / OT / cached prices         -> hdf key 'valid_test'   (load with train_eval: true)
    T2  held-out serving                         -> hdf key 'test'

Steps: parse (vectorised), drop the few events with bogus timestamps after the end of the crawl (2009-06), sessions
by a 30-minute gap per user, keep the `days` days before `end` (the crawl fades out in May 2009: 27.7k sessions
by 759 users in April, 7.3k in May, 2.5k in June, so the config ends the window on 2009-05-01), filter (item support, session length, sessions per user),
then cut the timeline: a session belongs to the part in which it *starts*. T1 / T2 events of items unknown in T0 are
removed and sessions that fall below min_session_length afterwards are dropped, so every evaluated session has a
prefix the frozen T0 backbones can score.

Unlike the Latifi et al. 2021 protocol (preprocess_lastfm_aware: last session of every user = test, within 5
windows), no training event is later than a test session's start.
'''
import csv
import os

import numpy as np
import pandas as pd
from datetime import datetime, timezone

USER_KEY, ITEM_KEY, TIME_KEY, SESSION_KEY = 'UserId', 'ItemId', 'Time', 'SessionId'
SESSION_THRESHOLD = 30 * 60


def load_data(file):
    names = [USER_KEY, TIME_KEY, 'artist_id', 'artist', ITEM_KEY, 'track']
    data = pd.read_csv(file + '.tsv', sep='\t', names=names, usecols=[USER_KEY, TIME_KEY, ITEM_KEY],
                       quoting=csv.QUOTE_NONE)
    data = data[~pd.isnull(data[ITEM_KEY])].copy()
    data[TIME_KEY] = (pd.to_datetime(data[TIME_KEY], utc=True) - pd.Timestamp(0, tz='UTC')) // pd.Timedelta('1s')

    # a few plays carry timestamps years after the end of the crawl (up to 2013); the data really ends 2009-06
    end = data[TIME_KEY].quantile(0.9999)
    print('dropping {} events after {}'.format(int((data[TIME_KEY] > end).sum()), _date(end)))
    data = data[data[TIME_KEY] <= end]

    data[USER_KEY] = pd.factorize(data[USER_KEY])[0]
    data[ITEM_KEY] = pd.factorize(data[ITEM_KEY])[0]
    data = data.sort_values([USER_KEY, TIME_KEY], kind='mergesort')
    new_session = np.r_[True, (np.diff(data[TIME_KEY].values) > SESSION_THRESHOLD) |
                        (data[USER_KEY].values[1:] != data[USER_KEY].values[:-1])]
    data[SESSION_KEY] = np.cumsum(new_session)
    _stats('Loaded', data)
    return data


def filter_data(data, end=None, days=180, min_item_support=10, min_session_length=5, max_session_length=200,
                min_user_sessions=3):
    '''
    Keep the sessions that start in the `days` days before `end` (ISO date, exclusive; default: the end of the data)
    and end before it, then filter until all conditions hold.
    '''
    start = data.groupby(SESSION_KEY)[TIME_KEY].transform('min')
    stop = data.groupby(SESSION_KEY)[TIME_KEY].transform('max')
    end_ts = data[TIME_KEY].max() + 1 if end is None else pd.Timestamp(end, tz='UTC').timestamp()
    data = data[(start >= end_ts - days * 86400) & (stop < end_ts)]
    while True:
        n = len(data)
        data = data[data[ITEM_KEY].map(data[ITEM_KEY].value_counts()) >= min_item_support]
        slen = data[SESSION_KEY].map(data[SESSION_KEY].value_counts())
        data = data[(slen >= min_session_length) & (slen <= max_session_length)]
        data = data[data[USER_KEY].map(data.groupby(USER_KEY)[SESSION_KEY].nunique()) >= min_user_sessions]
        if len(data) == n:
            break
    _stats('Filtered (last {} days)'.format(days), data)
    return data


def split_data(data, output_file, t1_share=0.1, t2_share=0.1, min_session_length=5):
    '''Global cut by session start time: T0 | T1 (t1_share of the timeline) | T2 (the last t2_share).'''
    start = data.groupby(SESSION_KEY)[TIME_KEY].transform('min')
    t_min, t_max = data[TIME_KEY].min(), data[TIME_KEY].max()
    cut1 = t_max - (t1_share + t2_share) * (t_max - t_min)
    cut2 = t_max - t2_share * (t_max - t_min)
    t0 = data[start < cut1]
    t1 = _known(data[(start >= cut1) & (start < cut2)], t0, min_session_length)
    t2 = _known(data[start >= cut2], t0, min_session_length)

    print('Cutoffs: T0 < {} <= T1 < {} <= T2'.format(_date(cut1), _date(cut2)))
    for name, part in [('T0 train', t0), ('T1 valid_test', t1), ('T2 test', t2)]:
        _stats(name, part)
    print('T2 users seen in T0: {:.1%}'.format(np.isin(t2[USER_KEY].unique(), t0[USER_KEY].unique()).mean()))

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    out = output_file + '.hdf'
    for key, part in [('train', t0), ('valid_train', t0), ('valid_test', t1), ('test', t2)]:
        part.reset_index(drop=True).to_hdf(out, key=key)
    print('written', out)


def _known(part, train, min_session_length):
    '''Drop events of items unseen in train, then sessions that became shorter than min_session_length.'''
    part = part[part[ITEM_KEY].isin(train[ITEM_KEY].unique())]
    slen = part[SESSION_KEY].map(part[SESSION_KEY].value_counts())
    return part[slen >= min_session_length]


def _date(ts):
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat()


def _stats(title, data):
    print('{}\n\tEvents: {}\n\tSessions: {}\n\tUsers: {}\n\tItems: {}\n\tSpan: {} / {}\n'.format(
        title, len(data), data[SESSION_KEY].nunique(), data[USER_KEY].nunique(), data[ITEM_KEY].nunique(),
        _date(data[TIME_KEY].min()), _date(data[TIME_KEY].max())))
