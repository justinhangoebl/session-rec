"""UGRU4Rec on the official GRU4Rec PyTorch implementation: GRU4RecTorch plus the reminder / extend-session hooks of
session-rec's UGRU4Rec (unchanged logic). The official PyTorch GRU4Rec has no top1-max loss."""
import numpy as np
import pandas as pd

from algorithms.extensions.reminder import Reminder
from algorithms.gru4rec.adapter import GRU4RecTorch


class UGRU4RecTorch(GRU4RecTorch):
    def __init__(self, extend_session_length=None, reminders=False, remind_strategy='recency', remind_sessions_num=6,
                 reminders_num=3, remind_mode='end', weight_base=1, weight_IRec=0, session_key='SessionId',
                 item_key='ItemId', time_key='Time', user_key='UserId', seed=None, device=None, **gru_params):
        super().__init__(seed=seed, device=device, **gru_params)
        self.session_key, self.item_key, self.time_key, self.user_key = session_key, item_key, time_key, user_key
        self.extend_session_length = extend_session_length
        self.last_user_items = {}
        self.hasReminders = reminders
        if self.hasReminders:
            if remind_strategy == 'hybrid':
                self.reminder = Reminder(remind_strategy=remind_strategy, remind_sessions_num=remind_sessions_num,
                                         weight_base=weight_base, weight_IRec=weight_IRec)
            else:
                self.reminder = Reminder(remind_strategy=remind_strategy, remind_sessions_num=remind_sessions_num,
                                         reminders_num=reminders_num, remind_mode=remind_mode)

    def fit(self, train, test=None):
        super().fit(train)
        # same data the original loops over: train merged with ItemIdx, sorted by session and time
        itemids = train[self.item_key].unique()
        itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)
        data = pd.merge(train, pd.DataFrame({self.item_key: itemids, 'ItemIdx': itemidmap[itemids].values}), on=self.item_key, how='inner')
        data.sort_values([self.session_key, self.time_key], inplace=True)
        index_session = data.columns.get_loc(self.session_key)
        index_item = data.columns.get_loc(self.item_key)
        index_user = data.columns.get_loc(self.user_key)
        for row in data.itertuples(index=False):
            if self.extend_session_length is not None:
                self.extend_session_model_in_loop(row, index_user, index_item)
            if self.hasReminders:
                self.reminder.reminders_fit_in_loop(row, index_user, index_session, index_item)
        if self.hasReminders:
            self.reminder.reminders_fit(data, self.user_key, self.item_key, self.time_key)

    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids=None, skip=False, mode_type='view', timestamp=0):
        series = super().predict_next(session_id, input_item_id, predict_for_item_ids, skip=skip, mode_type=mode_type, timestamp=timestamp)
        if series is None:
            return None
        if self.hasReminders:
            if self.reminder.remind_strategy == 'hybrid':
                series = self.reminder.reminders_predict_next(input_user_id, series, self.item_key, self.time_key, input_timestamp=timestamp)
            else:
                series = self.reminder.reminders_predict_next(input_user_id, series, self.item_key, self.time_key)
        return series

    def support_users(self):
        return True

    def predict_with_training_data(self):
        return False

    def extend_session_model_in_loop(self, row, index_user, index_item):
        if not row[index_user] in self.last_user_items:
            self.last_user_items[row[index_user]] = []
        self.last_user_items[row[index_user]].append(row[index_item])
        if len(self.last_user_items[row[index_user]]) > self.extend_session_length:
            self.last_user_items[row[index_user]] = self.last_user_items[row[index_user]][-self.extend_session_length:]

    def predict_for_extended_model(self, input_user_id):
        prev_items = []
        if self.extend_session_length is not None and input_user_id in self.last_user_items:
            prev_items = self.last_user_items[input_user_id]
        return prev_items
