"""PyTorch UNARM: the NARM port plus the reminder / extend-session hooks of session-rec's UNARM
(unchanged logic)."""
from algorithms.extensions.reminder import Reminder
from algorithms.narm.narm import NARM


class UNARM(NARM):
    def __init__(self, factors=100, hidden_units=100, epochs=30, lr=0.001, extend_session_length=None,
                 reminders=False, remind_strategy='recency', remind_sessions_num=6, reminders_num=3, remind_mode='end',
                 weight_base=1, weight_IRec=0, session_key='SessionId', item_key='ItemId', time_key='Time',
                 user_key='UserId', seed=None, device=None):
        super().__init__(factors=factors, hidden_units=hidden_units, epochs=epochs, lr=lr, session_key=session_key,
                         item_key=item_key, seed=seed, device=device)
        self.time_key = time_key
        self.user_key = user_key
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

    def create_training_data(self, data):
        index_session = data.columns.get_loc(self.session_key)
        index_item = data.columns.get_loc('ItemIdx')
        index_item_original = data.columns.get_loc('ItemId')
        index_user = data.columns.get_loc('UserId')
        out_seqs, labs, session, session_items = [], [], -1, []
        for row in data.itertuples(index=False):
            if row[index_session] != session:
                session = row[index_session]; session_items = list()
            session_items.append(row[index_item])
            if self.extend_session_length is not None:
                self.extend_session_model_in_loop(row, index_user, index_item_original)
            if self.hasReminders:
                self.reminder.reminders_fit_in_loop(row, index_user, index_session, index_item_original)
            if len(session_items) > 1:
                out_seqs += [session_items[:-1]]; labs += [session_items[-1]]
        if self.hasReminders:
            self.reminder.reminders_fit(data, self.user_key, self.item_key, self.time_key)
        return out_seqs, labs

    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids, timestamp=0, skip=False, mode_type='view'):
        series = super().predict_next(session_id, input_item_id, predict_for_item_ids, timestamp=timestamp, skip=skip, mode_type=mode_type)
        if skip:
            return
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
        self.session_items = list()
        if self.extend_session_length is not None and input_user_id in self.last_user_items:
            prev_items = self.last_user_items[input_user_id]
        return prev_items
