import pandas as pd


class Reminder:
    '''

    reminders: bool
        Include reminding items in the (main) recommendation list. (default: False)

    remind_strategy: string
        Ranking strategy of the reminding list (default: recency)
        'recency', 'session_similarity', 'hybrid' (only in improved version)

    remind_sessions_num: int
        Number of the last user's sessions that the possible items for reminding are taken from (default: 6)

    reminders_num: int
        length of the reminding list (default: 3)

    remind_mode: string
        The postion of the remining items in recommendation list (top, end). (default: end)


    '''

    def __init__(self, remind_strategy='hybrid', remind_sessions_num=6, reminders_num=3, remind_mode='end',
                 cut_off_threshold=20, weight_base=1, weight_IRec=0, weight_SSim=0):
        self.remind_sessions_num = remind_sessions_num
        self.remind_strategy = remind_strategy
        self.recent_user_items = {}
        self.recent_user_sessions = {}
        self.prev_session_id = -1
        if self.remind_strategy == 'hybrid':
            self.w_base = weight_base
            self.w_IRec = weight_IRec
            self.w_SSim = weight_SSim
        else:
            self.remind_mode = remind_mode
            self.reminders_num = reminders_num
            self.cut_off = cut_off_threshold
            if self.remind_strategy == 'session_similarity':
                self.user_item_intensity = dict()

    def reminders_fit_in_loop(self, row, index_user, index_session, index_item):
        if not row[index_user] in self.recent_user_sessions:
            self.recent_user_sessions[row[index_user]] = []
            self.recent_user_items[row[index_user]] = []
        if row[index_session] != self.prev_session_id:
            self.recent_user_sessions[row[index_user]].append(row[index_session])
            self.session_items_dict = dict()
            self.session_items_list = []
            self.session_items_list.append(row[index_item])
            self.session_items_dict[row[index_session]] = self.session_items_list
            self.recent_user_items[row[index_user]].append(self.session_items_dict)
            if len(self.recent_user_sessions[row[index_user]]) > self.remind_sessions_num:
                session_id_key = self.recent_user_sessions[row[index_user]][0]
                del self.recent_user_sessions[row[index_user]][0]
                del self.recent_user_items[row[index_user]][0][session_id_key]
                del self.recent_user_items[row[index_user]][0]
            self.prev_session_id = row[index_session]

        else:
            self.session_items_list.append(row[index_item])
        pass

    def reminders_fit(self, train, user_key, item_key, time_key):
        if self.remind_strategy == 'recency' or self.remind_strategy == 'hybrid':
            self.user_item_recency = train.groupby([user_key, item_key], as_index=False).last()
            self.user_item_recency = self.user_item_recency[[user_key, item_key, time_key]]
            self.user_item_recency.sort_values([user_key, time_key], ascending=[True, False],
                                               inplace=True)
            self.user_item_recency = self.user_item_recency.set_index([user_key])

        elif self.remind_strategy == 'session_similarity':
            for u_id in self.recent_user_sessions:
                item_intensity_series = pd.Series()
                for session_item_dic in self.recent_user_items[u_id]:
                    for s_id, i_list in session_item_dic.items():
                        for i_id in i_list:
                            if not i_id in item_intensity_series.index:
                                item_intensity_series.loc[i_id] = 1
                            else:
                                new_count = item_intensity_series.loc[i_id] + 1
                                item_intensity_series.loc[i_id] = new_count

                item_intensity_series.sort_values(ascending=False, inplace=True)
                self.user_item_intensity[u_id] = item_intensity_series

        pass

    def reminders_predict_next(self, input_user_id, series, item_key, time_key, past_user_sessions=None,
                               session_item_map=None, input_timestamp=None):
        reminder_series = pd.Series()

        if self.remind_strategy == 'hybrid':
            reminder_series_SSim = pd.Series()
            reminder_series_IRec = pd.Series()

            if self.w_IRec != 0:
                reminder_series = series.astype('float64')  # float64: writing float64 reminder scores into float32 model scores raises in pandas 3
                reminder_series.loc[:] = 0

                user_item_recency_series = self.user_item_recency.loc[input_user_id]
                if not isinstance(user_item_recency_series, pd.DataFrame):
                    user_item_recency_series = pd.DataFrame({item_key: [user_item_recency_series[item_key].astype(int)],
                                                             time_key: [user_item_recency_series[time_key]]}
                                                            , columns=[item_key, time_key])

                user_item_recency_series = user_item_recency_series.set_index(
                    [item_key])
                user_item_recency_series = user_item_recency_series.iloc[:, 0]
                user_item_recency_series = user_item_recency_series.astype(float)

                for index, value in user_item_recency_series.items():
                    reminder_series.loc[index] = input_timestamp / (input_timestamp - value)
                reminder_series_IRec = reminder_series.copy()

            if self.w_SSim != 0:
                past_user_sessions = sorted(past_user_sessions, reverse=True, key=lambda x: x[1])
                reminder_series = series.astype('float64')  # float64: writing float64 reminder scores into float32 model scores raises in pandas 3
                reminder_series.loc[:] = 0

                for sessions_sim_tuple in past_user_sessions:
                    s_id = sessions_sim_tuple[0]
                    s_score = sessions_sim_tuple[1]
                    sessions_items = session_item_map.get(s_id)
                    for i_id in sessions_items:
                        if not i_id in reminder_series.index:
                            reminder_series.loc[i_id] = s_score
                        else:
                            reminder_series.loc[i_id] = reminder_series.loc[i_id] + s_score

                reminder_series.sort_values(ascending=False, inplace=True)
                reminder_series_SSim = reminder_series.copy()

        if self.remind_strategy == 'recency':
            reminder_series = self.user_item_recency.loc[input_user_id]
            if not isinstance(reminder_series, pd.DataFrame):
                reminder_series = pd.DataFrame({item_key: [reminder_series[item_key].astype(int)],
                                                time_key: [reminder_series[time_key]]}
                                               , columns=[item_key, time_key])

            reminder_series = reminder_series.set_index(
                [item_key])
            reminder_series = reminder_series.iloc[:, 0]
            reminder_series = reminder_series.astype(float)

        if self.remind_strategy == 'session_similarity':
            past_user_sessions = sorted(past_user_sessions, reverse=True, key=lambda x: x[1])
            for sessions_sim_tuple in past_user_sessions:
                s_id = sessions_sim_tuple[0]
                sessions_items = session_item_map.get(s_id)
                for i_id in sessions_items:
                    if not i_id in reminder_series.index:
                        intensity = self.user_item_intensity[input_user_id].loc[i_id]
                        reminder_series.loc[i_id] = intensity

            reminder_series.sort_values(ascending=False, inplace=True)
            reminder_series = reminder_series.astype(float)

        if self.remind_strategy == 'hybrid':

            series_base = series.copy()
            if series.max() != 0:
                series_base = series.copy() / series.max()
            if self.w_IRec != 0 and reminder_series_IRec.max() != 0:
                reminder_series_IRec = reminder_series_IRec / reminder_series_IRec.max()
            if self.w_SSim != 0 and reminder_series_SSim.max() != 0:
                reminder_series_SSim = reminder_series_SSim / reminder_series_SSim.max()

            series = self.w_base * series_base
            if self.w_IRec != 0:
                series = series + (self.w_IRec * reminder_series_IRec)
            if self.w_SSim != 0:
                series = series + (self.w_SSim * reminder_series_SSim)

        elif len(reminder_series) > 0:

            reminder_series = reminder_series.iloc[:self.reminders_num]
            k = self.reminders_num
            if len(reminder_series) < k:
                k = len(reminder_series)

            if k > 0:

                series = series.sort_values(ascending=False).copy()

                if self.remind_mode == 'top':

                    for idx in reminder_series.index:
                        if idx in series[:self.cut_off].index:
                            # the reminder puts it on top, so drop it from the regular list
                            series = series.drop(idx)

                    series = series.iloc[:(self.cut_off - k)]
                    reminder_series = reminder_series.iloc[:k]

                    base_score = series.iloc[0]
                    for index, value in reminder_series.items():
                        reminder_series[index] = base_score + (k * 0.01)
                        k = k - 1

                    series = pd.concat([series, reminder_series])

                elif self.remind_mode == 'end':

                    for idx in reminder_series.index:
                        if idx in series[:self.cut_off].index:
                            # already recommended, no need to remind
                            reminder_series = reminder_series.drop(idx)
                            k = k - 1

                    series = series.iloc[:(self.cut_off - k)]
                    reminder_series = reminder_series.iloc[:k]

                    base_score = series.iloc[(self.cut_off - 1) - k]
                    k = 1
                    for index, value in reminder_series.items():
                        reminder_series[index] = base_score - (k * 0.01)
                        k = k + 1

                    series = pd.concat([series, reminder_series])

        return series


