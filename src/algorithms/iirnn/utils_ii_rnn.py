import collections
import datetime
import logging
import math
import numpy as np
import os
import pickle

class IIRNNDataHandler:
    
    def __init__(self, train_data, test_data, user_key, item_key, session_key, time_key, batch_size, max_sess_reps, lt_internalsize, timebuckets=[0]):


        self.batch_size = batch_size
        max_sess_len_training = train_data.groupby([user_key, session_key]).size().max()
        max_sess_len_test = test_data.groupby([user_key, session_key]).size().max()
        self.MAX_SESSION_LENGTH = max(max_sess_len_training, max_sess_len_test)
        self.map_user_items(train_data, user_key, item_key)
        self.prepare_data_format(train_data, user_key, item_key, session_key, time_key, 'training')
        self.prepare_data_format(test_data, user_key, item_key, session_key, time_key, 'test')

    
        self.num_users = len(self.trainset)

        self.MAX_SESSION_REPRESENTATIONS = max_sess_reps
        self.LT_INTERNALSIZE = lt_internalsize

    
        self.reset_user_batch_data()


    def map_user_items(self, data, user_key, item_key):
        self.item_map = {}
        self.item_map_reverse = {}
        self.user_map = {}
        self.user_map_reverse = {}

        for index, row in data.iterrows():
            user_id = row[user_key]
            item_id = row[item_key]
            if user_id not in self.user_map:
                map = len(self.user_map)
                self.user_map[user_id] = map
                self.user_map_reverse[map] = user_id
            if item_id not in self.item_map:
                map = len(self.item_map)
                self.item_map[item_id] = map
                self.item_map_reverse[map] = item_id

    def get_user_map(self, user_id):
        return self.user_map[user_id]

    def get_user_map_reverse(self, user_id):
        return self.user_map_reverse[user_id]

    def get_item_map(self, item_id):
        return self.item_map[item_id]

    def get_item_map_reverse(self, item_id):
        return self.item_map_reverse[item_id]

    def get_test_session_lengths(self):
        return self.test_session_lengths

    def get_sess_rep_lengths(self, user_id):
        return max(self.num_user_session_representations[user_id], 1)

    def get_user_session_representations(self, user_id):
        return self.user_session_representations[user_id]

    def get_session_lengths(self, dataset):
        session_lengths = {}
        for k, v in dataset.items():
            session_lengths[k] = []
            for session in v:
                session_lengths[k].append(len(session) - 1)

        return session_lengths

    def create_padded_sequence(self, session):
        if len(session) == self.MAX_SESSION_LENGTH:
            return session

        dummy_timestamp = 0
        dummy_label = 0
        length_to_pad = self.MAX_SESSION_LENGTH - len(session)
        padding = [[dummy_timestamp, dummy_label]] * length_to_pad
        session += padding
        return session

    def pad_sequences(self,dataset):
        for k, v in dataset.items():
            for session_index in range(len(v)):
                dataset[k][session_index] = self.create_padded_sequence(dataset[k][session_index])

    def prepare_data_format(self, data, user_key, item_key, session_key, time_key, mode):
        print("prepare data format "+mode)
        user_sessions = {}
        current_session = []
        prev_sid = -1
        user_id = -1
        for index, row in data.iterrows():
            sid = row[session_key]
            if sid != prev_sid:
                if user_id != -1:
                    user_sessions[user_id].append(current_session)
                current_session = []
                prev_sid = sid

            item_id_map = self.get_item_map(row[item_key])
            new_event = [row[time_key], item_id_map]
            user_id = row[user_key]
            user_id = self.get_user_map(user_id)
            if user_id not in user_sessions:
                user_sessions[user_id] = []


            current_session.append(new_event)

        user_sessions[user_id].append(current_session)

        if(mode == 'training'):
            train_session_lengths = self.get_session_lengths(user_sessions)
            self.train_session_lengths = train_session_lengths
            self.pad_sequences(user_sessions)
            self.trainset = user_sessions

        elif(mode == 'test'):
            test_session_lengths = self.get_session_lengths(user_sessions)
            self.test_session_lengths = test_session_lengths
            self.pad_sequences(user_sessions)
            self.testset = user_sessions


    def reset_user_batch_data(self):
        self.user_next_session_to_retrieve = [0]*self.num_users
        self.users_with_remaining_sessions = []
        self.num_remaining_sessions_for_user = [0]*self.num_users
        for k, v in self.trainset.items():
            # everyone has at least one session
            self.users_with_remaining_sessions.append(k)

    def reset_user_session_representations(self):
        istate = np.zeros([self.LT_INTERNALSIZE])

        self.user_session_representations = [None]*self.num_users
        self.num_user_session_representations = [0]*self.num_users
        for k, v in self.trainset.items():
            self.user_session_representations[k] = collections.deque(maxlen=self.MAX_SESSION_REPRESENTATIONS)
            for i in range(self.MAX_SESSION_REPRESENTATIONS):
                self.user_session_representations[k].append(istate)

    def get_N_highest_indexes(a,N):
        return np.argsort(a)[::-1][:N]

    def add_unique_items_to_dict(self, items, dataset):
        for k, v in dataset.items():
            for session in v:
                for event in session:
                    item = event[1]
                    if item not in items:
                        items[item] = True
        return items

    def get_num_items(self):
        items = {}
        items = self.add_unique_items_to_dict(items, self.trainset)
        items = self.add_unique_items_to_dict(items, self.testset)
        return len(items)

    def get_num_sessions(self, dataset):
        session_count = 0
        for k, v in dataset.items():
            session_count += len(v)
        return session_count

    def get_num_training_sessions(self):
        return self.get_num_sessions(self.trainset)
    
    # for the II-RNN this is only an estimate
    def get_num_batches(self, dataset):
        num_sessions = self.get_num_sessions(dataset)
        return math.ceil(num_sessions/self.batch_size)

    def get_num_training_batches(self):
        return self.get_num_batches(self.trainset)

    def get_num_test_batches(self):
        return self.get_num_batches(self.testset)

    def get_next_batch(self, dataset, dataset_session_lengths):

        session_batch = []
        session_lengths = []
        sess_rep_batch = []
        sess_rep_lengths = []
        
        remaining_sessions = [0]*len(self.users_with_remaining_sessions)
        for i in range(len(self.users_with_remaining_sessions)):
            user = self.users_with_remaining_sessions[i]
            remaining_sessions[i] = len(dataset[user]) - self.user_next_session_to_retrieve[user]
        
        user_list = IIRNNDataHandler.get_N_highest_indexes(remaining_sessions, self.batch_size)
        for i in range(len(user_list)):
            user_list[i] = self.users_with_remaining_sessions[user_list[i]]

        for user in user_list:
            session_index = self.user_next_session_to_retrieve[user]
            session_batch.append(dataset[user][session_index])
            session_lengths.append(dataset_session_lengths[user][session_index])
            srl = max(self.num_user_session_representations[user], 1)
            sess_rep_lengths.append(srl)
            sess_rep_batch.append(self.user_session_representations[user])

            self.user_next_session_to_retrieve[user] += 1
            if self.user_next_session_to_retrieve[user] >= len(dataset[user]):
                self.users_with_remaining_sessions.remove(user)

        session_batch = [[event[1] for event in session] for session in session_batch]
        x = [session[:-1] for session in session_batch]
        y = [session[1:] for session in session_batch]

        return x, y, session_lengths, sess_rep_batch, sess_rep_lengths, user_list

    def get_next_train_batch(self):
        return self.get_next_batch(self.trainset, self.train_session_lengths)

    def get_next_test_batch(self):
        return self.get_next_batch(self.testset, self.test_session_lengths)

    def get_latest_epoch(self, epoch_file):
        if not os.path.isfile(epoch_file):
            return 0
        return pickle.load(open(epoch_file, 'rb'))
    
    def store_current_epoch(self, epoch, epoch_file):
        pickle.dump(epoch, open(epoch_file, 'wb'))

    
    def add_timestamp_to_message(self, message):
        timestamp = str(datetime.datetime.now())
        message = timestamp+'\n'+message
        return message

    def log_test_stats(self, epoch_number, epoch_loss, stats):
        timestamp = str(datetime.datetime.now())
        message = timestamp+'\n\tEpoch #: '+str(epoch_number)
        message += '\n\tEpoch loss: '+str(epoch_loss)+'\n'
        message += stats
        logging.info(message)

    def log_config(self, config):
        config = self.add_timestamp_to_message(config)
        logging.info(config)

    
    def store_user_session_representations(self, sessions_representations, user_list):
        for i in range(len(user_list)):
            user = user_list[i]
            session_representation = sessions_representations[i]

            num_reps = self.num_user_session_representations[user]
            self.user_session_representations[user].append(session_representation)

            self.num_user_session_representations[user] = self.MAX_SESSION_REPRESENTATIONS

