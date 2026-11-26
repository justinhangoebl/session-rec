"""PyTorch port of session-rec's II-RNN (TensorFlow 1), kept as close to the original as possible.

Ruocco et al., Inter-Session Modeling for Session-Based Recommendation, DLRS 2017; session-rec's version.
The data handler (utils_ii_rnn.py) keeps the original logic unchanged. Reproduced exactly (they change results):
  * item embedding U(-1, 1); session representation = mean of the embedded *padded* sequence / true length
    (padding uses item index 0, which is a real item, as in the original); use_last_hidden_state=True uses H
  * inter-session GRU: TF1 GRUCell (gate bias 1.0, glorot kernels) in a DropoutWrapper (input and output keep
    probability pkeep, inverted dropout), dynamic_rnn over the stored representations; its last output (with
    output dropout) is the initial state of the intra-session GRU (input dropout inside, output dropout outside)
  * output layer: tf.contrib.layers.linear (glorot-uniform weights, zero bias)
  * loss: SUM over positions of the per-position cross-entropy masked by sign(target) (so the real item with
    index 0 is masked too), Adam (TF1 formulation)
  * training stops an epoch when a batch has <= batch_size/2 users, exactly as the original
  * prediction: rank scores TOP_K, TOP_K-1, ... from the top-k of the last position's logits
Prediction-time difference (see squeeze_session_rep): the original stores the (1, d) representation of a finished
test session in a deque of (d,) vectors; feeding that ragged deque to TensorFlow fails on the user's next session.
With squeeze_session_rep=True (default) the (1, d) row is stored as (d,), which is what the code intends.
The original also crashes in evaluation (iterrows() turns item ids into floats, so the prediction Series gets a
float index and pandas 1.x slices it by label); this port uses int item ids.
Extra parameters: seed (numpy, torch), device, squeeze_session_rep.
"""
import time

import numpy as np
import pandas as pd
import torch

from algorithms.iirnn.utils_ii_rnn import IIRNNDataHandler
from algorithms.nsar.nsar import tf_default_init
from algorithms.stamp.stamp import TFAdam
from evaluation import tracking


class TFGRUCell(torch.nn.Module):
    def __init__(self, input_size, H):
        super().__init__()
        self.H = H
        self.gates_kernel = tf_default_init(input_size + H, 2 * H); self.gates_bias = torch.nn.Parameter(torch.ones(2 * H))
        self.cand_kernel = tf_default_init(input_size + H, H); self.cand_bias = torch.nn.Parameter(torch.zeros(H))

    def forward(self, x, h):
        value = torch.sigmoid(torch.cat([x, h], 1) @ self.gates_kernel + self.gates_bias)
        r, u = value[:, :self.H], value[:, self.H:]
        c = torch.tanh(torch.cat([x, r * h], 1) @ self.cand_kernel + self.cand_bias)
        return u * h + (1 - u) * c


class IIRNNModel(torch.nn.Module):
    def __init__(self, n_items, E, LT, ST):
        super().__init__()
        self.W_embed = torch.nn.Parameter(torch.rand(n_items, E) * 2 - 1)
        self.lt_cell = TFGRUCell(LT, LT)
        self.st_cell = TFGRUCell(E, ST)
        self.W_out = tf_default_init(ST, n_items); self.b_out = torch.nn.Parameter(torch.zeros(n_items))
        self.ST = ST

    @staticmethod
    def drop(x, keep, training):
        return torch.nn.functional.dropout(x, p=1 - keep, training=True) if (training and keep < 1) else x

    def forward(self, X, seq_len, X_lt, seq_len_lt, pkeep, training):
        B = X.shape[0]
        X_embed = self.W_embed[X]
        X_avg = X_embed.sum(1) / seq_len[:, None].float()
        h = torch.zeros(B, X_lt.shape[-1], device=X.device); lt_out = []
        for s in range(X_lt.shape[1]):
            hn = self.lt_cell(self.drop(X_lt[:, s], pkeep, training), h)
            m = (s < seq_len_lt)[:, None]
            h = torch.where(m, hn, h)
            lt_out.append(torch.where(m, self.drop(hn, pkeep, training), torch.zeros_like(hn)))
        lt_out = torch.stack(lt_out, 1)
        h = lt_out[torch.arange(B, device=X.device), seq_len_lt - 1]
        outs = []
        for t in range(X.shape[1]):
            hn = self.st_cell(self.drop(X_embed[:, t], pkeep, training), h)
            m = (t < seq_len)[:, None]
            h = torch.where(m, hn, h)
            outs.append(torch.where(m, self.drop(hn, pkeep, training), torch.zeros_like(hn)))
        Yr = torch.stack(outs, 1)
        logits = Yr.reshape(-1, self.ST) @ self.W_out + self.b_out
        return logits, h, X_avg


class IIRNN:
    def __init__(self, embedding_size=100, max_epoch=20, batch_size=100, max_session_representation=15, learning_rate=0.05,
                 dropout_pkeep=0.0, use_last_hidden_state=True, do_training=True, save_best=True, session_key='SessionId',
                 item_key='ItemId', time_key='Time', user_key='UserId', seed=None, device=None, squeeze_session_rep=True):
        self.EMBEDDING_SIZE = self.ST_INTERNALSIZE = self.LT_INTERNALSIZE = embedding_size
        self.MAX_EPOCHS, self.BATCHSIZE, self.MAX_SESSION_REPRESENTATIONS = max_epoch, batch_size, max_session_representation
        self.learning_rate, self.dropout_pkeep, self.use_last_hidden_state = learning_rate, dropout_pkeep, use_last_hidden_state
        self.do_training = do_training
        self.session_key, self.item_key, self.time_key, self.user_key = session_key, item_key, time_key, user_key
        self.session, self.session_items = -1, []
        self.session_representation = None
        self.user_sess_representation = None
        self.squeeze_session_rep = squeeze_session_rep
        self.seed = seed
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def _run(self, xinput, sl, session_reps, sr_sl, pkeep, training):
        dev = self.device
        X = torch.tensor(np.asarray(xinput), dtype=torch.long, device=dev)
        X_lt = torch.tensor(np.asarray([np.asarray(list(r), dtype=np.float32) for r in session_reps]), device=dev)
        return self.model(X, torch.tensor(sl, device=dev), X_lt, torch.tensor(sr_sl, device=dev), pkeep, training)

    def fit(self, train_data, test_data, valid_data=None, **kwargs):
        if self.seed is not None:
            np.random.seed(self.seed); torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        self.TOP_K = len(train_data[self.item_key].unique())
        self.datahandler = IIRNNDataHandler(train_data, test_data, self.user_key, self.item_key, self.session_key, self.time_key,
                                            self.BATCHSIZE, self.MAX_SESSION_REPRESENTATIONS, self.LT_INTERNALSIZE)
        self.N_ITEMS = self.datahandler.get_num_items()
        self.model = IIRNNModel(self.N_ITEMS, self.EMBEDDING_SIZE, self.LT_INTERNALSIZE, self.ST_INTERNALSIZE).to(self.device)
        tracking.watch(self.model)
        opt = TFAdam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(1, self.MAX_EPOCHS + 1):
            start = time.time(); epoch_loss = 0
            self.datahandler.reset_user_batch_data()
            self.datahandler.reset_user_session_representations()
            if self.do_training:
                xinput, targetvalues, sl, session_reps, sr_sl, user_list = self.datahandler.get_next_train_batch()
                while len(xinput) > int(self.BATCHSIZE / 2):
                    logits, H, X_avg = self._run(xinput, sl, session_reps, sr_sl, self.dropout_pkeep, True)
                    target = torch.tensor(np.asarray(targetvalues), dtype=torch.long, device=self.device).reshape(-1)
                    loss = (torch.nn.functional.cross_entropy(logits, target, reduction='none') * torch.sign(target.float())).sum()
                    opt.zero_grad(); loss.backward(); opt.step()
                    epoch_loss += loss.item() / len(xinput) / logits.shape[0] * len(xinput)
                    sess_rep = (H if self.use_last_hidden_state else X_avg).detach().cpu().numpy()
                    self.datahandler.store_user_session_representations(sess_rep, user_list)
                    xinput, targetvalues, sl, session_reps, sr_sl, user_list = self.datahandler.get_next_train_batch()
            print('Epoch', epoch, 'loss', epoch_loss, 'time %.1fs' % (time.time() - start))
            tracking.log({'train/epoch': epoch, 'train/loss': float(epoch_loss), 'train/epoch_time_s': time.time() - start})
        self.model.eval()

    @torch.no_grad()
    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids=None, skip=False, mode_type='view', timestamp=0):
        user_id = self.datahandler.get_user_map(input_user_id)
        if self.session != session_id:
            self.session = session_id; self.session_items = list()
            if self.user_sess_representation is not None:
                rep = self.session_representation[0] if self.squeeze_session_rep else self.session_representation
                self.datahandler.store_user_session_representations([rep], [self.user_sess_representation])
            self.user_sess_representation = user_id
        self.session_items.append(self.datahandler.get_item_map(input_item_id))
        sess_rep = self.datahandler.get_user_session_representations(user_id)
        sr_sl = self.datahandler.get_sess_rep_lengths(user_id)
        logits, H, X_avg = self._run([self.session_items], [len(self.session_items)], [sess_rep], [sr_sl], 1.0, False)
        self.session_representation = (H if self.use_last_hidden_state else X_avg).cpu().numpy()
        last = logits.reshape(1, -1, self.N_ITEMS)[0, len(self.session_items) - 1]
        top = torch.topk(last, k=self.TOP_K).indices.cpu().numpy()
        pred_list = [int(self.datahandler.get_item_map_reverse(p)) for p in top]   # int ids: float ids crash the original IIRNN
        return pd.Series(np.arange(self.TOP_K, 0, -1), index=pred_list)

    def clear(self):
        self.model = None

    def support_users(self):
        return True

    def predict_with_training_data(self):
        return False
