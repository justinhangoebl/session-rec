# Adding a model

A model is a Python class below `src/algorithms/`. Configs refer to it by its class path relative to that folder,
so there is no registry to update: add the file, reference it in a config, run it.

## 1. Where the code goes

```
src/algorithms/successor/__init__.py      # empty, makes the folder a package
src/algorithms/successor/successor.py     # class Successor
```

Config entry: `class: successor.successor.Successor` (package.module.Class).

## 2. The interface

The runner (`src/run_config.py`) and the evaluation loops (`src/evaluation/`) call these methods:

| Method | Required | Called |
|---|---|---|
| `__init__(**params)` | yes | once per config entry, with the YAML `params` as keyword arguments |
| `fit(train, test=None)` | yes | once per split / slice, before evaluation |
| `predict_next(...)` | yes | once per evaluated event (signatures below) |
| `support_users()` | for `evaluation_user_based` | `True` if the model uses user ids |
| `predict_with_training_data()` | if `support_users()` is `True` | `True` replays the test users' training sessions through `predict_next` before their test session (scores ignored), e.g. to warm up a user state |
| `clear()` | yes | after the evaluation, to free memory |
| `init(train, test, slice=None)` | optional | before `fit` |
| `predict_for_extended_model(user_id)` | optional | session-aware only: at the start of a test session; the items it returns are fed through `predict_next` first (scores ignored) |

**Data.** `train` and `test` are pandas DataFrames with the columns `SessionId`, `ItemId`, `Time` (unix seconds) and,
for session-aware data, `UserId`. `test` is passed to `fit` only so a model can see which ids exist; it must not
learn from the test interactions.

**`predict_next`** returns a `pandas.Series` of scores indexed by item id, covering `predict_for_item_ids` (all
training items). Higher is better; the evaluation sets NaN to 0 and sorts. The arguments depend on the evaluation:

* `evaluation` (session-based): `predict_next(session_id, input_item_id, predict_for_item_ids, timestamp=...)`.
  Called from the second event of a session on, with the *previous* item; the target is the current one. A new
  `session_id` means a new session, so reset any per-session state when it changes.
* `evaluation_user_based` (session-aware): if `support_users()` is `True`,
  `predict_next(session_id, input_item_id, input_user_id, predict_for_item_ids, timestamp=...)`, otherwise the
  session-based signature. Called for every event of a test session except the last; the targets are the
  following item (HR, MRR) or all remaining items of the session (`accuracy_multiple` metrics).

## 3. Example

A complete model that works in both evaluation modes. It scores items by how often they directly followed the
current item in the training sessions; with `user_boost > 0` it also boosts the items the user played before.

```python
import pandas as pd


class Successor:
    '''
    Scores every item by how often it directly followed the current item in the training sessions.
    With user_boost > 0 it is session-aware: items the user played in training sessions get + user_boost.
    '''

    def __init__(self, user_boost=0.0, session_key='SessionId', item_key='ItemId', time_key='Time', user_key='UserId'):
        self.user_boost = user_boost
        self.session_key, self.item_key, self.time_key, self.user_key = session_key, item_key, time_key, user_key

    def fit(self, train, test=None):
        train = train.sort_values([self.session_key, self.time_key])
        items = train[self.item_key].values
        same_session = train[self.session_key].values[1:] == train[self.session_key].values[:-1]
        pairs = pd.DataFrame({'prev': items[:-1][same_session], 'next': items[1:][same_session]})
        self.successors = pairs.groupby(['prev', 'next']).size()
        if self.user_boost > 0:
            self.user_items = train.groupby(self.user_key)[self.item_key].unique()

    def predict_next(self, session_id, input_item_id, *args, timestamp=None):
        # session-based: (session_id, item, items_to_predict); session-aware: (session_id, item, user_id, items_to_predict)
        if self.support_users():
            input_user_id, predict_for_item_ids = args
        else:
            (predict_for_item_ids,) = args
        scores = pd.Series(0.0, index=predict_for_item_ids)
        if input_item_id in self.successors.index:
            counts = self.successors.loc[input_item_id]
            scores[counts.index] = counts.values
        if self.support_users() and input_user_id in self.user_items.index:
            scores[self.user_items[input_user_id]] += self.user_boost
        return scores

    def support_users(self):
        return self.user_boost > 0

    def predict_with_training_data(self):
        return False

    def clear(self):
        self.successors = None
```

Config (add it to the `algorithms` list of any config, e.g. a copy of `conf/rsc15_quick.yml`):

```yaml
algorithms:
- class: successor.successor.Successor
  key: successor
- class: successor.successor.Successor
  params: { user_boost: 0.5 }      # session-aware variant; needs evaluation: evaluation_user_based and user data
  key: successor_user
```

```bash
uv run session-rec conf/my_config.yml
```

The result key is the `key` plus every parameter (`successor_user-user_boost=0.5`), so several parameter settings
of one model can run in the same config.

## 4. Neural models

* Follow the conventions of the existing ports: a `seed` parameter that seeds numpy, `random` and torch in `fit`,
  and a `device` parameter (`'cuda:0'` / `'cpu'`).
* `predict_next` is called once per event, so keep per-session state (a hidden state, the session's items) and
  update it incrementally instead of re-encoding the whole session; see `gru4rec/adapter.py`. Use
  `torch.no_grad()` and return the scores as a CPU `pandas.Series`.
* Map item ids to embedding rows in `fit` (`pd.factorize` or a dict) and keep the reverse index for the Series.
* New dependencies: `uv add <package>` for the core, `uv add --optional torch <package>` for neural-only ones.

## 5. Tuning and checking

* Tuning: give the model's entry a `params_opt` instead of (or in addition to) `params` and use `type: opt` or
  `type: bayopt`; see `conf/tuning_example.yml`. Tuning runs on the validation split (`train_tr` / `train_valid`).
* First run: `conf/rsc15_quick.yml` with the new model added (bundled data, about a minute).
* Session-aware results: `conf/lastfm_baselines.yml` and `conf/lastfm_neural.yml` give the baselines on the same
  slices; neural models should be reported over several seeds.
