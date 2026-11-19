# uv port status

Python 3.12, numpy 2.5, pandas 3.1, scikit-learn 1.9 (see `uv.lock`). Run a config with
`uv run session-rec conf/<config>.yml`. The reference outputs (original Python 3.7 stack) are in `reference/`.

## How parity is judged
Recommendation lists are compared *score-equivalently* (`reference/parity_lists.py`): scores equal within 1e-9
(float summation order differs between library versions), and equal-scored items may be ordered differently
(sort stability differs). Because metrics are computed from the list order, HR/MRR/NDCG differ slightly (<= 0.003
absolute here) even when the lists are score-equivalent. Run everything with `python reference/compare_pm.py`.

## Non-neural models (rsc15-100k, full test set; session-aware on the synthetic rsc15-users data)
| Status | Models |
|---|---|
| Score-equivalent to reference | AR, SR, SKNN, V-SKNN, Pop, SessionPop, RPop, ItemKNN, ContextTree, Weighted hybrid; session-aware: SR, USR, VSKNN, UVSKNN, STAN, USTAN, VSTAN, UVSTAN |
| Runs, not fully verified | Mixed hybrid: 9290 of 10152 lists score-equivalent; the rest differ by a swapped tied item at a slice boundary (plausible, unproven) |

## Fixed to run on Python 3.12
`yaml.safe_load`; `time.clock` -> `process_time`; `np.in1d` -> `np.isin`; `np.bool` -> `np.bool_`; `iteritems` -> `items`;
`Series.append` -> `pd.concat`; optional Telegram import; single-file error path (it hid tracebacks) now reports and exits 1.

## Broken in the original too (signature fixes, no numbers changed)
Pop, SessionPop, RPop, ItemKNN crashed in the original runner (`fit(data)` without `test`, `predict_next` without
`timestamp`, missing `clear()`; RPop used the removed `Series.set_value`). The same fixes were applied to a temporary
copy of the original to obtain a reference (`reference/pm/orig_legacy_signature_fixes.patch`).

## Not ported yet
* Neural models (GRU4Rec, NARM, STAMP, SR-GNN, UGRU4Rec, UNARM): Theano/TF/Keras; references exist (`reference/neural`,
  `reference/aware/neural`), PyTorch replacements still to write (plan step 4).
* Preprocessing (`src/preprocessing`, `run_preprocessing.py`): imports fine, never run (no raw datasets in the repo).
* HGRU4Rec, IIRNN, SHAN, NSAR, NCFS, CSRM, NextItNet, AE, APGNN, SMF.
