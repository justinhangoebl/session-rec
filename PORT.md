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

## Neural: GRU4Rec (PyTorch, official implementation) -- ported, parity NOT met
`src/algorithms/gru4rec_torch/`: the official hidasib/GRU4Rec_PyTorch_Official file (two documented local changes, see
`NOTICE.md`, license terms there) plus `adapter.py` (per-event `predict_next` with per-session hidden state).
* The adapter is faithful: on the same trained model its `predict_next` gives exactly the same HR/MRR as the official
  `batch_eval` (`reference/neural_gru4rec_torch/adapter_check.py`).
* Matched comparison (same cross-entropy params in both, 3 seeds, rsc15-100k): old Theano HR@20 0.661 +-0.001,
  MRR@20 0.341 +-0.001; PyTorch HR@20 0.672 +-0.003, MRR@20 0.347 +-0.002. The gap (+0.011 / +0.006) is larger than the
  seed spread, so the plan's "within seed noise" criterion is **not met**. Cause not yet identified; it is a difference
  between the official PyTorch implementation and the older Theano code, not the adapter.
* With the authors' best rsc15 parameters (logq=1): HR@20 0.677, MRR@20 0.355 (vs the old top1-max reference 0.628 / 0.331).
  Not a like-for-like comparison: the PyTorch version has no top1-max loss.

## Not ported yet
* Neural models NARM, STAMP, SR-GNN, UGRU4Rec, UNARM: Theano/TF/Keras; references exist (`reference/neural`,
  `reference/aware/neural`), PyTorch replacements still to write (plan step 4).
* Preprocessing (`src/preprocessing`, `run_preprocessing.py`): imports fine, never run (no raw datasets in the repo).
* HGRU4Rec, IIRNN, SHAN, NSAR, NCFS, CSRM, NextItNet, AE, APGNN, SMF.
