# uv port status

Python 3.12, numpy 2.5, pandas 3.1, scikit-learn 1.9, PyTorch 2.15 (see `uv.lock`). Run a config with
`uv run session-rec conf/<config>.yml` (`uv sync --extra torch` for the neural models). The reference outputs
(original Python 3.7 / Theano / TF 1.14 / Keras 2.3.1 stack) are in `reference/`.

## How parity is judged
* **Deterministic models**: recommendation lists are compared *score-equivalently* (`reference/parity_lists.py`):
  scores equal within 1e-9 (float summation order differs between library versions), and equal-scored items may be
  ordered differently (sort stability differs). Because metrics are computed from the list order, HR/MRR/NDCG differ
  slightly (<= 0.003 absolute) even when the lists are score-equivalent. Run with `python reference/compare_pm.py`.
* **Neural models**: same config on both stacks with several seeds; the plan criterion is "the old mean lies inside
  the [min, max] of the new runs" (`python reference/compare_neural.py <ref_dir> <port_dir>`). With 3 seeds that range
  is narrow, so borderline models got 5 seeds on both stacks.

## Non-neural models (rsc15-100k, full test set; session-aware on the synthetic rsc15-users data)
| Status | Models |
|---|---|
| Score-equivalent to reference | AR, SR, SKNN, V-SKNN, Pop, SessionPop, RPop, ItemKNN, ContextTree, Weighted hybrid; session-aware: SR, USR, VSKNN, UVSKNN, STAN, USTAN, VSTAN, UVSTAN |
| Runs, not fully verified | Mixed hybrid: 9290 of 10152 lists score-equivalent; the rest differ by a swapped tied item at a slice boundary (plausible, unproven) |

## Neural models
Faithful PyTorch translations of session-rec's own Theano/TF/Keras code (not third-party re-implementations), except
GRU4Rec, which uses the authors' official PyTorch version. Old YAML class paths are mapped to the ports
(`LEGACY_CLASS_MAP` in `src/run_config.py`), so existing configs run unchanged (GRU4Rec/UGRU4Rec excepted).

| Model | Port | Seeds | HR@20 old -> port | MRR@20 old -> port | Verdict |
|---|---|---|---|---|---|
| NARM | `narm_torch` | 3 | 0.6785 -> 0.6782 | 0.3506 -> 0.3512 | match (HR@5, Pop@20 just outside the strict 3-seed range) |
| STAMP | `STAMP_torch` | 3 | 0.6524 -> 0.6538 | 0.3368 -> 0.3377 | match (criterion met) |
| SR-GNN | `sgnn_torch` | 5 | 0.6541 -> 0.6533 | 0.3411 -> 0.3414 | match (criterion met) |
| UNARM | `narm_torch.unarm` | 5 | 0.5809 -> 0.5808 | 0.3040 -> 0.3027 | match (Recall/Precision@20 ~1 sd outside the strict range) |
| NCFS | `NCFS_torch` | 3 | 0.6252 -> 0.6245 | 0.2983 -> 0.2958 | match (criterion met) |
| NSAR | `nsar_torch` | 5 | 0.5453 -> 0.5523 | 0.2321 -> 0.2350 | match (criterion met; high seed variance) |
| HGRU4Rec | `hgru4rec_torch` | 3 | 0.6125 -> 0.6136 | 0.3310 -> 0.3331 | match (Pop@20 just outside the strict range) |
| SHAN | `shan_torch` | 3 | 0.3790 -> 0.3779 | 0.1536 -> 0.1532 | match (criterion met) |
| IIRNN | `IIRNN_torch` | 3 | 0.6733 -> 0.6738 | 0.3437 -> 0.3443 | match (all diffs < 0.002; seed spread is tiny, Pop@20 just outside). Reference from the original with two crash fixes (`reference/aware2/orig_iirnn_fixes.patch`) |
| GRU4Rec | `gru4rec_torch` (official) | 3 | 0.661 -> 0.672 (matched params) | 0.341 -> 0.347 | official implementation; gap is in the old Theano code |
| UGRU4Rec | `gru4rec_torch.ugru4rec` | 3 | 0.6760 -> 0.6787 | 0.3599 -> 0.3513 | different model (official GRU4Rec, no top1-max) |

Session-based models on rsc15-100k, session-aware ones on rsc15-users. Each port's docstring lists the original
behaviours it reproduces (initialisers, non-inverted dropout, TF/Keras GRU variants, loss reductions, optimiser
formulas, batching, and quirks such as SHAN's misaligned pre-sessions or NCFS's session bookkeeping).

**GRU4Rec**: `src/algorithms/gru4rec_torch/` vendors hidasib/GRU4Rec_PyTorch_Official (two documented local changes for
pandas 3 / NumPy 2, license terms in `NOTICE.md`: free for research and education, contact the author for commercial
use). The adapter reproduces the official `batch_eval` exactly on the same model. With matched cross-entropy
parameters the official implementation scores above the old Theano code (outside seed noise), so the old repo's
GRU4Rec numbers are not reproducible with the official code.

## Fixed to run on Python 3.12
`yaml.safe_load`; `time.clock` -> `process_time`; `np.in1d` -> `np.isin`; `np.bool` -> `np.bool_`; `iteritems` -> `items`;
`Series.append` -> `pd.concat`; reminder scores start as float64 (pandas 1.x upcast silently, pandas 3 raises);
optional Telegram import; single-file error path (it hid tracebacks) now reports and exits 1.

## Broken in the original too (signature fixes, no numbers changed)
Pop, SessionPop, RPop, ItemKNN crashed in the original runner (`fit(data)` without `test`, `predict_next` without
`timestamp`, missing `clear()`; RPop used the removed `Series.set_value`). The same fixes were applied to a temporary
copy of the original to obtain a reference (`reference/pm/orig_legacy_signature_fixes.patch`).

## Not ported
* CSRM, NextItNet, AE, APGNN, SMF, RepeatNet, sbr_adapter: not used by the planned experiments (port plan step 4).

## Preprocessing (`uv run --extra hdf session-rec-preprocess <config.yml>`)
* rsc15 (`single` and `window`, on a yoochoose-format file built from the prepared data): every output byte-identical.
* LASTFM 1K (`/opt/datasets/diffrec/lastfm-dataset-1K.tar.gz`): first 200k rows, all four tables identical
  (`reference/compare_hdf.py`); full 19M-row file with `raw_quoting: legacy`: all four tables identical
  (train 4,117,563 / test 7,007 / valid_train 4,110,416 / valid_test 7,147 events), ~2x faster than the original.
* **LASTFM raw parsing**: the original (pandas 1.x default quoting) silently loses or merges 52,006 of the 19,150,868
  plays, because a field starting with `"` swallows the following lines. The port reads every line correctly by
  default (`raw_quoting: none`). `raw_quoting: legacy` reproduces the original row for row (checked against pandas
  1.2.4 on the full file) and is only for comparing with results produced by the original code.
* Not run (no raw data on this machine): RetailRocket, Cosmetics, Xing, Diginetica, Tmall, music/playlist preprocessors.
