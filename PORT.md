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

* **Reference lists are not in git.** The 106 `*-Saver@50.csv` top-50 recommendation lists under `reference/`
  (~1.2 GB) are kept out of the repository (`.gitignore`) and published as the release archive
  `session-rec-reference-lists.tar.gz` with a `.sha256` file. Extract it in the repository root
  (`tar -xzf session-rec-reference-lists.tar.gz && sha256sum -c session-rec-reference-lists.sha256`); the
  metric CSVs next to the lists are in git.

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
* LASTFM `window` (5 slices, Latifi et al. settings, `raw_quoting: legacy`): all 20 tables identical to the original.
* Not run (no raw data on this machine): RetailRocket, Cosmetics, Xing, Diginetica, Tmall, music/playlist preprocessors.

## Real data: Latifi et al. 2021, LASTFM (Table 6)
Session-aware LASTFM benchmark of Latifi, Mauro, Jannach, "Session-aware recommendation: a surprising quest for the
state of the art" (Information Sciences 2021, arXiv 2011.03424). Window preprocessing (5 slices), `window` runs with
`evaluation_user_based`, metrics averaged over the slices as in the paper. Scripts and configs: `reference/latifi/`,
`conf/latifi_lastfm/`. Compare with `python reference/latifi/compare.py` and `reference/latifi/compare_neural.py`.

**Non-neural (original and port, deterministic).** Every row reproduces the paper within 0.0004 on all seven
metrics (MAP, P, R, HR, MRR, COV, POP @20), and the port's lists are score-equivalent to the original's on every slice.

| Row | MAP@20 paper -> port | HR@20 paper -> port | MRR@20 paper -> port |
|---|---|---|---|
| vsknn_eb | 0.0493 -> 0.0493 | 0.5383 -> 0.5381 | 0.1771 -> 0.1770 |
| vsknn | 0.0474 -> 0.0474 | 0.5227 -> 0.5223 | 0.1725 -> 0.1724 |
| vstan_eb (`lambda_spw` 1.4525, see below) | 0.0449 -> 0.0449 | 0.5291 -> 0.5291 | 0.1938 -> 0.1938 |
| stan_ebr | 0.0442 -> 0.0442 | 0.5159 -> 0.5160 | 0.1902 -> 0.1901 |
| stan | 0.0400 -> 0.0400 | 0.5012 -> 0.5012 | 0.1886 -> 0.1883 |
| vstan | 0.0394 -> 0.0394 | 0.5100 -> 0.5100 | 0.1935 -> 0.1933 |
| sr_b | 0.0359 -> 0.0359 | 0.4622 -> 0.4621 | 0.3487 -> 0.3486 |
| sr | 0.0348 -> 0.0347 | 0.4615 -> 0.4614 | 0.3350 -> 0.3351 |

* **vstan_eb**: with the parameters printed in the paper's Table A.14 (`lambda_spw` 0.4525) both the original and the
  port give MAP 0.0437 / HR 0.5204 instead of 0.0449 / 0.5291. With `lambda_spw` 1.4525 (an off-grid value that
  appears in the repo's own `window_lastfm_vstan_EBR.yml`) both reproduce Table 6 exactly, so the published run used
  that value and Table A.14 shows 0.4525.
* vstan: the paper's `lambda_ipw` 0.0001 and the repo's 0.00001 give the same results.
* VSTAN / U_VSKNN_STAN accept `similarity` but always use cosine (`calc_similarity`), in the original too.

**Neural (port only, 3 seeds; the paper reports one run).** Mean of the seeds; each seed is the mean of the 5 slices.

| Model | MAP@20 paper -> port | HR@20 paper -> port | MRR@20 paper -> port | Verdict |
|---|---|---|---|---|
| IIRNN | 0.0311 -> 0.0311 | 0.4729 -> 0.4695 | 0.3491 -> 0.3489 | match; HR 0.7% low, paper just above the best seed |
| GRU4Rec (official PyTorch) | 0.0307 -> 0.0305 | 0.4681 -> 0.4678 | 0.3342 -> 0.3346 | match; all accuracy metrics inside the seed range |
| NSAR | 0.0280 -> 0.0279 | 0.4350 -> 0.4338 | 0.2906 -> 0.2906 | match |
| NARM | 0.0272 -> 0.0270 | 0.4510 -> 0.4534 | 0.3192 -> 0.3168 | match within 0.003; MRR consistently 0.8% low |
| NCSF | 0.0219 -> 0.0221 | 0.4393 -> 0.4410 | 0.2434 -> 0.2428 | match |
| HGRU4Rec | 0.0208 -> 0.0210 | 0.4206 -> 0.4210 | 0.3167 -> 0.3171 | match |
| SHAN | 0.0072 -> 0.0073 | 0.1149 -> 0.1164 | 0.0345 -> 0.0355 | match within 0.0015; slightly above the paper on all accuracy metrics |

* GRU4Rec: the official PyTorch implementation reproduces the old Theano numbers here (paper settings: bpr-max,
  lr 0.04, momentum 0.1; old Theano defaults for the rest). The gap seen on rsc15 is therefore specific to that setting.
* SHAN: the port now looks up embedding rows with sparse gradients, so plain SGD updates only the rows a record
  touches (as the original TF1 IndexedSlices did). Same losses and parameters as the dense version up to float
  rounding (max diff 5e-6), ~13x faster: about 2.5 h per run on LASTFM instead of ~35 h.
* NCSF needed a port fix for `max_nb_his_sess: 0` (no history branch, as in the original).
* Dataset statistics differ from the paper's Table 2 (e.g. 45,854 vs 107,134 items per slice) although every model
  reproduces, so Table 2 was presumably counted before the item and session filters.
