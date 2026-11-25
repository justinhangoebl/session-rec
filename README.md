# session-rec

Evaluation framework and baselines for session-based and session-aware recommendation. A port of
[rn5l/session-rec](https://github.com/rn5l/session-rec) (Ludewig, Mauro, Latifi, Jannach) to Python 3.12, current
NumPy / pandas / scikit-learn, and PyTorch for every neural model, managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync --extra torch --extra hdf    # torch: neural models; hdf: hdf data (data.type: hdf) and LASTFM preprocessing
wandb login                          # once; runs are logged to Weights & Biases (see below)
```

## Data

`data/` is not versioned. Layout:

| Folder | Content | How to get it |
|---|---|---|
| `data/rsc15/raw/` | `yoochoose-clicks.dat` (RecSys Challenge 2015, 33M clicks) | `curl -LO https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z && 7za e yoochoose-data.7z yoochoose-clicks.dat` |
| `data/rsc15/slices/` | 5 time slices (30 days train / 1 day test) | `uv run session-rec-preprocess conf/preprocess/rsc15_window.yml` (~3 min) |
| `data/rsc15/prepared/` | single split (last day = test) and the bundled 100k-click sample `yoochoose-clicks-100k` | `uv run session-rec-preprocess conf/preprocess/rsc15.yml` |
| `data/lastfm/raw/` | `userid-timestamp-artid-artname-traid-traname.tsv` from [lastfm-dataset-1K](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html) | download and extract the archive |
| `data/lastfm/prepared/` | 5 session-aware slices (hdf) | `uv run session-rec-preprocess conf/preprocess/lastfm.yml` |
| `data/lastfm/temporal/` | one global temporal split T0 / T1 / T2 (hdf keys `train` / `valid_test` / `test`), 180 days up to 2009-05-01 | `uv run session-rec-preprocess conf/preprocess/lastfm_temporal.yml` (~1 min) |
| `data/yambda/raw/` | Yambda-50M `listens.parquet` (+ likes / dislikes), [yandex/yambda](https://huggingface.co/datasets/yandex/yambda) `flat/50m/` | `curl -LO https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/50m/listens.parquet` |
| `data/yambda/temporal/` | global temporal split backward from the end with 30-min gaps: T0 train (~287 d) / T1a 3 d (`valid_test`) / T1b 4 d (`calib`) / T2 last 7 d (`test`) | `uv run session-rec-preprocess conf/preprocess/yambda50m.yml` (~3 min) |

## Run

```bash
uv run session-rec conf/rsc15_quick.yml           # one config; a folder runs every config in it
uv run session-rec-preprocess conf/preprocess/lastfm.yml
```

Results are written to `results.folder` of the config as `test_<type>_<key>_<data.name>[.<slice>].csv`.

**Saved models.** Every fitted algorithm is saved after `fit` (on every slice) to its own folder
`saved/<algorithm key>-<data.name>-<YYYYMMDD-HHMM>-<config key>[-s<slice>]/`, e.g.
`saved/gru4rec-yambda50m-20261125-0930-backbones/`:

* `model.pkl`: the whole fitted object (dill); `dill.load(f).predict_next(...)` or `filemodel.FileModel`.
* `model.pt` (models with a torch module or `export()`): `torch.load(path, weights_only=False)` gives `meta`
  (config, class, params, slice), `state_dict` and `export`. For SASRec, GRU4Rec and NARM, `export` holds `item_ids`
  and `item_embeddings`; row i of the embedding matrix belongs to `item_ids[i]`. Uploaded to wandb as a model
  artifact (files over 2 GB stay local).
* `meta.json`: the same meta, readable without torch.
* `candidates/<part>/`: candidate pools, see below.

EASE writes no `.pkl` (`export` = `item_ids`, the dense `B`, user histories); `EASE.load(path)` rebuilds it, and
`evaluation.model_store.load(folder)` loads any saved model. `results.save_models: false` turns saving off;
`results.save_folder: <folder>` replaces `saved/`.

**Candidate pools** (`src/evaluation/candidates.py`, for the calibration / Sinkhorn stage). With a `candidates`
section the frozen model scores every session prefix of the listed data parts after `fit` (batched on the GPU) and
writes per part `requests.parquet` (session, user, time, position, target, target rank / logit, logsumexp of all
logits, the target's extra columns such as `is_organic`), `items.npy` and `scores.npy` (`[requests, k]` top-k item
ids and logits) and `metrics.json` (next-item HitRate / MRR / NDCG @ 1-20, coverage). `evaluation: none` skips the
per-event session-rec evaluation and reports these metrics instead. For models saved earlier:
`uv run session-rec-candidates saved/<folder> [--parts valid_test test] [--k 200]`.

```yaml
candidates:
  parts: [valid_test, calib, test]   # hdf keys, in time order
  k: 200
  history: rolling                   # EASE's user history: all earlier listens (rolling) or the training data only
```

`--only <key> ...` runs a subset of a config's algorithms, e.g. one process per GPU:
`CUDA_VISIBLE_DEVICES=2 uv run session-rec conf/yambda50m_temporal.yml --only sasrec`.

**Weights & Biases.** Every algorithm on every split / slice is one wandb run (project `session-rec` unless the
config's `wandb` section says otherwise), grouped by `<config key>/<data>/<algorithm>` so the slices of a window
config average in the UI. A run holds the experiment config, the algorithm params, the data statistics, the
training curves (`train/*`, per epoch, for the neural models), gradient / parameter histograms (`wandb.watch`) and
the evaluation metrics (`test/*`, `valid/*` for tuning). Config section (all keys optional):

```yaml
wandb:
  project: session-rec
  entity: null                # team or user
  mode: online                # online | offline | disabled
  tags: [rsc15]
  watch: {log: gradients, log_freq: 500}    # false: no wandb.watch
```

`wandb: false` in a config or `WANDB_MODE=disabled` in the environment turns logging off. See
`src/evaluation/tracking.py`; models log through `tracking.log(...)` / `tracking.watch(model)`.

Neural models take a `device` param (`'cuda:0'`, `'cuda:1'`, `'cpu'`); the configs use `cuda:0`.

| Config | What it does |
|---|---|
| `conf/rsc15_quick.yml` | smoke test on the bundled rsc15 sample (`data/rsc15/prepared/`) |
| `conf/rsc15_neural.yml` | EASE, SASRec, GRU4Rec, NARM on RSC15 |
| `conf/rsc15_baselines.yml` | non-neural baselines on RSC15 (params of Ludewig et al. 2021) |
| `conf/lastfm_baselines.yml` | session-aware non-neural baselines on LASTFM with the tuned parameters of Latifi et al. 2021 |
| `conf/lastfm_neural.yml` | the neural models on LASTFM with the tuned parameters of Latifi et al. 2021 |
| `conf/lastfm_temporal.yml` | EASE, SASRec, GRU4Rec trained on T0 of the temporal LASTFM split, evaluated on T2; saved models for the OT kill test |
| `conf/yambda50m_temporal.yml` | EASE, SASRec, GRU4Rec trained on T0 of Yambda-50M; candidate pools on T1a / T1b / T2 |
| `conf/tuning_example.yml` | random-search tuning (`type: opt`; `type: bayopt` uses the same format) |
| `conf/preprocess/*.yml` | raw data to train/test splits: LASTFM (session-aware), Diginetica, 30Music, rsc15 (single, window) |

The window configs run only the last slice (`data.skip: [0, 1, 2, 3]`); remove `skip` for all 5.

A config has a `type` (`single`: one split, `window`: several time slices, `opt` / `bayopt`: tuning on the
validation split), an `evaluation` module (`evaluation`: next item in anonymous sessions, `evaluation_user_based`:
session-aware, plus `evaluation_last`, `evaluation_multiple`, `evaluation_next_multiple`), the `data`, the `metrics`
(classes in `src/evaluation/metrics/`) and the `algorithms`, each a class path below `src/algorithms/` with its
constructor `params`.

## Models

| | Class |
|---|---|
| Popularity, random | `baselines.pop.Pop`, `baselines.spop.SessionPop`, `baselines.rpop.RPop`, `baselines.random.RandomPred` |
| Rules, Markov, MF | `baselines.ar.AssociationRules`, `baselines.sr.SequentialRules`, `baselines.markov.MarkovModel`, `baselines.bpr.BPR`, `ct.ct.ContextTree`, `ease.ease.EASE` (session-aware with `use_history`) |
| Nearest neighbours | `knn.iknn.ItemKNN`, `knn.sknn.ContextKNN`, `knn.vsknn.VMContextKNN`, `knn.stan.STAN`, `knn.vstan.VSKNN_STAN` |
| Hybrids | `hybrid.weighted.WeightedHybrid`, `hybrid.mixed.MixedHybrid`, `hybrid.cascading.CascadingHybrid`, `hybrid.strategic*` |
| Session-aware non-neural | `baselines.usr.USequentialRules`, `knn.uvsknn.UVMContextKNN`, `knn.ustan.USTAN`, `knn.uvstan.U_VSKNN_STAN` (extend / boost / remind extensions) |
| Neural, session-based | `gru4rec.adapter.GRU4RecTorch`, `narm.narm.NARM`, `stamp.stamp.Seq2SeqAttNN`, `sgnn.gnn.GGNN`, `sasrec.sasrec.SASRec` |
| Neural, session-aware | `hgru4rec.hgru4rec.HGRU4Rec`, `iirnn.ii_rnn.IIRNN`, `ncfs.ncfs.NCFS`, `nsar.nsar.NSAR`, `shan.shan.SHAN`, `narm.unarm.UNARM`, `gru4rec.ugru4rec.UGRU4RecTorch` |
| Precomputed | `filemodel.filemodel.FileModel` (pickled model), `filemodel.resultfile.ResultFile` (recommendation lists) |

Adding a model: [docs/adding-a-model.md](docs/adding-a-model.md).

## Differences from the original session-rec

The port was checked against the original code (identical recommendation lists for the non-neural models,
neural models within seed noise) and against Latifi et al. 2021, Table 6 (LASTFM: all 15 rows reproduce). The
evidence is in the git tag `port-verified`.

* The neural models are PyTorch translations of session-rec's Theano / TensorFlow / Keras code. GRU4Rec is the
  authors' official PyTorch implementation (see `src/algorithms/gru4rec/NOTICE.md` for its license); it has no
  top1-max loss, so UGRU4Rec differs from the original.
* LASTFM preprocessing reads every line of the raw file. The original's parser lost or merged 52,006 of the
  19,150,868 plays (quoted fields), so the prepared data differs slightly from the data Latifi et al. used.
* `vstan_eb` in `conf/lastfm_baselines.yml` uses `lambda_spw` 1.4525; the 0.4525 printed in the paper's Table A.14
  does not reproduce its Table 6.
* Not ported: CSRM, NextItNet, AE, APGNN, SMF, RepeatNet, sbr_adapter.

## Credits

The framework and the evaluation protocol are from:

* S. Latifi, N. Mauro, D. Jannach. Session-aware recommendation: a surprising quest for the state-of-the-art.
  Information Sciences, 2021.
* M. Ludewig, N. Mauro, S. Latifi, D. Jannach. Empirical analysis of session-based recommendation algorithms.
  User Modeling and User-Adapted Interaction 31(1), 2021.
* M. Ludewig, N. Mauro, S. Latifi, D. Jannach. Performance comparison of neural and non-neural approaches to
  session-based recommendation. RecSys 2019.
* M. Ludewig, D. Jannach. Evaluation of session-based recommendation algorithms. User Modeling and User-Adapted
  Interaction 28(4-5), 2018.

The models follow the code of their authors, as collected in the original session-rec: HGRU4Rec (Quadrana et al.,
RecSys 2017), II-RNN (Ruocco et al., DLRS 2017), SHAN (Ying et al., IJCAI 2018), NCFS (Hu et al., IEEE Intelligent
Systems 2019), NSAR (Phuong et al., IEEE Access 2019), GRU4Rec (Hidasi et al., ICLR 2016 / CIKM 2018), STAMP (Liu
et al., KDD 2018), NARM (Li et al., CIKM 2017), SR-GNN (Wu et al., AAAI 2019), BPR (Rendle et al., UAI 2009),
Context Tree (Mi et al., 2018).
