# session-rec

Evaluation framework and baselines for session-based and session-aware recommendation. A port of
[rn5l/session-rec](https://github.com/rn5l/session-rec) (Ludewig, Mauro, Latifi, Jannach) to Python 3.12, current
NumPy / pandas / scikit-learn, and PyTorch for every neural model, managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync --extra torch --extra hdf    # torch: neural models; hdf: hdf data (data.type: hdf) and LASTFM preprocessing
```

## Run

```bash
uv run session-rec conf/rsc15_quick.yml           # one config; a folder runs every config in it
uv run session-rec-preprocess conf/preprocess/lastfm.yml
```

Results are written to `results.folder` of the config as `test_<type>_<key>_<data.name>[.<slice>].csv`.

| Config | What it does |
|---|---|
| `conf/rsc15_quick.yml` | smoke test on the bundled rsc15 sample (`data/rsc15/prepared/`) |
| `conf/lastfm_baselines.yml` | session-aware non-neural baselines on LASTFM with the tuned parameters of Latifi et al. 2021 |
| `conf/lastfm_neural.yml` | the neural models on LASTFM with the tuned parameters of Latifi et al. 2021 |
| `conf/tuning_example.yml` | random-search tuning (`type: opt`; `type: bayopt` uses the same format) |
| `conf/preprocess/*.yml` | raw data to train/test splits: LASTFM (session-aware), Diginetica, 30Music, rsc15 |

A config has a `type` (`single`: one split, `window`: several time slices, `opt` / `bayopt`: tuning on the
validation split), an `evaluation` module (`evaluation`: next item in anonymous sessions, `evaluation_user_based`:
session-aware, plus `evaluation_last`, `evaluation_multiple`, `evaluation_next_multiple`), the `data`, the `metrics`
(classes in `src/evaluation/metrics/`) and the `algorithms`, each a class path below `src/algorithms/` with its
constructor `params`.

## Models

| | Class |
|---|---|
| Popularity, random | `baselines.pop.Pop`, `baselines.spop.SessionPop`, `baselines.rpop.RPop`, `baselines.random.RandomPred` |
| Rules, Markov, MF | `baselines.ar.AssociationRules`, `baselines.sr.SequentialRules`, `baselines.markov.MarkovModel`, `baselines.bpr.BPR`, `ct.ct.ContextTree` |
| Nearest neighbours | `knn.iknn.ItemKNN`, `knn.sknn.ContextKNN`, `knn.vsknn.VMContextKNN`, `knn.stan.STAN`, `knn.vstan.VSKNN_STAN` |
| Hybrids | `hybrid.weighted.WeightedHybrid`, `hybrid.mixed.MixedHybrid`, `hybrid.cascading.CascadingHybrid`, `hybrid.strategic*` |
| Session-aware non-neural | `baselines.usr.USequentialRules`, `knn.uvsknn.UVMContextKNN`, `knn.ustan.USTAN`, `knn.uvstan.U_VSKNN_STAN` (extend / boost / remind extensions) |
| Neural, session-based | `gru4rec.adapter.GRU4RecTorch`, `narm.narm.NARM`, `stamp.stamp.Seq2SeqAttNN`, `sgnn.gnn.GGNN` |
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
