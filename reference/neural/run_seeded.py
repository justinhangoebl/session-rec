"""Run one session-rec config with a forced global seed, without editing algorithm code.

usage: SEED=1 python reference/neural/run_seeded.py conf.yml   (run from the repo root)
Overrides: np.random.seed (hard-coded 42 in GRU4Rec/NARM), the TF1 graph seed
(STAMP, SR-GNN are otherwise unseeded) and the Theano RandomStreams seeds.
"""
import os, sys, random
sys.path.insert(0, os.getcwd())
SEED = int(os.environ['SEED'])
random.seed(SEED)

import numpy as np
_orig = np.random.seed
np.random.seed = lambda *a, **k: _orig(SEED)
_orig(SEED)

import tensorflow as tf
_ginit = tf.Graph.__init__
def _graph_init(self, *a, **k):
    _ginit(self, *a, **k)
    self.seed = SEED
tf.Graph.__init__ = _graph_init
tf.set_random_seed(SEED)

from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
for cls in (RandomStreams, MRG_RandomStreams):
    _i = cls.__init__
    def _init(self, seed=None, *a, _i=_i, **k):
        _i(self, SEED, *a, **k)
    cls.__init__ = _init

import run_config
run_config.main(sys.argv[1])
