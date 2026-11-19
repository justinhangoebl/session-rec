#!/bin/bash
# Run every conf/ref_neural/<model>_seed<N>.yml in parallel, one thread each. Skips finished runs.
cd "$(dirname "$0")/../.."
LOG=${LOG:-reference/neural/logs}; mkdir -p "$LOG"
for f in conf/ref_neural/*_seed*.yml; do
  b=$(basename "$f" .yml); m=${b%_seed*}; s=${b##*seed}
  [ -f "results/ref/neural/$m/seed$s/test_single_baselines_and_models_and_hybrids_rsc15-100k-Saver@50.csv" ] && continue
  SEED=$s OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  THEANO_FLAGS="device=cpu,floatX=float32,base_compiledir=/tmp/justin-theano/${m}_$s" \
  nohup ./envs/srec37/bin/python reference/neural/run_seeded.py "$f" > "$LOG/${m}_$s.log" 2>&1 &
done
wait
