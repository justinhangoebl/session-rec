#!/bin/bash
# Run every conf/ref_pm/<set>/<key>.yml in the ORIGINAL env (envs/srec37, Python 3.7) from the pristine-layout code.
# NB: the original layout is the commit before the move; run from ../session-rec-orig-run (a writable checkout).
cd "$1"
for f in conf/ref_pm/*/*.yml; do
  s=$(basename $(dirname $f)); k=$(basename $f .yml)
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup "$2/envs/srec37/bin/python" run_config.py "$f" > "$2/reference/pm_logs/${s}_${k}.log" 2>&1 &
done
wait; touch "$2/reference/pm_logs/done"
