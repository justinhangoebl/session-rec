#!/bin/bash
# Run every conf/latifi_lastfm/<stack>/<row>.yml on both stacks in parallel (one process per config).
#   orig: envs/srec37 (Python 3.7) from the pristine-layout copy ../session-rec-origrun
#   port: .venv
# Logs: reference/latifi/logs/<stack>_<row>.log ; done marker: reference/latifi/logs/all.done
cd "$(dirname "$0")/../.."
D=$PWD
mkdir -p reference/latifi/logs; rm -f reference/latifi/logs/all.done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for f in conf/latifi_lastfm/orig/*.yml; do r=$(basename "$f" .yml)
  (cd ../session-rec-origrun && "$D/envs/srec37/bin/python" run_config.py "$D/$f" > "$D/reference/latifi/logs/orig_$r.log" 2>&1) &
done
for f in conf/latifi_lastfm/port/*.yml; do r=$(basename "$f" .yml)
  .venv/bin/python src/run_config.py "$f" > "reference/latifi/logs/port_$r.log" 2>&1 &
done
wait; touch reference/latifi/logs/all.done
