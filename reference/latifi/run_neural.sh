#!/bin/bash
# Run every conf/latifi_lastfm/neural/*.yml in parallel; GPU configs round-robin over the GPUs given as arguments.
# usage: reference/latifi/run_neural.sh <gpu> [<gpu> ...]
# Logs: reference/latifi/logs/neural/<config>.log ; done marker: reference/latifi/logs/neural/all.done
cd "$(dirname "$0")/../.."
gpus=("$@"); mkdir -p reference/latifi/logs/neural; rm -f reference/latifi/logs/neural/all.done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
i=0
for f in conf/latifi_lastfm/neural/*.yml; do c=$(basename "$f" .yml)
  g=${gpus[$((i % ${#gpus[@]}))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g .venv/bin/python src/run_config.py "$f" > "reference/latifi/logs/neural/$c.log" 2>&1 &
done
wait; touch reference/latifi/logs/neural/all.done
