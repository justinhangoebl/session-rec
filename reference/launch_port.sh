#!/bin/bash
# usage: reference/launch_port.sh <model> <gpu> <gpu> <gpu>   -> runs conf/port_neural/<model>_seed{1,2,3}.yml, one GPU each
cd "$(dirname "$0")/.."
m=$1; shift; mkdir -p reference/port_logs; rm -f reference/port_logs/$m.done
i=0
for g in "$@"; do i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 .venv/bin/python src/run_config.py conf/port_neural/${m}_seed$i.yml > reference/port_logs/${m}_$i.log 2>&1 &
done
wait; touch reference/port_logs/$m.done
