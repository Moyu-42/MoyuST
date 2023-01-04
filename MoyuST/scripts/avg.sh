#!/bin/bash
set -eo pipefail

PATH=$1

for i in {25..34}; do
    /data/hrsun/anaconda3/envs/lt/bin/python scripts/average_checkpoints.py --inputs /data/hrsun/models/$PATH --output /data/hrsun/models/$PATH/checkpoint.best_avg.$i-10.pt --num-epoch-checkpoints 10 --checkpoint-upper-bound $i
done