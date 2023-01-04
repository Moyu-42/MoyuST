#!/bin/bash
set -eo pipefail

export CUDA_VISIBLE_DEVICES=4

fairseq-generate /data/hrsun/data/MUST-C/en-de \
    --user-dir /home/hrsun/Speech/MoyuST/MoyuST \
    --quiet \
    --gen-subset tst-COMMON_st --task speech_to_text --prefix-size 1 \
    --lenpen 0.6 \
    --batch-size 16 --max-source-positions 4000000 --beam 10 \
    --config-yaml config_st.yaml  --path /data/hrsun/models/SpeechTrans/MoyuST_dist_27.pt \
    --scoring sacrebleu
