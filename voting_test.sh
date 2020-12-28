#!/usr/bin/env sh
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
log_name="PartSeg_VOTE_"$now""
CUDA_VISIBLE_DEVICES=2,3 python -u voting_test.py \
--config cfgs/config_partseg_test.yaml \
2>&1|tee log/$log_name.log &
