#!/usr/bin/env sh
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
log_name="PartSeg_LOG_"$now""
CUDA_VISIBLE_DEVICES=2,3 python -u train_partseg_gpus.py \
--config cfgs/config_partseg_gpus.yaml \
2>&1|tee log/$log_name.log &
