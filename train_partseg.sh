#!/usr/bin/env sh
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
log_name="PartSeg_LOG_"$now""
export CUDA_VISIBLE_DEVICES=0
python -u train_partseg.py \
--config cfgs/config_msn_partseg.yaml \
2>&1|tee log/$log_name.log &
