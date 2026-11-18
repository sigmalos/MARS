#!/usr/bin/env bash
# Usage: scripts/run_exp.sh <tag> <seed> <gpu> [universe] [model_kwargs_json]
# Trains on opensource data from scratch, writes logs/<tag>_<universe>_<seed>.log and results/<tag>_<universe>_<seed>.json
set -e
TAG=$1; SEED=$2; GPU=$3; UNI=${4:-csi300}; KW=${5:-'{}'}
cd "$(dirname "$0")/.."
mkdir -p logs results checkpoints
CUDA_VISIBLE_DEVICES=$GPU exec conda run --no-capture-output -n quant python -u main.py --mode train \
  --universe "$UNI" --prefix opensource --seeds "$SEED" --save_dir "checkpoints/$TAG" \
  --model_kwargs "$KW" --result_file "results/${TAG}_${UNI}_${SEED}.json" > "logs/${TAG}_${UNI}_${SEED}.log" 2>&1
