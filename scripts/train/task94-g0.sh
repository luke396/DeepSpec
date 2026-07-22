#!/usr/bin/env bash
set -euo pipefail

: "${GPU_ID:?set the approved single GPU id}"
: "${EXACT_TARGET_PATH:?set the read-back exact Qwen3-8B snapshot path}"
: "${ONE_ROW_CACHE:?set the clean one-row offline target cache path}"
: "${FRESH_EXP_NAME:?set one new experiment name; reuse it unchanged for resume}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py \
    --config config/dspark/dspark_qwen3_8b_fresh.py \
    --opts "exp_name=${FRESH_EXP_NAME}" \
    --opts "model.target_model_name_or_path=${EXACT_TARGET_PATH}" \
    --opts "data.target_cache_path=${ONE_ROW_CACHE}"
