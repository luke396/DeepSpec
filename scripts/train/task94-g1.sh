#!/usr/bin/env bash
set -euo pipefail

: "${GPU_ID:?set the approved single GPU id}"
: "${EXACT_TARGET_PATH:?set the read-back exact Qwen3-8B snapshot path}"
: "${SOURCE_MANIFEST:?set task #100 exact-eight source JSONL}"
: "${PRODUCER_MANIFESTS:?set the frozen two-boot producer JSON file}"
: "${PARITY_CACHE:?set task #100 exact-eight HF oracle cache}"
: "${LEDGER_DIR:?set a new empty ledger directory}"
: "${VLLM_ENDPOINT:?set the single frozen stock-vLLM router endpoint}"
: "${SELECTION_POLICY_DIGEST:?set task #100 exact selection-policy SHA256}"
: "${RUN_ID:?set a fresh immutable G1 run id}"
: "${CONSUMER_BOOT_ID:?set a fresh immutable consumer boot id}"
: "${FRESH_EXP_NAME:?set a new G1 experiment name}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py \
    --config config/dspark/dspark_qwen3_8b_live.py \
    --opts "exp_name=${FRESH_EXP_NAME}" \
    --opts "model.target_model_name_or_path=${EXACT_TARGET_PATH}" \
    --opts "train.global_batch_size=8" \
    --opts "train.num_train_epochs=1" \
    --opts "train.max_train_steps=1" \
    --opts "train.torch_compile=false" \
    --opts "data.source_manifest_path=${SOURCE_MANIFEST}" \
    --opts "data.producer_manifests_path=${PRODUCER_MANIFESTS}" \
    --opts "data.parity_cache_path=${PARITY_CACHE}" \
    --opts "data.ledger_dir=${LEDGER_DIR}" \
    --opts "data.vllm_endpoint=${VLLM_ENDPOINT}" \
    --opts "data.selection_policy_digest=${SELECTION_POLICY_DIGEST}" \
    --opts "data.run_id=${RUN_ID}" \
    --opts "data.consumer_boot_id=${CONSUMER_BOOT_ID}"
