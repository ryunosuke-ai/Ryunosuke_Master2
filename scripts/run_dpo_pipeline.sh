#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

PIPELINE_LOG_DIR="${PIPELINE_LOG_DIR:-logs/dpo_pipeline}"
mkdir -p "$PIPELINE_LOG_DIR"
mkdir -p artifacts/datasets
mkdir -p artifacts/training_runs

LOG_FILE="${PIPELINE_LOG_DIR}/dpo_pipeline_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "DPO pipeline started at $(date)"
echo "Log file: $LOG_FILE"
echo "========================================"

echo "[1/3] GPU status before run"
nvidia-smi

echo "[2/3] Generating DPO preference dataset"

python3 -m tools.translate_and_generate_dpo \
    --input artifacts/datasets/dailydialog_selected_en_500.jsonl \
    --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json \
    --output artifacts/datasets/dailydialog_ja_dpo_preferences_500_parallel.jsonl \
    --model gpt-5.4 \
    --score-model gpt-5.4 \
    --audit-model gpt-5.4-pro \
    --candidates 4 \
    --min-score-gap 0.25 \
    --min-chosen-posterior 0.70 \
    --max-rejected-posterior 0.55 \
    --max-records 300 \
    --workers 4 \
    --seed 42

echo "[3/3] Starting DPO LoRA training"

python3 -m tools.train_qwen35_dpo_lora \
    --dataset artifacts/datasets/dailydialog_ja_dpo_preferences_500_parallel.jsonl \
    --model-id "${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}" \
    --output-dir artifacts/training_runs/qwen35_bayes_dpo_lora_dailydialog_500_ep1_lr5e-6_r8_a16_no4bit \
    --num-train-epochs 1 \
    --learning-rate 5e-6 \
    --beta 0.1 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --save-steps 25 \
    --warmup-ratio 0.03 \
    --eval-ratio 0 \
    --seed 42 \
    --no-4bit

echo "========================================"
echo "DPO pipeline completed at $(date)"
echo "========================================"

nvidia-smi
