#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-esconv_5000_to_2000_random2500}"
RANDOM_DPO_TARGET_RECORDS="${RANDOM_DPO_TARGET_RECORDS:-2500}"
RANDOM_DPO_SEED="${RANDOM_DPO_SEED:-42}"
MAX_DIALOGUES="${MAX_DIALOGUES:-8000}"
MAX_CONTEXT_TURNS="${MAX_CONTEXT_TURNS:-8}"
DPO_WORKERS="${DPO_WORKERS:-2}"
GENERATION_MODEL="${GENERATION_MODEL:-${SCORING_LLM_MODEL:-gpt-5.4}}"
LOCAL_MODEL_ID="${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
TRAIN_DEVICE_MAP="${TRAIN_DEVICE_MAP:-auto}"
TRAIN_MAX_MEMORY="${TRAIN_MAX_MEMORY:-0=46GiB,1=46GiB,cpu=0GiB}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DAILYDIALOG_RANDOM_DPO_DATA="${DAILYDIALOG_RANDOM_DPO_DATA:-artifacts/datasets/dailydialog_ja_dpo_preferences_random2500_${RUN_TAG}_daily.jsonl}"
FINAL_DPO_DATA="${FINAL_DPO_DATA:-artifacts/datasets/dailydialog_random2500_ja_dpo_preferences_${RUN_TAG}.jsonl}"
TRAINING_OUTPUT="${TRAINING_OUTPUT:-artifacts/training_runs/qwen35_random2500_dailydialog_dpo_lora_${RUN_TAG}_ep1_lr5e-6_r8_a16_no4bit}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
PIPELINE_LOG_DIR="${PIPELINE_LOG_DIR:-logs/dpo_pipeline/random_dpo/${RUN_DATE}}"
LOG_FILE="${PIPELINE_LOG_FILE:-${PIPELINE_LOG_DIR}/random_dpo_pipeline_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$PIPELINE_LOG_DIR" artifacts/datasets artifacts/training_runs logs/dpo_pipeline/random_dpo
exec > >(tee -a "$LOG_FILE") 2>&1

export PYTORCH_CUDA_ALLOC_CONF
if [ -n "$TRAIN_CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES"
fi

TRAIN_PLACEMENT_ARGS=()
if [ -n "$TRAIN_DEVICE_MAP" ]; then
  TRAIN_PLACEMENT_ARGS+=(--device-map "$TRAIN_DEVICE_MAP")
fi
if [ -n "$TRAIN_MAX_MEMORY" ]; then
  TRAIN_PLACEMENT_ARGS+=(--max-memory "$TRAIN_MAX_MEMORY")
fi

echo "========================================"
echo "Random DailyDialog DPO pipeline started at $(date)"
echo "run_tag: $RUN_TAG"
echo "target_records: $RANDOM_DPO_TARGET_RECORDS"
echo "seed: $RANDOM_DPO_SEED"
echo "max_dialogues: $MAX_DIALOGUES"
echo "max_context_turns: $MAX_CONTEXT_TURNS"
echo "generation_model: $GENERATION_MODEL"
echo "dpo_data: $FINAL_DPO_DATA"
echo "daily_output: $DAILYDIALOG_RANDOM_DPO_DATA"
echo "training_output: $TRAINING_OUTPUT"
echo "log_file: $LOG_FILE"
echo "========================================"

echo "[STEP 1/4] Build Random DailyDialog DPO records"
python3 -m tools.build_random_dailydialog_dpo \
  --daily-output "$DAILYDIALOG_RANDOM_DPO_DATA" \
  --output "$FINAL_DPO_DATA" \
  --model "$GENERATION_MODEL" \
  --target-records "$RANDOM_DPO_TARGET_RECORDS" \
  --seed "$RANDOM_DPO_SEED" \
  --max-dialogues "$MAX_DIALOGUES" \
  --max-context-turns "$MAX_CONTEXT_TURNS" \
  --workers "$DPO_WORKERS" \
  --skip-sample-errors

echo "[STEP 2/4] Validate Random-DPO source composition"
python3 - "$FINAL_DPO_DATA" "$RANDOM_DPO_TARGET_RECORDS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
target = int(sys.argv[2])
rows = []
with path.open(encoding="utf-8") as file:
    for line in file:
        if line.strip():
            rows.append(json.loads(line))
daily = [
    row for row in rows
    if row.get("source_dataset") == "DailyDialog"
    and row.get("metadata", {}).get("source_dataset") == "DailyDialog"
]
gold = [row for row in rows if row.get("metadata", {}).get("esconv_gold_records", 0) not in (0, "0")]
missing = [index for index, row in enumerate(rows, start=1) if not all(row.get(key) for key in ("prompt", "chosen", "rejected"))]
print(f"records: {len(rows)}")
print(f"daily_dialog_random_records: {len(daily)}")
print(f"esconv_gold_records: {len(gold)}")
print(f"missing_required_rows: {len(missing)}")
if len(rows) != target or len(daily) != target or gold or missing:
    raise SystemExit(1)
PY

echo "[STEP 3/4] Dry-run training data validation"
python3 -m tools.train_qwen35_dpo_lora \
  --dataset "$FINAL_DPO_DATA" \
  --model-id "$LOCAL_MODEL_ID" \
  --output-dir "$TRAINING_OUTPUT" \
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
  --seed "$RANDOM_DPO_SEED" \
  --no-4bit \
  "${TRAIN_PLACEMENT_ARGS[@]}" \
  --dry-run

echo "[STEP 4/4] Train Qwen3.5 Random-DPO LoRA"
python3 -m tools.train_qwen35_dpo_lora \
  --dataset "$FINAL_DPO_DATA" \
  --model-id "$LOCAL_MODEL_ID" \
  --output-dir "$TRAINING_OUTPUT" \
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
  --seed "$RANDOM_DPO_SEED" \
  --no-4bit \
  "${TRAIN_PLACEMENT_ARGS[@]}"

echo "========================================"
echo "Random DailyDialog DPO pipeline completed at $(date)"
echo "dpo_data: $FINAL_DPO_DATA"
echo "training_output: $TRAINING_OUTPUT"
echo "========================================"
