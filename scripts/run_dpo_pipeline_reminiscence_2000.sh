#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-reminiscence_5000_to_2000}"
MAX_DIALOGUES="${MAX_DIALOGUES:-5000}"
MAX_CONTEXT_TURNS="${MAX_CONTEXT_TURNS:-8}"
TARGET_SELECTED="${TARGET_SELECTED:-3200}"
TARGET_DPO_RECORDS="${TARGET_DPO_RECORDS:-2000}"
MIN_POSTERIOR="${MIN_POSTERIOR:-0.75}"
MIN_CONTEXT_TURNS="${MIN_CONTEXT_TURNS:-1}"
PER_DIALOGUE_LIMIT="${PER_DIALOGUE_LIMIT:-3}"
SCORING_WORKERS="${SCORING_WORKERS:-4}"
DPO_WORKERS="${DPO_WORKERS:-4}"
SCORING_MODEL="${SCORING_MODEL:-gpt-5.4}"
GENERATION_MODEL="${GENERATION_MODEL:-gpt-5.4}"
LOCAL_MODEL_ID="${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"

BAYES_MODEL="${BAYES_MODEL:-artifacts/bayes_models/generated_transition_bayes_model.json}"
PREPARED_DATA="data/dailydialog_for_scoring_${RUN_TAG}.jsonl"
SCORED_DATA="artifacts/scored_dialogues/dailydialog_transition_scored_${RUN_TAG}.jsonl"
SELECTED_DATA="artifacts/datasets/dailydialog_selected_en_${RUN_TAG}.jsonl"
DPO_DATA="artifacts/datasets/dailydialog_ja_dpo_preferences_${RUN_TAG}.jsonl"
TRAINING_OUTPUT="artifacts/training_runs/qwen35_bayes_dpo_lora_${RUN_TAG}_ep1_lr5e-6_r8_a16_no4bit"

PIPELINE_LOG_DIR="${PIPELINE_LOG_DIR:-logs/dpo_pipeline}"
mkdir -p "$PIPELINE_LOG_DIR" data artifacts/scored_dialogues artifacts/datasets artifacts/training_runs

LOG_FILE="${PIPELINE_LOG_DIR}/dpo_pipeline_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Reminiscence DPO pipeline started at $(date)"
echo "run_tag: $RUN_TAG"
echo "log_file: $LOG_FILE"
echo "max_dialogues: $MAX_DIALOGUES"
echo "target_selected: $TARGET_SELECTED"
echo "target_dpo_records: $TARGET_DPO_RECORDS"
echo "========================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[STEP 0/7] GPU status before run"
  nvidia-smi
else
  echo "[STEP 0/7] nvidia-smi not found; training step still requires CUDA."
fi

echo "[STEP 1/7] Prepare DailyDialog records"
python3 -m tools.prepare_dailydialog_for_scoring \
  --split train \
  --max-dialogues "$MAX_DIALOGUES" \
  --max-context-turns "$MAX_CONTEXT_TURNS" \
  --output "$PREPARED_DATA"

echo "[STEP 2/7] Score DailyDialog with transition Bayes model"
python3 -m tools.score_dialogue_with_transition_bayes_model \
  --input "$PREPARED_DATA" \
  --bayes-model "$BAYES_MODEL" \
  --output "$SCORED_DATA" \
  --model "$SCORING_MODEL" \
  --workers "$SCORING_WORKERS"

echo "[STEP 3/7] Select reminiscence-oriented high-posterior candidates"
python3 -m tools.extract_high_posterior_dialogues \
  --input "$SCORED_DATA" \
  --output "$SELECTED_DATA" \
  --min-posterior "$MIN_POSTERIOR" \
  --min-context-turns "$MIN_CONTEXT_TURNS" \
  --target-records "$TARGET_SELECTED" \
  --per-dialogue-limit "$PER_DIALOGUE_LIMIT" \
  --require-preferred \
  --sort-by-selection

echo "[STEP 4/7] Generate Japanese DPO preference data"
python3 -m tools.translate_and_generate_dpo \
  --input "$SELECTED_DATA" \
  --bayes-model "$BAYES_MODEL" \
  --output "$DPO_DATA" \
  --model "$GENERATION_MODEL" \
  --score-model "$SCORING_MODEL" \
  --candidates 4 \
  --min-score-gap 0.25 \
  --min-chosen-posterior 0.70 \
  --max-rejected-posterior 0.55 \
  --target-records "$TARGET_DPO_RECORDS" \
  --workers "$DPO_WORKERS" \
  --seed 42

echo "[STEP 5/7] Dry-run training data validation"
python3 -m tools.train_qwen35_dpo_lora \
  --dataset "$DPO_DATA" \
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
  --seed 42 \
  --no-4bit \
  --dry-run

echo "[STEP 6/7] Train Qwen3.5 DPO LoRA"
python3 -m tools.train_qwen35_dpo_lora \
  --dataset "$DPO_DATA" \
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
  --seed 42 \
  --no-4bit

echo "[STEP 7/7] Output summary"
python3 - <<PY
import json
from pathlib import Path

paths = {
    "prepared": Path("$PREPARED_DATA"),
    "scored": Path("$SCORED_DATA"),
    "selected": Path("$SELECTED_DATA"),
    "dpo": Path("$DPO_DATA"),
}
for name, path in paths.items():
    count = 0
    if path.exists():
        count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    print(f"{name}: {path} ({count} lines)")

dpo_path = paths["dpo"]
if dpo_path.exists():
    rows = [json.loads(line) for line in dpo_path.open(encoding="utf-8") if line.strip()]
    if rows:
        gaps = [float(row["score_gap"]) for row in rows]
        chosen = [float(row["score_chosen"]) for row in rows]
        rejected = [float(row["score_rejected"]) for row in rows]
        print(f"score_gap: min={min(gaps):.3f} mean={sum(gaps)/len(gaps):.3f} max={max(gaps):.3f}")
        print(f"score_chosen: min={min(chosen):.3f} mean={sum(chosen)/len(chosen):.3f} max={max(chosen):.3f}")
        print(f"score_rejected: min={min(rejected):.3f} mean={sum(rejected)/len(rejected):.3f} max={max(rejected):.3f}")
print(f"training_dataset: $DPO_DATA")
print(f"training_output: $TRAINING_OUTPUT")
print(f"log_file: $LOG_FILE")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[STEP 7/7] GPU status after run"
  nvidia-smi
fi

echo "========================================"
echo "Reminiscence DPO pipeline completed at $(date)"
echo "========================================"
