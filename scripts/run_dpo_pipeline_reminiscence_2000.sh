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
PIPELINE_MODE="${PIPELINE_MODE:-streaming}"
STREAM_POLL_SECONDS="${STREAM_POLL_SECONDS:-10}"
STREAM_DPO_BATCH_SIZE="${STREAM_DPO_BATCH_SIZE:-$DPO_WORKERS}"
SCORING_MODEL="${SCORING_MODEL:-gpt-5.4}"
GENERATION_MODEL="${GENERATION_MODEL:-gpt-5.4}"
LOCAL_MODEL_ID="${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
TRAIN_DEVICE_MAP="${TRAIN_DEVICE_MAP:-auto}"
TRAIN_MAX_MEMORY="${TRAIN_MAX_MEMORY:-0=46GiB,1=46GiB,cpu=0GiB}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
RUN_ORACLE_EVALUATION="${RUN_ORACLE_EVALUATION:-1}"
ORACLE_PROMPTS="${ORACLE_PROMPTS:-${PROMPTS:-configs/evaluation_prompts/reminiscence_oracle_eval_v2_100.jsonl}}"
SMALL_CORPUS="${SMALL_CORPUS:-data/small_corpus.jsonl}"
ORACLE_MODEL="${ORACLE_MODEL:-gpt-5.4-pro}"
ORACLE_WORKERS="${ORACLE_WORKERS:-2}"
MAX_ORACLE_PROMPTS="${MAX_ORACLE_PROMPTS:-${MAX_PROMPTS:-}}"
ORACLE_SKIP_PROMPTS="${ORACLE_SKIP_PROMPTS:-${SKIP_PROMPTS:-}}"
ORACLE_CATEGORIES="${ORACLE_CATEGORIES:-${CATEGORIES:-}}"
LOCAL_PROMPT_MODE="${LOCAL_PROMPT_MODE:-instruction}"

BAYES_MODEL="${BAYES_MODEL:-artifacts/bayes_models/generated_transition_bayes_model.json}"
PREPARED_DATA="data/dailydialog_for_scoring_${RUN_TAG}.jsonl"
SCORED_DATA="artifacts/scored_dialogues/dailydialog_transition_scored_${RUN_TAG}.jsonl"
SELECTED_DATA="artifacts/datasets/dailydialog_selected_en_${RUN_TAG}.jsonl"
DPO_DATA="artifacts/datasets/dailydialog_ja_dpo_preferences_${RUN_TAG}.jsonl"
TRAINING_OUTPUT="artifacts/training_runs/qwen35_bayes_dpo_lora_${RUN_TAG}_ep1_lr5e-6_r8_a16_no4bit"
ORACLE_OUTPUT_DIR="${ORACLE_OUTPUT_DIR:-${OUTPUT_DIR:-artifacts/evaluations/oracle_eval_runs/${RUN_TAG}_oracle_v2}}"
SCORING_DONE_FILE="${SCORING_DONE_FILE:-artifacts/scored_dialogues/dailydialog_transition_scored_${RUN_TAG}.done}"
SCORING_STATUS_FILE="${SCORING_STATUS_FILE:-artifacts/scored_dialogues/dailydialog_transition_scored_${RUN_TAG}.status}"
STREAM_PROGRESS_LEDGER="${STREAM_PROGRESS_LEDGER:-artifacts/datasets/dailydialog_ja_dpo_preferences_${RUN_TAG}.progress.jsonl}"
PIPELINE_HEARTBEAT_FILE="${PIPELINE_HEARTBEAT_FILE:-artifacts/run_logs/dpo_pipeline_${RUN_TAG}.heartbeat.json}"

PIPELINE_LOG_DIR="${PIPELINE_LOG_DIR:-logs/dpo_pipeline/reminiscence}"
mkdir -p "$PIPELINE_LOG_DIR" data artifacts/scored_dialogues artifacts/datasets artifacts/training_runs artifacts/run_logs

LOG_FILE="${PIPELINE_LOG_DIR}/dpo_pipeline_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
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
echo "Reminiscence DPO pipeline started at $(date)"
echo "WARNING: this script is deprecated for the current presentation. Use ESConv scripts unless reproducing the old reminiscence run."
echo "run_tag: $RUN_TAG"
echo "log_file: $LOG_FILE"
echo "max_dialogues: $MAX_DIALOGUES"
echo "target_selected: $TARGET_SELECTED"
echo "target_dpo_records: $TARGET_DPO_RECORDS"
echo "pipeline_mode: $PIPELINE_MODE"
echo "run_oracle_evaluation: $RUN_ORACLE_EVALUATION"
echo "oracle_workers: $ORACLE_WORKERS"
echo "train_cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-}"
echo "train_device_map: ${TRAIN_DEVICE_MAP:-}"
echo "train_max_memory: ${TRAIN_MAX_MEMORY:-}"
echo "pytorch_cuda_alloc_conf: ${PYTORCH_CUDA_ALLOC_CONF:-}"
echo "========================================"

terminate_process_group() {
  local pid="$1"
  local reason="$2"
  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "$reason"
    kill -TERM -- "-$pid" >/dev/null 2>&1 || kill -TERM "$pid" >/dev/null 2>&1 || true
    sleep 5
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -KILL -- "-$pid" >/dev/null 2>&1 || kill -KILL "$pid" >/dev/null 2>&1 || true
    fi
  fi
}

start_scoring_background() {
  rm -f "$SCORING_DONE_FILE" "$SCORING_STATUS_FILE"
  if command -v setsid >/dev/null 2>&1; then
    setsid bash -c '
      set +e
      python3 -m tools.score_dialogue_with_transition_bayes_model \
        --input "$1" \
        --bayes-model "$2" \
        --output "$3" \
        --model "$4" \
        --workers "$5" \
        --fallback-on-errors
      status=$?
      echo "$status" > "$7"
      touch "$6"
      exit "$status"
    ' bash "$PREPARED_DATA" "$BAYES_MODEL" "$SCORED_DATA" "$SCORING_MODEL" "$SCORING_WORKERS" "$SCORING_DONE_FILE" "$SCORING_STATUS_FILE" &
  else
    (
      set +e
      python3 -m tools.score_dialogue_with_transition_bayes_model \
        --input "$PREPARED_DATA" \
        --bayes-model "$BAYES_MODEL" \
        --output "$SCORED_DATA" \
        --model "$SCORING_MODEL" \
        --workers "$SCORING_WORKERS" \
        --fallback-on-errors
      status=$?
      echo "$status" > "$SCORING_STATUS_FILE"
      touch "$SCORING_DONE_FILE"
      exit "$status"
    ) &
  fi
  SCORING_PID="$!"
}

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[STEP 0/8] GPU status before run"
  nvidia-smi
else
  echo "[STEP 0/8] nvidia-smi not found; training step still requires CUDA."
fi

echo "[STEP 1/8] Prepare DailyDialog records"
python3 -m tools.prepare_dailydialog_for_scoring \
  --split train \
  --max-dialogues "$MAX_DIALOGUES" \
  --max-context-turns "$MAX_CONTEXT_TURNS" \
  --output "$PREPARED_DATA"

if [ "$PIPELINE_MODE" = "streaming" ]; then
  echo "[STEP 2/8] Score DailyDialog with transition Bayes model in background"
  start_scoring_background
  echo "scoring_pid: $SCORING_PID"

  echo "[STEP 3/8] Stream scored candidates into Japanese DPO preference data"
  set +e
  python3 -m tools.stream_dpo_from_scored \
    --scored-input "$SCORED_DATA" \
    --selected-output "$SELECTED_DATA" \
    --dpo-output "$DPO_DATA" \
    --bayes-model "$BAYES_MODEL" \
    --model "$GENERATION_MODEL" \
    --score-model "$SCORING_MODEL" \
    --target-records "$TARGET_DPO_RECORDS" \
    --workers "$DPO_WORKERS" \
    --batch-size "$STREAM_DPO_BATCH_SIZE" \
    --poll-seconds "$STREAM_POLL_SECONDS" \
    --candidates 4 \
    --style-preset reminiscence \
    --min-score-gap 0.25 \
    --min-chosen-posterior 0.70 \
    --max-rejected-posterior 0.55 \
    --min-posterior "$MIN_POSTERIOR" \
    --min-context-turns "$MIN_CONTEXT_TURNS" \
    --per-dialogue-limit "$PER_DIALOGUE_LIMIT" \
    --require-preferred \
    --ledger "$STREAM_PROGRESS_LEDGER" \
    --done-file "$SCORING_DONE_FILE" \
    --heartbeat-file "$PIPELINE_HEARTBEAT_FILE" \
    --seed 42
  stream_status=$?
  set -e

  if [ "$stream_status" -eq 0 ]; then
    echo "[STEP 4/8] DPO target reached; stop background scoring and continue"
    terminate_process_group "$SCORING_PID" "target DPO records reached; terminating background scoring pid=$SCORING_PID"
    wait "$SCORING_PID" >/dev/null 2>&1 || true
  elif [ "$stream_status" -eq 2 ]; then
    echo "[STEP 4/8] Scoring finished before DPO target was reached"
    if ! wait "$SCORING_PID"; then
      echo "background scoring failed before target DPO records were available"
      exit 1
    fi
    echo "DPO target was not reached after all scored rows were consumed"
    exit 1
  else
    echo "[STEP 4/8] Streaming DPO generation failed with status=$stream_status"
    terminate_process_group "$SCORING_PID" "streaming DPO generation failed; terminating background scoring pid=$SCORING_PID"
    wait "$SCORING_PID" >/dev/null 2>&1 || true
    exit "$stream_status"
  fi
else
  echo "[STEP 2/8] Score DailyDialog with transition Bayes model"
  python3 -m tools.score_dialogue_with_transition_bayes_model \
    --input "$PREPARED_DATA" \
    --bayes-model "$BAYES_MODEL" \
    --output "$SCORED_DATA" \
    --model "$SCORING_MODEL" \
    --workers "$SCORING_WORKERS" \
    --fallback-on-errors

  echo "[STEP 3/8] Select reminiscence-oriented high-posterior candidates"
  python3 -m tools.extract_high_posterior_dialogues \
    --input "$SCORED_DATA" \
    --output "$SELECTED_DATA" \
    --min-posterior "$MIN_POSTERIOR" \
    --min-context-turns "$MIN_CONTEXT_TURNS" \
    --target-records "$TARGET_SELECTED" \
    --per-dialogue-limit "$PER_DIALOGUE_LIMIT" \
    --require-preferred \
    --sort-by-selection

  echo "[STEP 4/8] Generate Japanese DPO preference data"
  python3 -m tools.translate_and_generate_dpo \
    --input "$SELECTED_DATA" \
    --bayes-model "$BAYES_MODEL" \
    --output "$DPO_DATA" \
    --model "$GENERATION_MODEL" \
    --score-model "$SCORING_MODEL" \
    --candidates 4 \
    --style-preset reminiscence \
    --min-score-gap 0.25 \
    --min-chosen-posterior 0.70 \
    --max-rejected-posterior 0.55 \
    --target-records "$TARGET_DPO_RECORDS" \
    --workers "$DPO_WORKERS" \
    --skip-sample-errors \
    --seed 42
fi

echo "[STEP 5/8] Dry-run training data validation"
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
  "${TRAIN_PLACEMENT_ARGS[@]}" \
  --dry-run

echo "[STEP 6/8] Train Qwen3.5 DPO LoRA"
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
  "${TRAIN_PLACEMENT_ARGS[@]}"

if [ "$RUN_ORACLE_EVALUATION" = "1" ]; then
  echo "[STEP 7/8] Oracle evaluation for base vs DPO"
  oracle_args=(
    python3 -m tools.run_oracle_evaluation
    --prompts "$ORACLE_PROMPTS"
    --small-corpus "$SMALL_CORPUS"
    --bayes-model "$BAYES_MODEL"
    --base-model-id "$LOCAL_MODEL_ID"
    --lora-path "$TRAINING_OUTPUT"
    --oracle-model "$ORACLE_MODEL"
    --oracle-workers "$ORACLE_WORKERS"
    --output-dir "$ORACLE_OUTPUT_DIR"
    --seed 42
    --max-new-tokens 192
    --temperature 0.7
    --top-p 0.8
    --repetition-penalty 1.0
    --local-prompt-mode "$LOCAL_PROMPT_MODE"
    --no-4bit
  )
  if [ -n "$MAX_ORACLE_PROMPTS" ]; then
    oracle_args+=(--max-prompts "$MAX_ORACLE_PROMPTS")
  fi
  if [ -n "$ORACLE_SKIP_PROMPTS" ]; then
    oracle_args+=(--skip-prompts "$ORACLE_SKIP_PROMPTS")
  fi
  if [ -n "$ORACLE_CATEGORIES" ]; then
    oracle_args+=(--categories "$ORACLE_CATEGORIES")
  fi
  "${oracle_args[@]}"
else
  echo "[STEP 7/8] Skip Oracle evaluation because RUN_ORACLE_EVALUATION=$RUN_ORACLE_EVALUATION"
fi

echo "[STEP 8/8] Output summary"
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
print(f"oracle_output: $ORACLE_OUTPUT_DIR")
print(f"log_file: $LOG_FILE")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[STEP 8/8] GPU status after run"
  nvidia-smi
fi

echo "========================================"
echo "Reminiscence DPO pipeline completed at $(date)"
echo "========================================"
