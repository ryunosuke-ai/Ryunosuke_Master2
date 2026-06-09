#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-esconv_support_mixed_8000_to_2500}"
ESCONV_ANALYSIS_CONVERSATIONS="${ESCONV_ANALYSIS_CONVERSATIONS:-80}"
MAX_DIALOGUES="${MAX_DIALOGUES:-8000}"
CHUNK_DIALOGUES="${CHUNK_DIALOGUES:-500}"
MAX_CONTEXT_TURNS="${MAX_CONTEXT_TURNS:-8}"
TARGET_SELECTED_PER_CHUNK="${TARGET_SELECTED_PER_CHUNK:-400}"
TARGET_DPO_RECORDS="${TARGET_DPO_RECORDS:-2000}"
ESCONV_GOLD_TARGET_RECORDS="${ESCONV_GOLD_TARGET_RECORDS:-500}"
ESCONV_GOLD_MAX_SOURCE_RECORDS="${ESCONV_GOLD_MAX_SOURCE_RECORDS:-2000}"
CHUNK_TARGET_DPO="${CHUNK_TARGET_DPO:-150}"
ESCONV_DPO_CANDIDATES="${ESCONV_DPO_CANDIDATES:-8}"
ESCONV_DPO_MAX_OUTPUT_TOKENS="${ESCONV_DPO_MAX_OUTPUT_TOKENS:-6144}"
ESCONV_MAX_REJECTED_POSTERIOR="${ESCONV_MAX_REJECTED_POSTERIOR:-0.55}"
ESCONV_GAP_RESCUE_MAX_REJECTED_POSTERIOR="${ESCONV_GAP_RESCUE_MAX_REJECTED_POSTERIOR:-0.65}"
ESCONV_GAP_RESCUE_MIN_SCORE_GAP="${ESCONV_GAP_RESCUE_MIN_SCORE_GAP:-0.30}"
EARLY_STOP_DAILY_DPO="${EARLY_STOP_DAILY_DPO:-1}"
EARLY_STOP_DAILY_DPO_BUFFER="${EARLY_STOP_DAILY_DPO_BUFFER:-300}"
MIN_POSTERIOR="${MIN_POSTERIOR:-0.72}"
MIN_CONTEXT_TURNS="${MIN_CONTEXT_TURNS:-1}"
PER_DIALOGUE_LIMIT="${PER_DIALOGUE_LIMIT:-3}"
SCORING_WORKERS="${SCORING_WORKERS:-4}"
DPO_WORKERS="${DPO_WORKERS:-2}"
DPO_CHUNK_JOBS="${DPO_CHUNK_JOBS:-1}"
ANALYSIS_MODEL="${ANALYSIS_MODEL:-gpt-5.4-pro}"
SCORING_MODEL="${SCORING_MODEL:-gpt-5.4}"
GENERATION_MODEL="${GENERATION_MODEL:-gpt-5.4-pro}"
LOCAL_MODEL_ID="${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
TRAIN_DEVICE_MAP="${TRAIN_DEVICE_MAP:-auto}"
TRAIN_MAX_MEMORY="${TRAIN_MAX_MEMORY:-0=46GiB,1=46GiB,cpu=0GiB}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
REBUILD_BAYES_MODEL="${REBUILD_BAYES_MODEL:-0}"
REBUILD_CHUNKS="${REBUILD_CHUNKS:-0}"
PIPELINE_HEARTBEAT_FILE="${PIPELINE_HEARTBEAT_FILE:-artifacts/run_logs/dpo_pipeline_${RUN_TAG}.heartbeat.json}"

ESCONV_CORPUS="${ESCONV_CORPUS:-data/esconv_analysis_corpus_${RUN_TAG}.jsonl}"
BAYES_MODEL="${BAYES_MODEL:-artifacts/bayes_models/generated_transition_bayes_model_esconv_${RUN_TAG}.json}"
CHUNK_DIR="${CHUNK_DIR:-artifacts/run_logs/${RUN_TAG}/chunks}"
DAILYDIALOG_DPO_DATA="${DAILYDIALOG_DPO_DATA:-artifacts/datasets/dailydialog_ja_dpo_preferences_${RUN_TAG}_daily.jsonl}"
ESCONV_GOLD_DPO_DATA="${ESCONV_GOLD_DPO_DATA:-artifacts/datasets/esconv_gold_ja_dpo_preferences_${RUN_TAG}.jsonl}"
FINAL_DPO_DATA="${FINAL_DPO_DATA:-artifacts/datasets/esconv_mixed_ja_dpo_preferences_${RUN_TAG}.jsonl}"
TRAINING_OUTPUT="${TRAINING_OUTPUT:-artifacts/training_runs/qwen35_bayes_dpo_lora_${RUN_TAG}_ep1_lr5e-6_r8_a16_no4bit}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
PIPELINE_LOG_DIR="${PIPELINE_LOG_DIR:-logs/dpo_pipeline/esconv/${RUN_DATE}}"
LOG_FILE="${PIPELINE_LOG_FILE:-${PIPELINE_LOG_DIR}/dpo_pipeline_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$PIPELINE_LOG_DIR" "$(dirname "$LOG_FILE")" "$CHUNK_DIR" data artifacts/bayes_models artifacts/scored_dialogues artifacts/datasets artifacts/training_runs artifacts/run_logs
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

write_heartbeat() {
  local stage="$1"
  mkdir -p "$(dirname "$PIPELINE_HEARTBEAT_FILE")"
  printf '{"timestamp":"%s","run_tag":"%s","stage":"%s"}\n' "$(date -Iseconds)" "$RUN_TAG" "$stage" > "$PIPELINE_HEARTBEAT_FILE"
}

echo "========================================"
echo "ESConv DPO chunked pipeline started at $(date)"
echo "run_tag: $RUN_TAG"
echo "log_file: $LOG_FILE"
echo "analysis_model: $ANALYSIS_MODEL"
echo "scoring_model: $SCORING_MODEL"
echo "generation_model: $GENERATION_MODEL"
echo "max_dialogues: $MAX_DIALOGUES"
echo "chunk_dialogues: $CHUNK_DIALOGUES"
echo "target_dpo_records: $TARGET_DPO_RECORDS"
echo "esconv_gold_target_records: $ESCONV_GOLD_TARGET_RECORDS"
echo "esconv_dpo_candidates: $ESCONV_DPO_CANDIDATES"
echo "esconv_dpo_max_output_tokens: $ESCONV_DPO_MAX_OUTPUT_TOKENS"
echo "esconv_max_rejected_posterior: $ESCONV_MAX_REJECTED_POSTERIOR"
echo "esconv_gap_rescue_max_rejected_posterior: $ESCONV_GAP_RESCUE_MAX_REJECTED_POSTERIOR"
echo "esconv_gap_rescue_min_score_gap: $ESCONV_GAP_RESCUE_MIN_SCORE_GAP"
echo "early_stop_daily_dpo: $EARLY_STOP_DAILY_DPO"
echo "early_stop_daily_dpo_buffer: $EARLY_STOP_DAILY_DPO_BUFFER"
echo "dpo_chunk_jobs: $DPO_CHUNK_JOBS"
echo "train_cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-}"
echo "train_device_map: ${TRAIN_DEVICE_MAP:-}"
echo "train_max_memory: ${TRAIN_MAX_MEMORY:-}"
echo "pytorch_cuda_alloc_conf: ${PYTORCH_CUDA_ALLOC_CONF:-}"
echo "heartbeat_file: $PIPELINE_HEARTBEAT_FILE"
echo "========================================"
write_heartbeat "started"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[STEP 0/10] GPU status before run"
  nvidia-smi
else
  echo "[STEP 0/10] nvidia-smi not found; training step still requires CUDA."
fi

echo "[STEP 1/10] Prepare ESConv analysis corpus"
write_heartbeat "prepare_esconv_corpus"
python3 -m tools.prepare_esconv_for_analysis \
  --split train \
  --max-conversations "$ESCONV_ANALYSIS_CONVERSATIONS" \
  --sampling stratified \
  --seed 42 \
  --output "$ESCONV_CORPUS"

echo "[STEP 2/10] Generate ESConv transition Bayes model"
write_heartbeat "generate_bayes_model"
if [ "$REBUILD_BAYES_MODEL" = "1" ] || [ ! -f "$BAYES_MODEL" ]; then
  python3 -m tools.analyze_esconv_corpus_transition_bayes \
    --input "$ESCONV_CORPUS" \
    --output "$BAYES_MODEL" \
    --model "$ANALYSIS_MODEL" \
    --strategy-guidance strong \
    --max-output-tokens 24000
else
  echo "Bayes model already exists; reuse: $BAYES_MODEL"
fi

dpo_pids=()
dpo_names=()

count_chunk_dpo_records() {
  python3 - "$CHUNK_DIR" <<'PY'
import glob
import json
import sys
from pathlib import Path

chunk_dir = Path(sys.argv[1])
seen = set()
count = 0
for path_text in sorted(glob.glob(str(chunk_dir / "*_dpo.jsonl"))):
    path = Path(path_text)
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            key = (row.get("source_dialogue_id"), row.get("turn_index"))
            if key in seen:
                continue
            seen.add(key)
            count += 1
print(count)
PY
}

daily_dpo_early_stop_reached() {
  if [ "$EARLY_STOP_DAILY_DPO" != "1" ]; then
    return 1
  fi
  local current
  local threshold
  current="$(count_chunk_dpo_records)"
  threshold=$((TARGET_DPO_RECORDS + EARLY_STOP_DAILY_DPO_BUFFER))
  echo "[STEP 5/10] daily DPO chunk records=$current early_stop_threshold=$threshold"
  [ "$current" -ge "$threshold" ]
}

wait_for_dpo_slot() {
  while [ "${#dpo_pids[@]}" -ge "$DPO_CHUNK_JOBS" ]; do
    local pid="${dpo_pids[0]}"
    local name="${dpo_names[0]}"
    echo "[STEP 5/10] waiting DPO chunk job: $name pid=$pid"
    write_heartbeat "wait_dpo_slot_${name}"
    wait "$pid"
    write_heartbeat "dpo_slot_released_${name}"
    dpo_pids=("${dpo_pids[@]:1}")
    dpo_names=("${dpo_names[@]:1}")
  done
}

launch_dpo_chunk() {
  local chunk_name="$1"
  local selected_data="$2"
  local dpo_data="$3"
  local skipped_data="${dpo_data%.jsonl}_skipped.jsonl"
  local selected_count
  selected_count="$(wc -l < "$selected_data" 2>/dev/null || echo 0)"
  if [ "$selected_count" -eq 0 ]; then
    echo "[STEP 5/10] skip DPO chunk $chunk_name because selected_count=0"
    return
  fi
  wait_for_dpo_slot
  echo "[STEP 5/10] launch DPO chunk $chunk_name selected=$selected_count"
  (
    python3 -m tools.translate_and_generate_dpo \
      --input "$selected_data" \
      --bayes-model "$BAYES_MODEL" \
      --output "$dpo_data" \
      --skipped-output "$skipped_data" \
      --model "$GENERATION_MODEL" \
      --score-model "$SCORING_MODEL" \
      --style-preset esconv_support \
      --candidates "$ESCONV_DPO_CANDIDATES" \
      --max-output-tokens "$ESCONV_DPO_MAX_OUTPUT_TOKENS" \
      --min-score-gap 0.25 \
      --min-chosen-posterior 0.70 \
      --max-rejected-posterior "$ESCONV_MAX_REJECTED_POSTERIOR" \
      --gap-rescue-max-rejected-posterior "$ESCONV_GAP_RESCUE_MAX_REJECTED_POSTERIOR" \
      --gap-rescue-min-score-gap "$ESCONV_GAP_RESCUE_MIN_SCORE_GAP" \
      --target-records "$CHUNK_TARGET_DPO" \
      --workers "$DPO_WORKERS" \
      --skip-sample-errors \
      --heartbeat-file "$PIPELINE_HEARTBEAT_FILE" \
      --heartbeat-stage-prefix "dpo_generation_${chunk_name}" \
      --seed 42
  ) &
  dpo_pids+=("$!")
  dpo_names+=("$chunk_name")
}

echo "[STEP 3/10] Chunked DailyDialog scoring and overlapped DPO generation"
write_heartbeat "chunk_loop_started"
chunk_index=0
for ((start=0; start<MAX_DIALOGUES; start+=CHUNK_DIALOGUES)); do
  chunk_name="$(printf 'chunk_%04d_%06d' "$chunk_index" "$start")"
  prepared_data="${CHUNK_DIR}/${chunk_name}_prepared.jsonl"
  scored_data="${CHUNK_DIR}/${chunk_name}_scored.jsonl"
  selected_data="${CHUNK_DIR}/${chunk_name}_selected.jsonl"
  dpo_data="${CHUNK_DIR}/${chunk_name}_dpo.jsonl"

  echo "----------------------------------------"
  echo "[STEP 3/10] chunk=$chunk_name start_dialogue=$start"
  write_heartbeat "prepare_${chunk_name}"
  if [ "$REBUILD_CHUNKS" = "1" ] || [ ! -f "$prepared_data" ]; then
    python3 -m tools.prepare_dailydialog_for_scoring \
      --split train \
      --start-dialogue "$start" \
      --max-dialogues "$CHUNK_DIALOGUES" \
      --max-context-turns "$MAX_CONTEXT_TURNS" \
      --output "$prepared_data"
  else
    echo "reuse prepared chunk: $prepared_data"
  fi

  prepared_count="$(wc -l < "$prepared_data" 2>/dev/null || echo 0)"
  if [ "$prepared_count" -eq 0 ]; then
    echo "[STEP 3/10] no prepared records; stop chunk loop"
    break
  fi

  echo "[STEP 4/10] score chunk=$chunk_name records=$prepared_count"
  write_heartbeat "score_${chunk_name}"
  python3 -m tools.score_dialogue_with_transition_bayes_model \
    --input "$prepared_data" \
    --bayes-model "$BAYES_MODEL" \
    --output "$scored_data" \
    --model "$SCORING_MODEL" \
    --workers "$SCORING_WORKERS" \
    --fallback-on-errors

  echo "[STEP 5/10] select high-posterior chunk=$chunk_name"
  write_heartbeat "select_${chunk_name}"
  python3 -m tools.extract_high_posterior_dialogues \
    --input "$scored_data" \
    --output "$selected_data" \
    --bayes-model "$BAYES_MODEL" \
    --min-posterior "$MIN_POSTERIOR" \
    --min-context-turns "$MIN_CONTEXT_TURNS" \
    --target-records "$TARGET_SELECTED_PER_CHUNK" \
    --per-dialogue-limit "$PER_DIALOGUE_LIMIT" \
    --require-preferred \
    --sort-by-selection

  launch_dpo_chunk "$chunk_name" "$selected_data" "$dpo_data"
  write_heartbeat "launched_dpo_${chunk_name}"
  if daily_dpo_early_stop_reached; then
    echo "[STEP 5/10] DailyDialog DPO候補が十分集まったため、追加チャンクのスコアリングを早期終了します。"
    write_heartbeat "daily_dpo_early_stop"
    break
  fi
  chunk_index=$((chunk_index + 1))
done

echo "[STEP 5/10] Wait for remaining DPO chunk jobs"
write_heartbeat "wait_dpo_chunks"
for index in "${!dpo_pids[@]}"; do
  echo "[STEP 5/10] waiting ${dpo_names[$index]} pid=${dpo_pids[$index]}"
  write_heartbeat "wait_dpo_chunk_${dpo_names[$index]}"
  wait "${dpo_pids[$index]}"
done
if daily_dpo_early_stop_reached; then
  echo "[STEP 5/10] Early stop confirmed after waiting DPO jobs."
fi

echo "[STEP 6/10] Merge DailyDialog DPO chunks"
write_heartbeat "merge_dpo_chunks"
python3 - <<PY
import glob
import json
import sys
from pathlib import Path

chunk_dir = Path("$CHUNK_DIR")
output = Path("$DAILYDIALOG_DPO_DATA")
target = int("$TARGET_DPO_RECORDS")
rows = []
seen = set()
skipped_invalid = 0
for path_text in sorted(glob.glob(str(chunk_dir / "*_dpo.jsonl"))):
    path = Path(path_text)
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped_invalid += 1
                print(f"skip invalid chunk DPO line: {path}:{line_number}: {exc}")
                continue
            if not isinstance(row, dict):
                skipped_invalid += 1
                print(f"skip non-object chunk DPO line: {path}:{line_number}")
                continue
            key = (row.get("source_dialogue_id"), row.get("turn_index"))
            if key in seen:
                continue
            seen.add(key)
            row.setdefault("metadata", {})["source_chunk_file"] = str(path)
            rows.append(row)
rows.sort(key=lambda item: float(item.get("score_gap", 0.0)), reverse=True)
rows = rows[:target]
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as file:
    for row in rows:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"merged_dpo_records: {len(rows)}")
print(f"merged_dpo_invalid_skipped: {skipped_invalid}")
print(f"merged_dpo_output: {output}")
if len(rows) < target:
    print(f"ERROR: target_dpo_records={target} but merged={len(rows)}")
    sys.exit(1)
PY

echo "[STEP 7/10] Build ESConv gold DPO records"
write_heartbeat "build_esconv_gold_dpo"
python3 -m tools.build_esconv_gold_dpo \
  --split train \
  --bayes-model "$BAYES_MODEL" \
  --output "$ESCONV_GOLD_DPO_DATA" \
  --model "$GENERATION_MODEL" \
  --score-model "$SCORING_MODEL" \
  --target-records "$ESCONV_GOLD_TARGET_RECORDS" \
  --max-source-records "$ESCONV_GOLD_MAX_SOURCE_RECORDS" \
  --max-context-turns "$MAX_CONTEXT_TURNS" \
  --max-output-tokens "$ESCONV_DPO_MAX_OUTPUT_TOKENS" \
  --candidates "$ESCONV_DPO_CANDIDATES" \
  --min-score-gap 0.25 \
  --min-chosen-posterior 0.70 \
  --max-rejected-posterior "$ESCONV_MAX_REJECTED_POSTERIOR" \
  --gap-rescue-max-rejected-posterior "$ESCONV_GAP_RESCUE_MAX_REJECTED_POSTERIOR" \
  --gap-rescue-min-score-gap "$ESCONV_GAP_RESCUE_MIN_SCORE_GAP" \
  --workers "$DPO_WORKERS" \
  --skip-sample-errors \
  --seed 42

echo "[STEP 8/10] Merge DailyDialog DPO and ESConv gold DPO"
write_heartbeat "merge_mixed_dpo"
python3 - <<PY
import json
import sys
from pathlib import Path

daily_path = Path("$DAILYDIALOG_DPO_DATA")
gold_path = Path("$ESCONV_GOLD_DPO_DATA")
output = Path("$FINAL_DPO_DATA")
daily_target = int("$TARGET_DPO_RECORDS")
gold_target = int("$ESCONV_GOLD_TARGET_RECORDS")

def read_jsonl(path: Path):
    rows = []
    skipped = 0
    if not path.exists():
        return rows, skipped
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped += 1
                print(f"skip invalid mixed source line: {path}:{line_number}: {exc}")
                continue
            if not isinstance(row, dict):
                skipped += 1
                print(f"skip non-object mixed source line: {path}:{line_number}")
                continue
            rows.append(row)
    return rows, skipped

daily_all, daily_skipped = read_jsonl(daily_path)
gold_all, gold_skipped = read_jsonl(gold_path)

def dedupe_rows(rows, source_name):
    best_by_key = {}
    duplicates = 0
    for row in rows:
        key = (row.get("source_dataset", source_name), row.get("source_dialogue_id"), row.get("turn_index"))
        current = best_by_key.get(key)
        if current is not None:
            duplicates += 1
        if current is None or float(row.get("score_gap", 0.0)) > float(current.get("score_gap", 0.0)):
            best_by_key[key] = row
    return list(best_by_key.values()), duplicates

daily_unique, daily_duplicates = dedupe_rows(daily_all, "DailyDialog")
gold_unique, gold_duplicates = dedupe_rows(gold_all, "ESConv")
daily_rows = sorted(daily_unique, key=lambda item: float(item.get("score_gap", 0.0)), reverse=True)[:daily_target]
gold_rows = sorted(gold_unique, key=lambda item: float(item.get("score_gap", 0.0)), reverse=True)[:gold_target]

rows = []
seen = set()
for source_name, source_rows in (("DailyDialog", daily_rows), ("ESConv", gold_rows)):
    for row in source_rows:
        key = (row.get("source_dataset", source_name), row.get("source_dialogue_id"), row.get("turn_index"))
        if key in seen:
            continue
        seen.add(key)
        row.setdefault("metadata", {})["mixture_source"] = source_name
        row.setdefault("metadata", {})["mixture_run_tag"] = "$RUN_TAG"
        rows.append(row)

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as file:
    for row in rows:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"daily_dpo_records: {len(daily_rows)}")
print(f"esconv_gold_records: {len(gold_rows)}")
print(f"mixed_dpo_records: {len(rows)}")
print(f"mixed_invalid_skipped: daily={daily_skipped} gold={gold_skipped}")
print(f"mixed_duplicate_skipped: daily={daily_duplicates} gold={gold_duplicates}")
print(f"mixed_dpo_output: {output}")
if len(daily_rows) < daily_target:
    print(f"ERROR: daily target={daily_target} but daily={len(daily_rows)}")
    sys.exit(1)
if len(gold_rows) < gold_target:
    print(f"ERROR: esconv gold target={gold_target} but gold={len(gold_rows)}")
    sys.exit(1)
if len(rows) < daily_target + gold_target:
    print(f"ERROR: mixed target={daily_target + gold_target} but mixed={len(rows)}")
    sys.exit(1)
PY

echo "[STEP 9/10] Dry-run training data validation"
write_heartbeat "training_dry_run"
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
  --seed 42 \
  --no-4bit \
  "${TRAIN_PLACEMENT_ARGS[@]}" \
  --dry-run

echo "[STEP 10/10] Train Qwen3.5 DPO LoRA"
write_heartbeat "training"
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
  --seed 42 \
  --no-4bit \
  "${TRAIN_PLACEMENT_ARGS[@]}"

echo "========================================"
echo "ESConv DPO chunked pipeline completed at $(date)"
write_heartbeat "completed"
echo "bayes_model: $BAYES_MODEL"
echo "dailydialog_dpo_data: $DAILYDIALOG_DPO_DATA"
echo "esconv_gold_dpo_data: $ESCONV_GOLD_DPO_DATA"
echo "dpo_data: $FINAL_DPO_DATA"
echo "training_output: $TRAINING_OUTPUT"
echo "log_file: $LOG_FILE"
echo "========================================"
