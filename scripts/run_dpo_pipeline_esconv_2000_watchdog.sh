#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-esconv_support_mixed_8000_to_2500}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-${SCRIPT_DIR}/run_dpo_pipeline_esconv_2000_chunked.sh}"
CHUNK_DIR="${CHUNK_DIR:-artifacts/run_logs/${RUN_TAG}/chunks}"
DPO_DATA="${FINAL_DPO_DATA:-artifacts/datasets/esconv_mixed_ja_dpo_preferences_${RUN_TAG}.jsonl}"
ESCONV_GOLD_DPO_DATA="${ESCONV_GOLD_DPO_DATA:-artifacts/datasets/esconv_gold_ja_dpo_preferences_${RUN_TAG}.jsonl}"
TARGET_DPO_RECORDS="${TARGET_DPO_RECORDS:-2000}"
ESCONV_GOLD_TARGET_RECORDS="${ESCONV_GOLD_TARGET_RECORDS:-500}"
WATCHDOG_TARGET_DPO_RECORDS="${WATCHDOG_TARGET_DPO_RECORDS:-$((TARGET_DPO_RECORDS + ESCONV_GOLD_TARGET_RECORDS))}"
PIPELINE_HEARTBEAT_FILE="${PIPELINE_HEARTBEAT_FILE:-artifacts/run_logs/dpo_pipeline_${RUN_TAG}.heartbeat.json}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
WATCHDOG_LOG_DIR="${WATCHDOG_LOG_DIR:-logs/dpo_pipeline_watchdog/esconv/${RUN_DATE}}"
PIPELINE_LOG_DIR="${PIPELINE_LOG_DIR:-logs/dpo_pipeline/esconv/${RUN_DATE}}"
PIPELINE_LOG_FILE="${PIPELINE_LOG_FILE:-${PIPELINE_LOG_DIR}/dpo_pipeline_${RUN_TAG}_${RUN_STAMP}.log}"
WATCHDOG_LOG="${WATCHDOG_LOG_DIR}/dpo_pipeline_${RUN_TAG}_watchdog_${RUN_STAMP}.log"
AUDIT_LOG="${AUDIT_LOG:-audit_log.md}"
STALL_SECONDS="${STALL_SECONDS:-300}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
INITIAL_SCORING_WORKERS="${SCORING_WORKERS:-4}"
RESTART_SCORING_WORKERS="${RESTART_SCORING_WORKERS:-4}"
STARTED_PIPELINE_PID=""

mkdir -p "$WATCHDOG_LOG_DIR" "$PIPELINE_LOG_DIR" "$CHUNK_DIR" artifacts/datasets artifacts/run_logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "$WATCHDOG_LOG"
}

append_audit() {
  {
    echo
    echo "## $(date '+%Y-%m-%d %H:%M:%S %Z'): ESConv DPOパイプラインwatchdogイベント"
    echo
    echo "- 対象ファイル:"
    echo "  - \`$PIPELINE_SCRIPT\`"
    echo "  - \`$CHUNK_DIR\`"
    echo "  - \`$DPO_DATA\`"
    echo "  - \`$ESCONV_GOLD_DPO_DATA\`"
    echo "  - \`$PIPELINE_HEARTBEAT_FILE\`"
    echo "  - \`$WATCHDOG_LOG\`"
    echo "- 実行した操作:"
    echo "  - $1"
    echo "- なぜその操作が必要だったか:"
    echo "  - ESConv用の長時間DPO生成・スコアリング中にAPI待ちやcontent_filter再試行で停止し続けるリスクを下げるため。"
    echo "- 代替案があったか:"
    echo "  - 手動監視する案があったが、長時間不在時に停止を検出できないため採用しなかった。"
    echo "- 実行したコマンド:"
    echo "  - \`$0\`"
    echo "- 変更前後の要約:"
    echo "  - 進捗停止判定秒数: $STALL_SECONDS"
    echo "  - 最大再起動回数: $MAX_RESTARTS"
    echo "  - 初回SCORING_WORKERS: $INITIAL_SCORING_WORKERS"
    echo "  - 再起動時SCORING_WORKERS: $RESTART_SCORING_WORKERS"
    echo "  - TARGET_DPO_RECORDS: $TARGET_DPO_RECORDS"
    echo "  - ESCONV_GOLD_TARGET_RECORDS: $ESCONV_GOLD_TARGET_RECORDS"
    echo "  - WATCHDOG_TARGET_DPO_RECORDS: $WATCHDOG_TARGET_DPO_RECORDS"
    echo "- リスクや注意点:"
    echo "  - 再起動時、実行中のパイプラインプロセスグループへTERM/KILLを送る。保存済みJSONLは再実行時に再利用される。"
    echo "  - ベイズモデル生成とQwen学習中はJSONL行数が増えないため、heartbeatのstageも参考にする。"
  } >> "$AUDIT_LOG"
}

line_count_glob() {
  local pattern="$1"
  python3 - "$pattern" <<'PY'
import glob
import sys

total = 0
for path in glob.glob(sys.argv[1]):
    try:
        with open(path, encoding="utf-8") as file:
            total += sum(1 for line in file if line.strip())
    except FileNotFoundError:
        pass
print(total)
PY
}

final_dpo_line_count() {
  if [ -f "$DPO_DATA" ]; then
    python3 - "$DPO_DATA" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
seen = set()
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
        if not all(key in row for key in ("prompt", "chosen", "rejected", "source_dialogue_id", "turn_index")):
            continue
        source_dataset = row.get("source_dataset") or row.get("metadata", {}).get("mixture_source", "")
        seen.add((source_dataset, row.get("source_dialogue_id"), row.get("turn_index")))
print(len(seen))
PY
  else
    echo 0
  fi
}

progress_line_count() {
  local scored_count
  local chunk_dpo_count
  local esconv_gold_count
  local final_dpo_count
  scored_count="$(line_count_glob "${CHUNK_DIR}/*_scored.jsonl")"
  chunk_dpo_count="$(line_count_glob "${CHUNK_DIR}/*_dpo.jsonl")"
  if [ -f "$ESCONV_GOLD_DPO_DATA" ]; then
    esconv_gold_count="$(wc -l < "$ESCONV_GOLD_DPO_DATA")"
  else
    esconv_gold_count=0
  fi
  final_dpo_count="$(final_dpo_line_count)"
  echo $((scored_count + chunk_dpo_count + esconv_gold_count + final_dpo_count))
}

heartbeat_mtime() {
  if [ -f "$PIPELINE_HEARTBEAT_FILE" ]; then
    stat -c %Y "$PIPELINE_HEARTBEAT_FILE" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

heartbeat_stage() {
  if [ ! -f "$PIPELINE_HEARTBEAT_FILE" ]; then
    echo ""
    return
  fi
  python3 - "$PIPELINE_HEARTBEAT_FILE" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as file:
        payload = json.load(file)
except Exception:
    print("")
else:
    print(str(payload.get("stage", "")))
PY
}

dpo_target_reached() {
  local current_dpo="$1"
  [ "$WATCHDOG_TARGET_DPO_RECORDS" -gt 0 ] && [ "$current_dpo" -ge "$WATCHDOG_TARGET_DPO_RECORDS" ]
}

protected_stage() {
  case "$1" in
    generate_bayes_model|merge_dpo_chunks|merge_mixed_dpo|training_dry_run|training|completed)
      return 0
      ;;
    wait_dpo_*|launched_dpo_*|dpo_generation_*|daily_dpo_early_stop)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

terminate_pipeline() {
  local pid="$1"
  if kill -0 "$pid" >/dev/null 2>&1; then
    log "no progress for ${STALL_SECONDS}s; terminating pipeline process group pid=$pid"
    kill -TERM -- "-$pid" >/dev/null 2>&1 || kill -TERM "$pid" >/dev/null 2>&1 || true
    sleep 10
    if kill -0 "$pid" >/dev/null 2>&1; then
      log "pipeline process group pid=$pid still alive; sending SIGKILL"
      kill -KILL -- "-$pid" >/dev/null 2>&1 || kill -KILL "$pid" >/dev/null 2>&1 || true
    fi
  fi
}

start_pipeline() {
  local scoring_workers="$1"
  if command -v setsid >/dev/null 2>&1; then
    setsid bash -c '
      set -euo pipefail
      export SCORING_WORKERS="$1"
      export PIPELINE_LOG_DIR="$2"
      export PIPELINE_HEARTBEAT_FILE="$3"
      export PIPELINE_LOG_FILE="$4"
      exec "$5"
    ' bash "$scoring_workers" "$PIPELINE_LOG_DIR" "$PIPELINE_HEARTBEAT_FILE" "$PIPELINE_LOG_FILE" "$PIPELINE_SCRIPT" >> "$WATCHDOG_LOG" 2>&1 &
  else
    (
      export SCORING_WORKERS="$scoring_workers"
      export PIPELINE_LOG_DIR="$PIPELINE_LOG_DIR"
      export PIPELINE_HEARTBEAT_FILE="$PIPELINE_HEARTBEAT_FILE"
      export PIPELINE_LOG_FILE="$PIPELINE_LOG_FILE"
      exec "$PIPELINE_SCRIPT"
    ) >> "$WATCHDOG_LOG" 2>&1 &
  fi
  STARTED_PIPELINE_PID="$!"
}

append_audit "watchdog付きでESConv DPOパイプラインを開始した。"
log "watchdog started"
log "run_tag=$RUN_TAG stall_seconds=$STALL_SECONDS max_restarts=$MAX_RESTARTS"
log "pipeline_script=$PIPELINE_SCRIPT"
log "chunk_dir=$CHUNK_DIR"
log "pipeline_log_file=$PIPELINE_LOG_FILE"
log "target_dpo_records=$TARGET_DPO_RECORDS esconv_gold_target_records=$ESCONV_GOLD_TARGET_RECORDS watchdog_target_dpo_records=$WATCHDOG_TARGET_DPO_RECORDS"

restart_count=0
scoring_workers="$INITIAL_SCORING_WORKERS"

while [ "$restart_count" -le "$MAX_RESTARTS" ]; do
  log "starting pipeline attempt=$((restart_count + 1)) SCORING_WORKERS=$scoring_workers"
  start_pipeline "$scoring_workers"
  pipeline_pid="$STARTED_PIPELINE_PID"
  log "pipeline pid=$pipeline_pid"

  last_progress_count="$(progress_line_count)"
  last_heartbeat_mtime="$(heartbeat_mtime)"
  last_progress_ts="$(date +%s)"

  while kill -0 "$pipeline_pid" >/dev/null 2>&1; do
    sleep 30
    current_progress_count="$(progress_line_count)"
    current_heartbeat_mtime="$(heartbeat_mtime)"
    current_dpo_count="$(final_dpo_line_count)"
    current_stage="$(heartbeat_stage)"
    now="$(date +%s)"

    if [ "$current_progress_count" != "$last_progress_count" ]; then
      log "progress total_jsonl_lines=$current_progress_count final_dpo_lines=$current_dpo_count target_dpo=$WATCHDOG_TARGET_DPO_RECORDS stage=$current_stage"
      last_progress_count="$current_progress_count"
      last_heartbeat_mtime="$current_heartbeat_mtime"
      last_progress_ts="$now"
      continue
    fi

    if [ "$current_heartbeat_mtime" != "$last_heartbeat_mtime" ]; then
      log "heartbeat updated mtime=$current_heartbeat_mtime final_dpo_lines=$current_dpo_count stage=$current_stage"
      last_heartbeat_mtime="$current_heartbeat_mtime"
      last_progress_ts="$now"
    fi

    if dpo_target_reached "$current_dpo_count"; then
      continue
    fi

    if protected_stage "$current_stage"; then
      continue
    fi

    if [ $((now - last_progress_ts)) -ge "$STALL_SECONDS" ]; then
      terminate_pipeline "$pipeline_pid"
      wait "$pipeline_pid" >/dev/null 2>&1 || true
      restart_count=$((restart_count + 1))
      scoring_workers="$RESTART_SCORING_WORKERS"
      append_audit "ESConvパイプラインのJSONL行数が${STALL_SECONDS}秒間増えなかったため、パイプラインを再起動した。再起動回数: $restart_count"
      continue 2
    fi
  done

  if wait "$pipeline_pid"; then
    log "pipeline completed successfully"
    append_audit "watchdog監視下のESConv DPOパイプラインが正常終了した。"
    exit 0
  fi

  restart_count=$((restart_count + 1))
  if [ "$restart_count" -gt "$MAX_RESTARTS" ]; then
    log "max restarts exceeded: $MAX_RESTARTS"
    append_audit "ESConv DPOパイプラインの最大再起動回数を超えたため、watchdogを終了した。"
    exit 1
  fi
  scoring_workers="$RESTART_SCORING_WORKERS"
  append_audit "ESConv DPOパイプラインが非0終了したため再起動した。再起動回数: $restart_count"
done

log "watchdog loop ended without successful pipeline completion; treating as failure"
append_audit "ESConv DPOパイプラインwatchdogの再起動ループが正常終了なしに終了したため、失敗終了した。"
exit 1
