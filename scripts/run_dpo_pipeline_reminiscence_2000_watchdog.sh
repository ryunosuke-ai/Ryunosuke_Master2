#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-reminiscence_5000_to_2000}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-${SCRIPT_DIR}/run_dpo_pipeline_reminiscence_2000.sh}"
SCORED_DATA="artifacts/scored_dialogues/dailydialog_transition_scored_${RUN_TAG}.jsonl"
DPO_DATA="artifacts/datasets/dailydialog_ja_dpo_preferences_${RUN_TAG}.jsonl"
TARGET_DPO_RECORDS="${TARGET_DPO_RECORDS:-2000}"
PIPELINE_HEARTBEAT_FILE="${PIPELINE_HEARTBEAT_FILE:-artifacts/run_logs/dpo_pipeline_${RUN_TAG}.heartbeat.json}"
WATCHDOG_LOG_DIR="${WATCHDOG_LOG_DIR:-logs/dpo_pipeline_watchdog/reminiscence}"
PIPELINE_LOG_DIR="${PIPELINE_LOG_DIR:-logs/dpo_pipeline/reminiscence}"
WATCHDOG_LOG="${WATCHDOG_LOG_DIR}/dpo_pipeline_${RUN_TAG}_watchdog_$(date +%Y%m%d_%H%M%S).log"
AUDIT_LOG="${AUDIT_LOG:-audit_log.md}"
STALL_SECONDS="${STALL_SECONDS:-300}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
INITIAL_SCORING_WORKERS="${SCORING_WORKERS:-4}"
RESTART_SCORING_WORKERS="${RESTART_SCORING_WORKERS:-4}"
STARTED_PIPELINE_PID=""

mkdir -p "$WATCHDOG_LOG_DIR" "$PIPELINE_LOG_DIR" artifacts/scored_dialogues artifacts/datasets artifacts/run_logs

echo "WARNING: this watchdog is deprecated for the current presentation. Use ESConv watchdog scripts unless reproducing the old reminiscence run."

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "$WATCHDOG_LOG"
}

append_audit() {
  {
    echo
    echo "## $(date '+%Y-%m-%d %H:%M:%S %Z'): DPOパイプラインwatchdogイベント"
    echo
    echo "- 対象ファイル:"
    echo "  - \`$PIPELINE_SCRIPT\`"
    echo "  - \`$SCORED_DATA\`"
    echo "  - \`$DPO_DATA\`"
    echo "  - \`$PIPELINE_HEARTBEAT_FILE\`"
    echo "  - \`$WATCHDOG_LOG\`"
    echo "- 実行した操作:"
    echo "  - $1"
    echo "- なぜその操作が必要だったか:"
    echo "  - 席を離している間にAPI待ちやcontent_filter再試行で処理が停止したままになるリスクを下げるため。"
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
    echo "- リスクや注意点:"
    echo "  - 再起動時、実行中のパイプラインプロセスグループへTERM/KILLを送る。保存済みJSONLは再実行時にスキップされる。"
    echo "  - 学習・Oracle評価ステップ中はスコア済みJSONLの行数が増えないため、DPO目標件数到達後は停止判定しない。"
  } >> "$AUDIT_LOG"
}

scored_line_count() {
  if [ -f "$SCORED_DATA" ]; then
    wc -l < "$SCORED_DATA"
  else
    echo 0
  fi
}

dpo_line_count() {
  if [ -f "$DPO_DATA" ]; then
    wc -l < "$DPO_DATA"
  else
    echo 0
  fi
}

heartbeat_mtime() {
  if [ -f "$PIPELINE_HEARTBEAT_FILE" ]; then
    stat -c %Y "$PIPELINE_HEARTBEAT_FILE" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

dpo_target_reached() {
  local current_dpo="$1"
  [ "$TARGET_DPO_RECORDS" -gt 0 ] && [ "$current_dpo" -ge "$TARGET_DPO_RECORDS" ]
}

expected_total() {
  local prepared="data/dailydialog_for_scoring_${RUN_TAG}.jsonl"
  if [ -f "$prepared" ]; then
    wc -l < "$prepared"
  else
    echo 0
  fi
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
      exec "$4"
    ' bash "$scoring_workers" "$PIPELINE_LOG_DIR" "$PIPELINE_HEARTBEAT_FILE" "$PIPELINE_SCRIPT" >> "$WATCHDOG_LOG" 2>&1 &
  else
    (
      export SCORING_WORKERS="$scoring_workers"
      export PIPELINE_LOG_DIR="$PIPELINE_LOG_DIR"
      export PIPELINE_HEARTBEAT_FILE="$PIPELINE_HEARTBEAT_FILE"
      exec "$PIPELINE_SCRIPT"
    ) >> "$WATCHDOG_LOG" 2>&1 &
  fi
  STARTED_PIPELINE_PID="$!"
}

append_audit "watchdog付きでDPOパイプラインを開始した。"
log "watchdog started"
log "run_tag=$RUN_TAG stall_seconds=$STALL_SECONDS max_restarts=$MAX_RESTARTS"
log "pipeline_script=$PIPELINE_SCRIPT"
log "target_dpo_records=$TARGET_DPO_RECORDS"

restart_count=0
scoring_workers="$INITIAL_SCORING_WORKERS"

while [ "$restart_count" -le "$MAX_RESTARTS" ]; do
  log "starting pipeline attempt=$((restart_count + 1)) SCORING_WORKERS=$scoring_workers"
  start_pipeline "$scoring_workers"
  pipeline_pid="$STARTED_PIPELINE_PID"
  log "pipeline pid=$pipeline_pid"

  last_scored_count="$(scored_line_count)"
  last_dpo_count="$(dpo_line_count)"
  last_heartbeat_mtime="$(heartbeat_mtime)"
  last_progress_ts="$(date +%s)"

  while kill -0 "$pipeline_pid" >/dev/null 2>&1; do
    sleep 30
    current_scored_count="$(scored_line_count)"
    current_dpo_count="$(dpo_line_count)"
    current_heartbeat_mtime="$(heartbeat_mtime)"
    total="$(expected_total)"
    now="$(date +%s)"

    if [ "$current_scored_count" != "$last_scored_count" ] || [ "$current_dpo_count" != "$last_dpo_count" ]; then
      log "progress scored_lines=$current_scored_count expected_total=$total dpo_lines=$current_dpo_count target_dpo=$TARGET_DPO_RECORDS"
      last_scored_count="$current_scored_count"
      last_dpo_count="$current_dpo_count"
      last_heartbeat_mtime="$current_heartbeat_mtime"
      last_progress_ts="$now"
      continue
    fi

    if [ "$current_heartbeat_mtime" != "$last_heartbeat_mtime" ]; then
      log "heartbeat updated mtime=$current_heartbeat_mtime scored_lines=$current_scored_count dpo_lines=$current_dpo_count"
      last_heartbeat_mtime="$current_heartbeat_mtime"
    fi

    if dpo_target_reached "$current_dpo_count"; then
      continue
    fi

    if [ "$total" -gt 0 ] && [ "$current_scored_count" -ge "$total" ]; then
      continue
    fi

    if [ $((now - last_progress_ts)) -ge "$STALL_SECONDS" ]; then
      terminate_pipeline "$pipeline_pid"
      wait "$pipeline_pid" >/dev/null 2>&1 || true
      restart_count=$((restart_count + 1))
      scoring_workers="$RESTART_SCORING_WORKERS"
      append_audit "スコアリング行数とDPO行数が${STALL_SECONDS}秒間増えなかったため、パイプラインを再起動した。再起動回数: $restart_count"
      continue 2
    fi
  done

  if wait "$pipeline_pid"; then
    log "pipeline completed successfully"
    append_audit "watchdog監視下のDPOパイプラインが正常終了した。"
    exit 0
  fi

  current_scored_count="$(scored_line_count)"
  current_dpo_count="$(dpo_line_count)"
  total="$(expected_total)"
  if [ "$total" -gt 0 ] && [ "$current_scored_count" -ge "$total" ] && ! dpo_target_reached "$current_dpo_count"; then
    log "pipeline stopped after scoring all rows but DPO target was not reached: scored_lines=$current_scored_count dpo_lines=$current_dpo_count target_dpo=$TARGET_DPO_RECORDS"
    append_audit "全スコアリング完了後もDPO目標件数に達しなかったため、watchdog再起動ではなく失敗終了した。"
    exit 1
  fi

  restart_count=$((restart_count + 1))
  if [ "$restart_count" -gt "$MAX_RESTARTS" ]; then
    log "max restarts exceeded: $MAX_RESTARTS"
    append_audit "最大再起動回数を超えたため、watchdogを終了した。"
    exit 1
  fi
  scoring_workers="$RESTART_SCORING_WORKERS"
  append_audit "パイプラインが非0終了したため再起動した。再起動回数: $restart_count"
done
