#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

RUN_TAG="${RUN_TAG:-mathdial_v6_instruction_discriminative_followup_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/mathdial_wildchat/evaluation_rechecks/${RUN_TAG}}"
PIPELINE="${MATHDIAL_INSTRUCTION_DISCRIMINATIVE_PIPELINE_SCRIPT:-$PROJECT_ROOT/scripts/run_mathdial_instruction_discriminative_v2_pipeline.sh}"
WATCHDOG_MAX_RESTARTS="${WATCHDOG_MAX_RESTARTS:-20}"
WATCHDOG_RESTART_DELAY_SECONDS="${WATCHDOG_RESTART_DELAY_SECONDS:-15}"
WATCHDOG_DIR="$OUTPUT_ROOT/watchdog"
WATCHDOG_LOG="$WATCHDOG_DIR/watchdog.log"
WATCHDOG_PID_FILE="$WATCHDOG_DIR/watchdog.pid"
mkdir -p "$WATCHDOG_DIR"
printf '%s\n' "$$" > "$WATCHDOG_PID_FILE"

child_pid=""

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$WATCHDOG_LOG"
}

stop_process_group() {
  local pid="$1"
  kill -TERM -- "-$pid" 2>/dev/null || true
  sleep 10
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
}

on_exit() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    stop_process_group "$child_pid"
  fi
  rm -f "$WATCHDOG_PID_FILE"
}
trap on_exit EXIT
trap 'exit 130' INT TERM

restart=0
while [[ "$restart" -le "$WATCHDOG_MAX_RESTARTS" ]]; do
  attempt=$((restart + 1))
  log "pipeline起動 attempt=$attempt run_tag=$RUN_TAG"
  setsid env \
    RUN_TAG="$RUN_TAG" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    WATCHDOG_ATTEMPT="$attempt" \
    PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}" \
    "$PIPELINE" &
  child_pid=$!
  set +e
  wait "$child_pid"
  status=$?
  set -e
  child_pid=""
  if [[ "$status" -eq 0 ]]; then
    log "pipeline完了"
    exit 0
  fi
  if [[ "$status" -eq 20 ]]; then
    log "研究条件または成果物の致命的エラーで終了しました。"
    exit 20
  fi
  restart=$((restart + 1))
  if [[ "$restart" -gt "$WATCHDOG_MAX_RESTARTS" ]]; then
    log "最大再起動回数を超えました: $WATCHDOG_MAX_RESTARTS"
    exit 1
  fi
  log "resume再起動 restart=$restart/$WATCHDOG_MAX_RESTARTS status=$status"
  sleep "$WATCHDOG_RESTART_DELAY_SECONDS"
done
