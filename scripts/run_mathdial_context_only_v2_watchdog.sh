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

RUN_TAG="${RUN_TAG:-mathdial_wildchat_gpt56_v10_neutral_prompt_boundary_fixed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/mathdial_wildchat/runs/${RUN_TAG}}"
PIPELINE="${MATHDIAL_NEUTRAL_PROMPT_PIPELINE_SCRIPT:-${MATHDIAL_CONTEXT_ONLY_PIPELINE_SCRIPT:-$PROJECT_ROOT/scripts/run_mathdial_context_only_v2_pipeline.sh}}"
WATCHDOG_MAX_RESTARTS="${WATCHDOG_MAX_RESTARTS:-20}"
WATCHDOG_RESTART_DELAY_SECONDS="${WATCHDOG_RESTART_DELAY_SECONDS:-15}"
TRAIN_MAX_MEMORY="${TRAIN_MAX_MEMORY:-0=38GiB,1=46GiB,cpu=0GiB}"
WATCHDOG_OOM_TRAIN_MAX_MEMORY="${WATCHDOG_OOM_TRAIN_MAX_MEMORY:-0=36GiB,1=46GiB,cpu=0GiB}"
TRAIN_GPU0_MIN_HEADROOM_MIB="${TRAIN_GPU0_MIN_HEADROOM_MIB:-8192}"
TRAIN_CUDA_ALLOC_CONF="${TRAIN_CUDA_ALLOC_CONF:-expandable_segments:True}"
ALLOW_TRAIN_PLACEMENT_CONTINUATION="${ALLOW_TRAIN_PLACEMENT_CONTINUATION:-0}"
WATCHDOG_DIR="$OUTPUT_ROOT/watchdog"
WATCHDOG_LOG="$WATCHDOG_DIR/watchdog.log"
WATCHDOG_PID_FILE="$WATCHDOG_DIR/watchdog.pid"
OOM_MARKER="$OUTPUT_ROOT/training/OOM_DETECTED.json"
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
oom_fallback_used=0
while [[ "$restart" -le "$WATCHDOG_MAX_RESTARTS" ]]; do
  attempt=$((restart + 1))
  log "pipeline起動 attempt=$attempt run_tag=$RUN_TAG"
  setsid env \
    RUN_TAG="$RUN_TAG" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    WATCHDOG_ATTEMPT="$attempt" \
    TRAIN_MAX_MEMORY="$TRAIN_MAX_MEMORY" \
    TRAIN_GPU0_MIN_HEADROOM_MIB="$TRAIN_GPU0_MIN_HEADROOM_MIB" \
    TRAIN_CUDA_ALLOC_CONF="$TRAIN_CUDA_ALLOC_CONF" \
    ALLOW_TRAIN_PLACEMENT_CONTINUATION="$ALLOW_TRAIN_PLACEMENT_CONTINUATION" \
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
  if [[ -s "$OOM_MARKER" ]]; then
    if [[ "$oom_fallback_used" -eq 0 &&
      "$TRAIN_MAX_MEMORY" != "$WATCHDOG_OOM_TRAIN_MAX_MEMORY" ]]; then
      log "CUDA OOMを検出しました。checkpointを保持したままGPU 0のmodel budgetを下げます: ${TRAIN_MAX_MEMORY} -> ${WATCHDOG_OOM_TRAIN_MAX_MEMORY}"
      TRAIN_MAX_MEMORY="$WATCHDOG_OOM_TRAIN_MAX_MEMORY"
      ALLOW_TRAIN_PLACEMENT_CONTINUATION=1
      oom_fallback_used=1
      rm -f "$OOM_MARKER"
    else
      log "headroom拡大後もCUDA OOMが再発したため、安全に停止します。marker=$OOM_MARKER"
      exit 20
    fi
  fi
  restart=$((restart + 1))
  if [[ "$restart" -gt "$WATCHDOG_MAX_RESTARTS" ]]; then
    log "最大再起動回数を超えました: $WATCHDOG_MAX_RESTARTS"
    exit 1
  fi
  log "resume再起動 restart=$restart/$WATCHDOG_MAX_RESTARTS status=$status"
  sleep "$WATCHDOG_RESTART_DELAY_SECONDS"
done
