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

RUN_TAG="${RUN_TAG:-mathdial_wildchat_gpt56_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/mathdial_wildchat/runs/${RUN_TAG}}"
WATCHDOG_INTERVAL_SECONDS="${WATCHDOG_INTERVAL_SECONDS:-30}"
WATCHDOG_STALL_SECONDS="${WATCHDOG_STALL_SECONDS:-300}"
WATCHDOG_KILL_GRACE_SECONDS="${WATCHDOG_KILL_GRACE_SECONDS:-10}"
WATCHDOG_MAX_RESTARTS="${WATCHDOG_MAX_RESTARTS:-20}"
INITIAL_WORKERS="${WORKERS:-8}"
RESTART_WORKERS="${WATCHDOG_RESTART_WORKERS:-4}"
PIPELINE="${MATHDIAL_PIPELINE_SCRIPT:-$PROJECT_ROOT/scripts/run_mathdial_wildchat_pipeline.sh}"
WATCHDOG_DIR="$OUTPUT_ROOT/watchdog"
WATCHDOG_LOG="$WATCHDOG_DIR/watchdog.log"
WATCHDOG_PID_FILE="$WATCHDOG_DIR/watchdog.pid"
STATUS_FILE="$OUTPUT_ROOT/pipeline_status.json"
mkdir -p "$WATCHDOG_DIR"
printf '%s\n' "$$" > "$WATCHDOG_PID_FILE"

child_pid=""

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$WATCHDOG_LOG"
}

current_stage() {
  python3 - "$STATUS_FILE" <<'PY'
import json,pathlib,sys
path=pathlib.Path(sys.argv[1])
try:
    print(json.loads(path.read_text(encoding="utf-8")).get("stage", "startup"))
except Exception:
    print("startup")
PY
}

progress_signature() {
  python3 - "$OUTPUT_ROOT" <<'PY'
import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1])
parts=[]
if root.exists():
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".jsonl", ".json", ".log"}:
            continue
        if path.relative_to(root).parts[0] == "watchdog":
            continue
        try:
            stat=path.stat()
        except FileNotFoundError:
            continue
        parts.append(f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}")
print(hashlib.sha256("\n".join(parts).encode()).hexdigest())
PY
}

stage_is_stall_protected() {
  case "$1" in
    preprocess|build_basis|train|statistics|report) return 0 ;;
    *) return 1 ;;
  esac
}

stop_process_group() {
  local pid="$1"
  kill -TERM -- "-$pid" 2>/dev/null || true
  local waited=0
  while kill -0 "$pid" 2>/dev/null && [[ "$waited" -lt "$WATCHDOG_KILL_GRACE_SECONDS" ]]; do
    sleep 1
    waited=$((waited + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    log "TERM後も残存したためprocess groupをKILLします: pgid=$pid"
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

write_fatal_status() {
  local message="$1"
  python3 - "$STATUS_FILE" "$RUN_TAG" "$message" <<'PY'
import datetime,json,pathlib,sys
path=pathlib.Path(sys.argv[1]); path.parent.mkdir(parents=True,exist_ok=True)
payload={"timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),"state":"fatal","stage":"watchdog","message":sys.argv[3],"run_tag":sys.argv[2]}
path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
PY
}

restart=0
while [[ "$restart" -le "$WATCHDOG_MAX_RESTARTS" ]]; do
  attempt=$((restart + 1))
  workers="$INITIAL_WORKERS"
  [[ "$restart" -gt 0 ]] && workers="$RESTART_WORKERS"
  log "pipeline起動 attempt=$attempt workers=$workers run_tag=$RUN_TAG"
  setsid env \
    RUN_TAG="$RUN_TAG" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    WORKERS="$workers" \
    WATCHDOG_ATTEMPT="$attempt" \
    PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}" \
    "$PIPELINE" &
  child_pid=$!

  last_signature="$(progress_signature)"
  last_progress_epoch="$(date +%s)"
  stalled=0
  while kill -0 "$child_pid" 2>/dev/null; do
    sleep "$WATCHDOG_INTERVAL_SECONDS"
    signature="$(progress_signature)"
    stage="$(current_stage)"
    now="$(date +%s)"
    if [[ "$signature" != "$last_signature" ]]; then
      last_signature="$signature"
      last_progress_epoch="$now"
      log "heartbeat stage=$stage"
      continue
    fi
    if stage_is_stall_protected "$stage"; then
      last_progress_epoch="$now"
      continue
    fi
    idle=$((now - last_progress_epoch))
    if [[ "$idle" -ge "$WATCHDOG_STALL_SECONDS" ]]; then
      log "進捗停止を検出 stage=$stage idle=${idle}s; 再開可能地点まで停止します"
      stop_process_group "$child_pid"
      stalled=1
      break
    fi
  done

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
    message="研究整合性を保てない致命的エラーで終了しました。pipeline logを確認してください。"
    log "$message"
    write_fatal_status "$message"
    exit 20
  fi
  restart=$((restart + 1))
  if [[ "$restart" -gt "$WATCHDOG_MAX_RESTARTS" ]]; then
    message="最大再起動回数${WATCHDOG_MAX_RESTARTS}を超えました（last_status=$status stalled=$stalled）。"
    log "$message"
    write_fatal_status "$message"
    exit 1
  fi
  log "resume再起動を予定 restart=$restart/$WATCHDOG_MAX_RESTARTS last_status=$status stalled=$stalled"
  sleep 5
done
