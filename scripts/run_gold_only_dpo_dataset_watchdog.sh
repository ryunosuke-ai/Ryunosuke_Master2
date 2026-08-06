#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
DATASET="${DATASET:?DATASETを指定してください}"
RUN_TAG="${RUN_TAG:-gold_only_dpo500_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/gold_only_dpo/runs/$RUN_TAG/$DATASET}"
PIPELINE="${GOLD_ONLY_PIPELINE_SCRIPT:-$PROJECT_ROOT/scripts/run_gold_only_dpo_dataset_pipeline.sh}"
MAX_RESTARTS="${WATCHDOG_MAX_RESTARTS:-20}"
INTERVAL="${WATCHDOG_INTERVAL_SECONDS:-30}"
STALL_SECONDS="${WATCHDOG_STALL_SECONDS:-900}"
RESTART_DELAY="${WATCHDOG_RESTART_DELAY_SECONDS:-15}"
WATCHDOG_DIR="$OUTPUT_ROOT/watchdog"
mkdir -p "$WATCHDOG_DIR"
printf '%s\n' "$$" > "$WATCHDOG_DIR/watchdog.pid"
child_pid=""

stop_child() {
  [[ -z "$child_pid" ]] && return
  kill -TERM -- "-$child_pid" 2>/dev/null || true
  for _ in {1..10}; do kill -0 "$child_pid" 2>/dev/null || return 0; sleep 1; done
  kill -KILL -- "-$child_pid" 2>/dev/null || true
  return 0
}

progress_signature() {
  python3 - "$OUTPUT_ROOT" <<'PY'
import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1]); parts=[]
if root.exists():
 for path in sorted(root.rglob('*')):
  if not path.is_file() or 'watchdog' in path.parts or 'logs' in path.parts: continue
  stat=path.stat(); parts.append(f'{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}')
print(hashlib.sha256('\n'.join(parts).encode()).hexdigest())
PY
}

current_stage() {
  python3 - "$OUTPUT_ROOT/pipeline_status.json" <<'PY'
import json,pathlib,sys
try: print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')).get('stage','startup'))
except Exception: print('startup')
PY
}

stall_protected() {
  case "$1" in prepare_data|train|statistics|report) return 0 ;; *) return 1 ;; esac
}
trap 'stop_child; rm -f "$WATCHDOG_DIR/watchdog.pid"' EXIT
trap 'exit 130' INT TERM

for ((restart=0; restart<=MAX_RESTARTS; restart++)); do
  echo "[$(date --iso-8601=seconds)] Gold-only pipeline start dataset=$DATASET attempt=$((restart+1))" | tee -a "$WATCHDOG_DIR/watchdog.log"
  setsid env DATASET="$DATASET" RUN_TAG="$RUN_TAG" OUTPUT_ROOT="$OUTPUT_ROOT" WATCHDOG_ATTEMPT="$((restart+1))" \
    PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}" \
    "$PIPELINE" &
  child_pid=$!
  last_signature="$(progress_signature)"; last_progress="$(date +%s)"; stalled=0
  while kill -0 "$child_pid" 2>/dev/null; do
    sleep "$INTERVAL"
    signature="$(progress_signature)"; now="$(date +%s)"; stage="$(current_stage)"
    if [[ "$signature" != "$last_signature" ]]; then
      last_signature="$signature"; last_progress="$now"
      echo "[$(date --iso-8601=seconds)] heartbeat dataset=$DATASET stage=$stage" | tee -a "$WATCHDOG_DIR/watchdog.log"
    elif stall_protected "$stage"; then
      last_progress="$now"
    elif (( now - last_progress >= STALL_SECONDS )); then
      echo "[$(date --iso-8601=seconds)] stall dataset=$DATASET stage=$stage" | tee -a "$WATCHDOG_DIR/watchdog.log"
      stop_child; stalled=1; break
    fi
  done
  set +e; wait "$child_pid"; status=$?; set -e; child_pid=""
  (( status == 0 )) && exit 0
  (( status == 20 )) && { echo "研究整合性に関わる致命的エラーです。" | tee -a "$WATCHDOG_DIR/watchdog.log"; exit 20; }
  (( restart == MAX_RESTARTS )) && break
  echo "[$(date --iso-8601=seconds)] resume after status=$status stalled=$stalled" | tee -a "$WATCHDOG_DIR/watchdog.log"
  sleep "$RESTART_DELAY"
done
echo "watchdog最大再起動回数を超えました。" >&2
exit 1
