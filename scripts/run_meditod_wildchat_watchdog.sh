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

RUN_TAG="${RUN_TAG:-meditod_wildchat_gpt56_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/meditod_wildchat/runs/${RUN_TAG}}"
PIPELINE="${MEDITOD_PIPELINE_SCRIPT:-$PROJECT_ROOT/scripts/run_meditod_wildchat_pipeline.sh}"
INTERVAL="${WATCHDOG_INTERVAL_SECONDS:-30}"
STALL="${WATCHDOG_STALL_SECONDS:-300}"
GRACE="${WATCHDOG_KILL_GRACE_SECONDS:-10}"
MAX_RESTARTS="${WATCHDOG_MAX_RESTARTS:-20}"
INITIAL_WORKERS="${WORKERS:-4}"
RESTART_WORKERS="${WATCHDOG_RESTART_WORKERS:-4}"
WATCHDOG_DIR="$OUTPUT_ROOT/watchdog"
LOG="$WATCHDOG_DIR/watchdog.log"
PID_FILE="$WATCHDOG_DIR/watchdog.pid"
mkdir -p "$WATCHDOG_DIR"
printf '%s\n' "$$" > "$PID_FILE"
child=""

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"; }
stage() { python3 - "$OUTPUT_ROOT/pipeline_status.json" <<'PY'
import json,pathlib,sys
try: print(json.loads(pathlib.Path(sys.argv[1]).read_text()).get("stage","startup"))
except Exception: print("startup")
PY
}
signature() { python3 - "$OUTPUT_ROOT" <<'PY'
import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1]); parts=[]
if root.exists():
 for path in sorted(root.rglob("*")):
  if path.is_file() and path.suffix in {".jsonl",".json",".csv",".log"} and "watchdog" not in path.parts:
   stat=path.stat(); parts.append(f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}")
print(hashlib.sha256("\n".join(parts).encode()).hexdigest())
PY
}
protected() {
  # 成果物行数が一定時間増えないことが正常なstageだけをstall判定から除外する。
  case "$1" in preprocess|build_basis|select_data|train|statistics|report|prepare_user_eval) return 0;; *) return 1;; esac
}
stop_group() {
  kill -TERM -- "-$1" 2>/dev/null || true
  local waited=0
  while kill -0 "$1" 2>/dev/null && (( waited < GRACE )); do sleep 1; waited=$((waited+1)); done
  kill -KILL -- "-$1" 2>/dev/null || true
}
cleanup() { [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null && stop_group "$child"; rm -f "$PID_FILE"; }
trap cleanup EXIT
trap 'exit 130' INT TERM

restart=0
while (( restart <= MAX_RESTARTS )); do
  workers="$INITIAL_WORKERS"; (( restart > 0 )) && workers="$RESTART_WORKERS"
  log "pipeline起動 attempt=$((restart+1)) workers=$workers run_tag=$RUN_TAG"
  setsid env RUN_TAG="$RUN_TAG" OUTPUT_ROOT="$OUTPUT_ROOT" WORKERS="$workers" PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}" "$PIPELINE" &
  child=$!
  previous="$(signature)"; progressed="$(date +%s)"
  while kill -0 "$child" 2>/dev/null; do
    sleep "$INTERVAL"
    current="$(signature)"; current_stage="$(stage)"; now="$(date +%s)"
    if [[ "$current" != "$previous" ]]; then previous="$current"; progressed="$now"; log "heartbeat stage=$current_stage"; continue; fi
    protected "$current_stage" && { progressed="$now"; continue; }
    if (( now - progressed >= STALL )); then log "stall検出 stage=$current_stage; process groupを停止"; stop_group "$child"; break; fi
  done
  set +e; wait "$child"; status=$?; set -e; child=""
  (( status == 0 )) && { log "pipeline完了"; exit 0; }
  (( status == 20 )) && { log "研究整合性上の致命的エラー。pipeline logを確認してください。"; exit 20; }
  restart=$((restart+1))
  (( restart > MAX_RESTARTS )) && { log "最大再起動回数を超えました。"; exit 1; }
  log "resume再起動 restart=$restart/$MAX_RESTARTS status=$status"
  sleep 5
done
