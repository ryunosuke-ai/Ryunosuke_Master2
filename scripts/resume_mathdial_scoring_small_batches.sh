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

RUN_TAG="${RUN_TAG:?RUN_TAGを指定してください。}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/mathdial_wildchat/runs/${RUN_TAG}}"
SCORING_BATCH_RECORDS="${SCORING_BATCH_RECORDS:-3000}"
ORIGINAL_SCORING_BATCH_RECORDS="${ORIGINAL_SCORING_BATCH_RECORDS:-20000}"
SELECTION_POOL_COUNT="${SELECTION_POOL_COUNT:-5000}"
DPO_MAX_SOURCE_CHARACTERS="${DPO_MAX_SOURCE_CHARACTERS:-16000}"
DPO_MAX_OUTPUT_TOKENS="${DPO_MAX_OUTPUT_TOKENS:-6144}"
WORKERS="${WORKERS:-4}"
SCORING_REPAIR_WORKERS="${SCORING_REPAIR_WORKERS:-4}"
SCORING_REPAIR_ROUNDS="${SCORING_REPAIR_ROUNDS:-2}"
SCORING_REQUESTS_PER_MINUTE="${SCORING_REQUESTS_PER_MINUTE:-120}"
SCORING_REPAIR_REQUESTS_PER_MINUTE="${SCORING_REPAIR_REQUESTS_PER_MINUTE:-90}"
SCORING_RATE_LIMIT_MAX_RETRIES="${SCORING_RATE_LIMIT_MAX_RETRIES:-6}"
SCORING_RATE_LIMIT_BACKOFF_SECONDS="${SCORING_RATE_LIMIT_BACKOFF_SECONDS:-15}"
export SCORING_REQUESTS_PER_MINUTE SCORING_REPAIR_REQUESTS_PER_MINUTE
export SCORING_RATE_LIMIT_MAX_RETRIES SCORING_RATE_LIMIT_BACKOFF_SECONDS
SCORING_MODEL="${MATHDIAL_SCORING_LLM_MODEL:-${AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT:-gpt-5.6-terra}}"
SCORING_PRESET="${SCORING_PRESET:-mathdial_tutoring}"
INVALID_OBSERVATION_RETRIES="${INVALID_OBSERVATION_RETRIES:-2}"
SELECTION_LABEL_METHOD="${SELECTION_LABEL_METHOD:-state_specific_margin}"
SELECTION_EMISSION_MARGIN="${SELECTION_EMISSION_MARGIN:-0.05}"
WARN_SCORING_FALLBACK_RATE="${WARN_SCORING_FALLBACK_RATE:-0.01}"
FATAL_SCORING_FALLBACK_RATE="${FATAL_SCORING_FALLBACK_RATE:-0.05}"
CONTINUATION_END_STAGE="${CONTINUATION_END_STAGE:-build_dpo}"
MAIN_PIPELINE="${MATHDIAL_MAIN_PIPELINE_SCRIPT:-$PROJECT_ROOT/scripts/run_mathdial_wildchat_pipeline.sh}"

[[ "$SCORING_BATCH_RECORDS" -gt 0 ]] || { echo "SCORING_BATCH_RECORDSは正数にしてください。" >&2; exit 20; }
[[ "$DPO_MAX_SOURCE_CHARACTERS" -gt 0 ]] || { echo "DPO_MAX_SOURCE_CHARACTERSは正数にしてください。" >&2; exit 20; }
[[ "$DPO_MAX_OUTPUT_TOKENS" -gt 0 ]] || { echo "DPO_MAX_OUTPUT_TOKENSは正数にしてください。" >&2; exit 20; }
[[ "$SCORING_BATCH_RECORDS" -lt "$ORIGINAL_SCORING_BATCH_RECORDS" ]] || {
  echo "小batch再開ではSCORING_BATCH_RECORDSを元のbatchより小さくしてください。" >&2
  exit 20
}

RAW="$OUTPUT_ROOT/scoring/wildchat_scored_raw.jsonl"
SCORED="$OUTPUT_ROOT/scoring/wildchat_scored.jsonl"
PRIORITIZED="$OUTPUT_ROOT/scoring/prioritized_candidates.jsonl"
MODEL="$OUTPUT_ROOT/basis_model/mathdial_transition_compat.json"
POOL_REPORT="$OUTPUT_ROOT/scoring/selection_pool_progress.json"
POOL_HISTORY="$OUTPUT_ROOT/scoring/selection_pool_history.jsonl"
AMENDMENTS="$OUTPUT_ROOT/scoring/scoring_configuration_amendments.jsonl"
SUCCESS="$OUTPUT_ROOT/stage_state/scoring_small_batch_CONTINUATION_SUCCESS.json"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR" "$(dirname "$SUCCESS")"
LOG_FILE="$LOG_DIR/scoring_small_batch_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

for path in "$OUTPUT_ROOT/run_metadata.json" "$RAW" "$PRIORITIZED" "$MODEL"; do
  [[ -f "$path" ]] || { echo "小batch再開に必要な成果物がありません: $path" >&2; exit 20; }
done

# 元runの研究条件は変更せず、batch幅だけを運用上のamendmentとして追跡する。
python3 - "$OUTPUT_ROOT/run_metadata.json" "$RUN_TAG" "$SCORING_MODEL" \
  "$SCORING_PRESET" "$ORIGINAL_SCORING_BATCH_RECORDS" "$SCORING_BATCH_RECORDS" \
  "$SELECTION_POOL_COUNT" "$RAW" "$AMENDMENTS" "$DPO_MAX_SOURCE_CHARACTERS" \
  "$DPO_MAX_OUTPUT_TOKENS" <<'PY'
import datetime
import json
import os
import pathlib
import sys

metadata_path = pathlib.Path(sys.argv[1])
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if metadata.get("run_tag") != sys.argv[2]:
    raise SystemExit("run_metadataのRUN_TAGが一致しません。")
if metadata.get("models", {}).get("scoring") != sys.argv[3]:
    raise SystemExit("scoring modelが元runと一致しません。")
if metadata.get("scoring", {}).get("preset") != sys.argv[4]:
    raise SystemExit("scoring presetが元runと一致しません。")
original = int(sys.argv[5])
recorded = int(metadata.get("early_stop", {}).get("scoring_batch_records", 0))
if recorded != original:
    raise SystemExit(
        f"元runのscoring batchが期待値と一致しません: recorded={recorded} expected={original}"
    )
raw = pathlib.Path(sys.argv[8])
with raw.open(encoding="utf-8", errors="strict") as file:
    records = sum(bool(line.strip()) for line in file)
payload = {
    "amendment_id": (
        f"length_bounded_v1:{metadata['experiment_fingerprint']}:"
        f"{sys.argv[6]}:{sys.argv[10]}:{sys.argv[11]}"
    ),
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "experiment_fingerprint": metadata["experiment_fingerprint"],
    "reason": "fallback会話をBASiS対象外にし、clean候補を小batchで必要数まで追加評価",
    "original_scoring_batch_records": original,
    "continued_scoring_batch_records": int(sys.argv[6]),
    "selection_pool_records": int(sys.argv[7]),
    "starting_scored_records": records,
    "mandatory_fallback_repair": False,
    "exclude_fallback_conversations_from_basis": True,
    "models": metadata.get("models", {}),
    "scoring": metadata.get("scoring", {}),
    "runtime_rate_limit": {
        "scoring_requests_per_minute": float(os.environ["SCORING_REQUESTS_PER_MINUTE"]),
        "repair_requests_per_minute": float(os.environ["SCORING_REPAIR_REQUESTS_PER_MINUTE"]),
        "max_retries": int(os.environ["SCORING_RATE_LIMIT_MAX_RETRIES"]),
        "initial_backoff_seconds": float(os.environ["SCORING_RATE_LIMIT_BACKOFF_SECONDS"]),
    },
    "selection": metadata.get("selection", {}),
    "length_eligibility": {
        "max_source_characters": int(sys.argv[10]),
        "policy": "exclude_whole_sample_without_truncating_history",
    },
    "dpo_max_output_tokens": int(sys.argv[11]),
}
amendments = pathlib.Path(sys.argv[9])
amendments.parent.mkdir(parents=True, exist_ok=True)
existing = []
if amendments.exists():
    existing = [json.loads(line) for line in amendments.open(encoding="utf-8") if line.strip()]
if not any(row.get("amendment_id") == payload["amendment_id"] for row in existing):
    with amendments.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
print(
    f"[small-batch resume] starting_scored={records} "
    f"batch={payload['continued_scoring_batch_records']} target={payload['selection_pool_records']}",
    flush=True,
)
PY

if [[ "${SMALL_BATCH_VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "[small-batch resume] validate-only完了"
  exit 0
fi

retry_command() {
  local attempt=1 delay=15 status
  while true; do
    set +e
    "$@"
    status=$?
    set -e
    [[ "$status" -eq 0 ]] && return 0
    [[ "$status" -eq 20 ]] && return 20
    if [[ "$attempt" -ge 3 ]]; then
      return "$status"
    fi
    echo "[retry] attempt=${attempt}/3 status=${status}; ${delay}s後に再試行します。" >&2
    sleep "$delay"
    delay=$((delay * 2))
    attempt=$((attempt + 1))
  done
}

while true; do
  python3 -m tools.mathdial_pipeline_support enrich-score --input "$RAW" --output "$SCORED"
  python3 -m tools.measure_basis_selection_pool \
    --input "$SCORED" \
    --bayes-model "$MODEL" \
    --output "$POOL_REPORT" \
    --history "$POOL_HISTORY" \
    --method "$SELECTION_LABEL_METHOD" \
    --margin "$SELECTION_EMISSION_MARGIN" \
    --required "$SELECTION_POOL_COUNT" \
    --exclude-fallback-conversations \
    --max-source-characters "$DPO_MAX_SOURCE_CHARACTERS"
  sufficient="$(python3 -c 'import json,sys; print("1" if json.load(open(sys.argv[1]))["sufficient"] else "0")' "$POOL_REPORT")"
  if [[ "$sufficient" == "1" ]]; then
    break
  fi
  before="$(wc -l < "$RAW")"
  retry_command python3 -m tools.score_dialogue_with_transition_bayes_model \
    --input "$PRIORITIZED" \
    --bayes-model "$MODEL" \
    --output "$RAW" \
    --model "$SCORING_MODEL" \
    --workers "$WORKERS" \
    --max-new-records "$SCORING_BATCH_RECORDS" \
    --scoring-preset "$SCORING_PRESET" \
    --invalid-observation-retries "$INVALID_OBSERVATION_RETRIES" \
    --requests-per-minute "$SCORING_REQUESTS_PER_MINUTE" \
    --rate-limit-max-retries "$SCORING_RATE_LIMIT_MAX_RETRIES" \
    --rate-limit-initial-backoff-seconds "$SCORING_RATE_LIMIT_BACKOFF_SECONDS" \
    --fallback-on-errors
  after="$(wc -l < "$RAW")"
  if [[ "$after" -le "$before" ]]; then
    echo "WildChat全粗候補をscoringしても選別候補が不足しています。" >&2
    exit 20
  fi
done

python3 -m tools.validate_scoring_fallbacks \
  --input "$SCORED" \
  --output "$OUTPUT_ROOT/scoring/fallback_diagnostics.json" \
  --warning-rate "$WARN_SCORING_FALLBACK_RATE" \
  --fatal-rate "$FATAL_SCORING_FALLBACK_RATE" \
  --diagnostic-only || exit 20

python3 - "$SUCCESS" "$RAW" "$SCORED" "$MODEL" "$POOL_REPORT" \
  "$SCORING_BATCH_RECORDS" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

path = pathlib.Path(sys.argv[1])
raw, scored, model, report = map(pathlib.Path, sys.argv[2:6])
pool = json.loads(report.read_text(encoding="utf-8"))
if not pool.get("sufficient"):
    raise SystemExit("選別候補数が完了条件を満たしていません。")
payload = {
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "continued_scoring_batch_records": int(sys.argv[6]),
    "scored_records": pool["scored_records"],
    "eligible_records": pool["eligible_records"],
    "eligible_records_before_fallback_exclusion": pool["eligible_records_before_fallback_exclusion"],
    "excluded_eligible_records": pool["excluded_eligible_records"],
    "exclude_fallback_conversations": pool["exclude_fallback_conversations"],
    "required_records": pool["required_records"],
    "hashes": {
        str(raw): sha256(raw),
        str(scored): sha256(scored),
        str(model): sha256(model),
    },
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY

echo "[small-batch resume] scoring完了。select_data以降を元fingerprint条件で再開します。"
ORIGINAL_FINGERPRINT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment_fingerprint"])' "$OUTPUT_ROOT/run_metadata.json")"
env \
  RUN_TAG="$RUN_TAG" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  START_STAGE=select_data \
  END_STAGE="$CONTINUATION_END_STAGE" \
  SCORING_BATCH_RECORDS="$ORIGINAL_SCORING_BATCH_RECORDS" \
  WORKERS="$WORKERS" \
  DPO_MAX_SOURCE_CHARACTERS="$DPO_MAX_SOURCE_CHARACTERS" \
  DPO_MAX_OUTPUT_TOKENS="$DPO_MAX_OUTPUT_TOKENS" \
  SCORING_REPAIR_WORKERS="$SCORING_REPAIR_WORKERS" \
  ALLOW_CLEAN_FALLBACK_CONTINUATION=1 \
  OPERATIONAL_FINGERPRINT_OVERRIDE="$ORIGINAL_FINGERPRINT" \
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}" \
  "$MAIN_PIPELINE"
