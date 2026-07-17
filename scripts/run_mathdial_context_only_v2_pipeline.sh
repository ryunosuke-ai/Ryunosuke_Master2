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

SOURCE_RUN="${SOURCE_RUN:-artifacts/mathdial_wildchat/runs/mathdial_wildchat_gpt56_v6_candidates4_mixed}"
RUN_TAG="${RUN_TAG:-mathdial_wildchat_gpt56_v10_neutral_prompt_boundary_fixed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/mathdial_wildchat/runs/${RUN_TAG}}"
DRY_RUN="${DRY_RUN:-0}"
START_STAGE="${START_STAGE:-rewrite_dpo}"
END_STAGE="${END_STAGE:-report}"
FORCE_STAGE="${FORCE_STAGE:-}"
WORKERS="${WORKERS:-4}"
TRAIN_SEED="${TRAIN_SEED:-42}"
EVAL_SEED="${EVAL_SEED:-20260717}"
EVAL_COUNT="${EVAL_COUNT:-100}"
EVAL_CANDIDATE_RESERVE="${EVAL_CANDIDATE_RESERVE:-20}"
LOCAL_MODEL="${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"
TRANSLATION_MODEL="${MATHDIAL_SCORING_MODEL:-${AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT:-gpt-5.6-terra}}"
JUDGE_MODEL="${MATHDIAL_JUDGE_MODEL:-${AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT:-gpt-5.6-terra}}"
TRAIN_CUDA_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
EVAL_CUDA_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1}"
TRAIN_DEVICE_MAP="${TRAIN_DEVICE_MAP:-auto}"
TRAIN_MAX_MEMORY="${TRAIN_MAX_MEMORY:-0=38GiB,1=46GiB,cpu=0GiB}"
EVAL_MAX_MEMORY="${EVAL_MAX_MEMORY:-0=46GiB,1=46GiB,cpu=0GiB}"
TRAIN_SAVE_TOTAL_LIMIT="${TRAIN_SAVE_TOTAL_LIMIT:-2}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-4096}"
TRAIN_MIN_FREE_MEMORY_MIB="${TRAIN_MIN_FREE_MEMORY_MIB:-36000}"
TRAIN_GPU0_MIN_HEADROOM_MIB="${TRAIN_GPU0_MIN_HEADROOM_MIB:-8192}"
TRAIN_CUDA_ALLOC_CONF="${TRAIN_CUDA_ALLOC_CONF:-expandable_segments:True}"
ALLOW_TRAIN_PLACEMENT_CONTINUATION="${ALLOW_TRAIN_PLACEMENT_CONTINUATION:-0}"
PIPELINE_MIN_FREE_GB="${PIPELINE_MIN_FREE_GB:-8}"
RECORDS_PER_ARM="${RECORDS_PER_ARM:-2500}"
BASIS_GOLD_RECORDS="${BASIS_GOLD_RECORDS:-500}"
if [[ "$DRY_RUN" == "1" ]]; then
  EVAL_COUNT="${DRY_RUN_EVAL_COUNT:-5}"
  EVAL_CANDIDATE_RESERVE="${DRY_RUN_EVAL_CANDIDATE_RESERVE:-3}"
  RECORDS_PER_ARM="${DRY_RUN_RECORDS_PER_ARM:-6}"
  BASIS_GOLD_RECORDS="${DRY_RUN_BASIS_GOLD_RECORDS:-2}"
fi

STAGES=(rewrite_dpo train prepare_eval generate_responses oracle_v2 statistics report)
STATE_DIR="$OUTPUT_ROOT/stage_state"
LOG_DIR="$OUTPUT_ROOT/logs"
STATUS_FILE="$OUTPUT_ROOT/pipeline_status.json"
HEARTBEAT_FILE="$OUTPUT_ROOT/pipeline_heartbeat.json"
DPO_DIR="$OUTPUT_ROOT/dpo_neutral_conversation"
TRAIN_DIR="$OUTPUT_ROOT/training"
EVAL_DIR="$OUTPUT_ROOT/evaluation"
mkdir -p "$STATE_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/neutral_prompt_v2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

SOURCE_BASIS="$SOURCE_RUN/dpo/mathdial_basis_train.jsonl"
SOURCE_RANDOM="$SOURCE_RUN/dpo/mathdial_random_train.jsonl"
SOURCE_SAMPLES="$SOURCE_RUN/mathdial/data/mathdial_assistant_samples.jsonl"
SOURCE_CONVERSATIONS="$SOURCE_RUN/mathdial/data/mathdial_conversations.jsonl"
PREVIOUS_PROMPTS="$SOURCE_RUN/evaluation/prompts_ja.jsonl"
CONTEXT_BASIS="$DPO_DIR/mathdial_basis_train.jsonl"
CONTEXT_RANDOM="$DPO_DIR/mathdial_random_train.jsonl"
PROMPTS="$EVAL_DIR/prompts_ja.jsonl"
RESPONSES="$EVAL_DIR/responses.jsonl"
ORACLE_INPUT="$EVAL_DIR/oracle_input.jsonl"
PEDAGOGICAL_DIR="$EVAL_DIR/oracle/pedagogical_v2"
GENERAL_DIR="$EVAL_DIR/oracle/general"
STATISTICS_DIR="$EVAL_DIR/statistics"

for required in "$SOURCE_BASIS" "$SOURCE_RANDOM" "$SOURCE_SAMPLES" \
  "$SOURCE_CONVERSATIONS" "$PREVIOUS_PROMPTS"; do
  [[ -f "$required" ]] || { echo "必要なsource成果物がありません: $required" >&2; exit 20; }
done

available_gb="$(df -Pk "$OUTPUT_ROOT" | awk 'NR == 2 {print int($4 / 1024 / 1024)}')"
if [[ "$DRY_RUN" != "1" && "$available_gb" -lt "$PIPELINE_MIN_FREE_GB" ]]; then
  echo "pipeline開始前の空き容量が不足しています: ${available_gb}GB/${PIPELINE_MIN_FREE_GB}GB" >&2
  exit 20
fi

stage_index() {
  local target="$1" index
  for index in "${!STAGES[@]}"; do
    if [[ "${STAGES[$index]}" == "$target" ]]; then
      echo "$index"
      return 0
    fi
  done
  echo "未知のstageです: $target" >&2
  return 1
}

START_INDEX="$(stage_index "$START_STAGE")"
END_INDEX="$(stage_index "$END_STAGE")"
[[ "$START_INDEX" -le "$END_INDEX" ]] || {
  echo "START_STAGEはEND_STAGE以前にしてください。" >&2
  exit 2
}

COMPUTED_EXPERIMENT_FINGERPRINT="$(
  python3 - "$SOURCE_BASIS" "$SOURCE_RANDOM" "$SOURCE_SAMPLES" \
    "$SOURCE_CONVERSATIONS" "$PREVIOUS_PROMPTS" "$TRAIN_SEED" "$EVAL_SEED" \
    "$EVAL_COUNT" "$LOCAL_MODEL" "$TRANSLATION_MODEL" "$JUDGE_MODEL" \
    "$RECORDS_PER_ARM" "$BASIS_GOLD_RECORDS" "$EVAL_CANDIDATE_RESERVE" \
    "$TRAIN_MAX_LENGTH" <<'PY'
import hashlib
import json
import pathlib
import sys

def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

files = {
    path: sha256(path)
    for path in sys.argv[1:6]
}
for path in (
    "core/dpo_prompting.py",
    "tools/rewrite_mathdial_dpo_context_only.py",
    "tools/mathdial_evaluation.py",
    "tools/train_qwen35_dpo_lora.py",
    "scripts/eval_oracle_mathdial_v2.py",
    "scripts/run_mathdial_statistics.py",
    "scripts/run_mathdial_context_only_v2_pipeline.sh",
    "configs/evaluations/mathdial_oracle_v2.yaml",
):
    files[path] = sha256(path)
payload = {
    "files": files,
    "values": sys.argv[6:],
    "local_prompt_mode": "neutral_conversation",
    "template_version": "dpo_user_ai_neutral_instruction.v3",
}
print(hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest())
PY
)"
EXPERIMENT_FINGERPRINT="$COMPUTED_EXPERIMENT_FINGERPRINT"

if [[ -s "$OUTPUT_ROOT/run_metadata.json" ]]; then
  EXISTING_EXPERIMENT_FINGERPRINT="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment_fingerprint"])' \
      "$OUTPUT_ROOT/run_metadata.json"
  )"
  if [[ "$EXISTING_EXPERIMENT_FINGERPRINT" != "$COMPUTED_EXPERIMENT_FINGERPRINT" ]]; then
    if [[ "$ALLOW_TRAIN_PLACEMENT_CONTINUATION" != "1" ]]; then
      echo "同じRUN_TAGのコードfingerprintが変わっています。" >&2
      echo "学習配置だけを修正してcheckpointから続ける場合は、ALLOW_TRAIN_PLACEMENT_CONTINUATION=1を指定してください。" >&2
      exit 20
    fi
    python3 - "$OUTPUT_ROOT/run_metadata.json" "$RUN_TAG" "$SOURCE_RUN" \
      "$TRAIN_SEED" "$EVAL_SEED" "$EVAL_COUNT" "$LOCAL_MODEL" \
      "$TRANSLATION_MODEL" "$JUDGE_MODEL" "$RECORDS_PER_ARM" \
      "$BASIS_GOLD_RECORDS" "$EVAL_CANDIDATE_RESERVE" "$TRAIN_MAX_LENGTH" <<'PY'
import json
import pathlib
import sys

metadata = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "run_tag": sys.argv[2],
    "source_run": sys.argv[3],
    "training_seed": int(sys.argv[4]),
    "evaluation_seed": int(sys.argv[5]),
    "evaluation_count": int(sys.argv[6]),
    "models": {
        "local": sys.argv[7],
        "translation": sys.argv[8],
        "judge": sys.argv[9],
    },
}
for key, value in expected.items():
    if metadata.get(key) != value:
        raise SystemExit(
            f"配置変更として継続できない研究条件差分です: {key} "
            f"existing={metadata.get(key)!r} current={value!r}"
        )
dpo = metadata.get("dpo", {})
if int(dpo.get("records_per_arm", -1)) != int(sys.argv[10]):
    raise SystemExit("records_per_armが変わっているため継続できません。")
if int(dpo.get("basis_gold_records", -1)) != int(sys.argv[11]):
    raise SystemExit("basis_gold_recordsが変わっているため継続できません。")
evaluation = metadata.get("evaluation", {})
if int(evaluation.get("candidate_reserve", -1)) != int(sys.argv[12]):
    raise SystemExit("candidate_reserveが変わっているため継続できません。")
training = metadata.get("training", {})
if int(training.get("max_length", -1)) != int(sys.argv[13]):
    raise SystemExit("max_lengthが変わっているため継続できません。")
PY
    EXPERIMENT_FINGERPRINT="$EXISTING_EXPERIMENT_FINGERPRINT"
  fi
fi

python3 - "$OUTPUT_ROOT/run_metadata.json" "$EXPERIMENT_FINGERPRINT" \
  "$RUN_TAG" "$SOURCE_RUN" "$TRAIN_SEED" "$EVAL_SEED" "$EVAL_COUNT" \
  "$LOCAL_MODEL" "$TRANSLATION_MODEL" "$JUDGE_MODEL" "$RECORDS_PER_ARM" \
  "$BASIS_GOLD_RECORDS" "$EVAL_CANDIDATE_RESERVE" "$TRAIN_MAX_LENGTH" \
  "$TRAIN_DEVICE_MAP" "$TRAIN_MAX_MEMORY" "$TRAIN_GPU0_MIN_HEADROOM_MIB" \
  "$TRAIN_CUDA_ALLOC_CONF" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
payload = {
    "experiment_fingerprint": sys.argv[2],
    "run_tag": sys.argv[3],
    "source_run": sys.argv[4],
    "training_seed": int(sys.argv[5]),
    "evaluation_seed": int(sys.argv[6]),
    "evaluation_count": int(sys.argv[7]),
    "models": {
        "local": sys.argv[8],
        "translation": sys.argv[9],
        "judge": sys.argv[10],
    },
    "dpo": {
        "records_per_arm": int(sys.argv[11]),
        "basis_gold_records": int(sys.argv[12]),
        "prompt_mode": "neutral_conversation",
        "prompt_template_version": "dpo_user_ai_neutral_instruction.v3",
        "chosen_rejected_policy": "unchanged_from_source_run",
    },
    "training": {
        "max_length": int(sys.argv[14]),
        "token_truncation_policy": "reject_if_exceeds",
        "device_map": sys.argv[15],
        "max_memory": sys.argv[16],
        "gpu0_minimum_activation_headroom_mib": int(sys.argv[17]),
        "cuda_allocator_configuration": sys.argv[18],
    },
    "evaluation": {
        "status": "confirmatory_axes_frozen_before_scoring",
        "exclude_v1_qids": True,
        "candidate_reserve": int(sys.argv[13]),
    },
    "created_at": datetime.now(timezone.utc).isoformat(),
}
if path.exists():
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("experiment_fingerprint") != payload["experiment_fingerprint"]:
        raise SystemExit(
            "同じRUN_TAGの実験条件が変わっています。新しいRUN_TAGを使ってください。"
        )
else:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
PY

python3 - "$OUTPUT_ROOT/training_runtime_attempts.jsonl" \
  "$RUN_TAG" "$EXPERIMENT_FINGERPRINT" "$COMPUTED_EXPERIMENT_FINGERPRINT" \
  "$TRAIN_DEVICE_MAP" "$TRAIN_MAX_MEMORY" "$TRAIN_GPU0_MIN_HEADROOM_MIB" \
  "$TRAIN_CUDA_ALLOC_CONF" "${WATCHDOG_ATTEMPT:-1}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
record = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "run_tag": sys.argv[2],
    "experiment_fingerprint": sys.argv[3],
    "computed_code_fingerprint": sys.argv[4],
    "placement_only_continuation": sys.argv[3] != sys.argv[4],
    "device_map": sys.argv[5],
    "max_memory": sys.argv[6],
    "gpu0_minimum_activation_headroom_mib": int(sys.argv[7]),
    "cuda_allocator_configuration": sys.argv[8],
    "watchdog_attempt": int(sys.argv[9]),
}
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as file:
    file.write(json.dumps(record, ensure_ascii=False) + "\n")
PY

write_status() {
  local state="$1" stage="$2" message="$3"
  python3 - "$STATUS_FILE" "$HEARTBEAT_FILE" "$RUN_TAG" "$state" "$stage" "$message" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

payload = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "run_tag": sys.argv[3],
    "state": sys.argv[4],
    "stage": sys.argv[5],
    "message": sys.argv[6],
}
for value in sys.argv[1:3]:
    path = pathlib.Path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
PY
}

stage_outputs_valid() {
  local expected_oracle=$((EVAL_COUNT * 3))
  case "$1" in
    rewrite_dpo)
      [[ -s "$CONTEXT_BASIS" && -s "$CONTEXT_RANDOM" &&
        -s "$DPO_DIR/rewrite_manifest.json" ]] &&
        [[ "$(wc -l < "$CONTEXT_BASIS")" -eq "$RECORDS_PER_ARM" ]] &&
        [[ "$(wc -l < "$CONTEXT_RANDOM")" -eq "$RECORDS_PER_ARM" ]]
      ;;
    train)
      if [[ "$DRY_RUN" == "1" ]]; then
        [[ -s "$TRAIN_DIR/dry_run_SUCCESS" ]]
      else
        [[ -s "$TRAIN_DIR/basis_lora/adapter_model.safetensors" &&
          -s "$TRAIN_DIR/random_lora/adapter_model.safetensors" ]]
      fi
      ;;
    prepare_eval)
      [[ -s "$PROMPTS" && "$(wc -l < "$PROMPTS")" -eq "$EVAL_COUNT" ]]
      ;;
    generate_responses)
      [[ -s "$RESPONSES" && -s "$ORACLE_INPUT" ]] &&
        [[ "$(wc -l < "$RESPONSES")" -eq "$EVAL_COUNT" ]] &&
        [[ "$(wc -l < "$ORACLE_INPUT")" -eq "$expected_oracle" ]]
      ;;
    oracle_v2)
      [[ -s "$PEDAGOGICAL_DIR/raw.jsonl" && -s "$GENERAL_DIR/raw.jsonl" ]] &&
        [[ "$(wc -l < "$PEDAGOGICAL_DIR/raw.jsonl")" -eq "$expected_oracle" ]] &&
        [[ "$(wc -l < "$GENERAL_DIR/raw.jsonl")" -eq "$expected_oracle" ]]
      ;;
    statistics)
      [[ -s "$STATISTICS_DIR/omnibus_friedman.csv" &&
        -f "$STATISTICS_DIR/posthoc_pairwise.csv" ]]
      ;;
    report) [[ -s "$OUTPUT_ROOT/report.md" && -s "$OUTPUT_ROOT/manifest.json" ]] ;;
    *) return 1 ;;
  esac
}

run_stage() {
  local name="$1" function="$2" index marker
  index="$(stage_index "$name")"
  [[ "$index" -ge "$START_INDEX" && "$index" -le "$END_INDEX" ]] || return 0
  marker="$STATE_DIR/${name}_SUCCESS.json"
  if [[ "$FORCE_STAGE" != "$name" && -s "$marker" ]]; then
    python3 - "$marker" "$EXPERIMENT_FINGERPRINT" <<'PY'
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("experiment_fingerprint") != sys.argv[2]:
    raise SystemExit("stage markerのfingerprintが一致しません。")
PY
    if stage_outputs_valid "$name"; then
      echo "[SKIP] $name"
      return 0
    fi
    rm -f "$marker"
  fi
  echo "[START] $name"
  write_status "running" "$name" "stage started"
  "$function"
  stage_outputs_valid "$name" || {
    echo "$name の完了成果物が不足しています。" >&2
    return 20
  }
  python3 - "$marker" "$EXPERIMENT_FINGERPRINT" "$name" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone
path = pathlib.Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "experiment_fingerprint": sys.argv[2],
            "stage": sys.argv[3],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY
  write_status "running" "$name" "stage completed"
  echo "[DONE] $name"
}

gpu_preflight() {
  local devices="$1" label="$2"
  [[ "$DRY_RUN" == "1" ]] && return 0
  command -v nvidia-smi >/dev/null || {
    echo "$label: nvidia-smiが見つかりません。" >&2
    return 20
  }
  python3 - "$devices" "$TRAIN_MIN_FREE_MEMORY_MIB" "$label" <<'PY'
import subprocess
import sys
devices = [int(value) for value in sys.argv[1].split(",") if value.strip()]
minimum = int(sys.argv[2])
rows = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
    text=True,
)
free = {}
for row in rows.splitlines():
    index, memory = [item.strip() for item in row.split(",", 1)]
    free[int(index)] = int(memory)
missing = [index for index in devices if free.get(index, 0) < minimum]
if missing:
    raise SystemExit(
        f"{sys.argv[3]}: GPU空きメモリ不足 devices={missing} "
        f"required={minimum}MiB actual={free}"
    )
PY
}

train_placement_preflight() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  [[ "$TRAIN_DEVICE_MAP" == "auto" ]] || return 0
  python3 - "$TRAIN_CUDA_DEVICES" "$TRAIN_MAX_MEMORY" \
    "$TRAIN_GPU0_MIN_HEADROOM_MIB" <<'PY'
import re
import subprocess
import sys

physical_devices = [
    int(value) for value in sys.argv[1].split(",") if value.strip()
]
if not physical_devices:
    raise SystemExit("TRAIN_CUDA_VISIBLE_DEVICESが空です。")

budgets = {}
for item in sys.argv[2].split(","):
    key, separator, raw_value = item.partition("=")
    if not separator or key.strip().lower() == "cpu":
        continue
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(GiB|MiB)\s*",
        raw_value,
        flags=re.IGNORECASE,
    )
    if not match:
        raise SystemExit(f"TRAIN_MAX_MEMORYを解釈できません: {item}")
    amount = float(match.group(1))
    unit = match.group(2).lower()
    budgets[int(key.strip())] = amount * (1024 if unit == "gib" else 1)

if 0 not in budgets:
    raise SystemExit(
        "device_map=autoではTRAIN_MAX_MEMORYにlogical GPU 0の上限が必要です。"
    )

rows = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.total",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
totals = {}
for row in rows.splitlines():
    index, memory = [item.strip() for item in row.split(",", 1)]
    totals[int(index)] = int(memory)

physical_gpu0 = physical_devices[0]
total_mib = totals.get(physical_gpu0)
if total_mib is None:
    raise SystemExit(f"GPU total memoryを取得できません: {physical_gpu0}")
headroom_mib = total_mib - budgets[0]
minimum_mib = int(sys.argv[3])
print(
    "[preflight] training placement "
    f"physical_gpu0={physical_gpu0} total={total_mib}MiB "
    f"model_budget={budgets[0]:.0f}MiB "
    f"activation_headroom={headroom_mib:.0f}MiB "
    f"required={minimum_mib}MiB"
)
if headroom_mib < minimum_mib:
    raise SystemExit(
        "GPU 0のactivation用余白が不足しています。"
        f" headroom={headroom_mib:.0f}MiB required={minimum_mib}MiB。"
        "TRAIN_MAX_MEMORYのlogical GPU 0上限を下げてください。"
    )
PY
}

run_training_arm() {
  local arm="$1" dataset="$2" output_dir="$3"
  local attempt="${WATCHDOG_ATTEMPT:-1}"
  local attempt_log="$TRAIN_DIR/${arm}_training_attempt_${attempt}.log"
  local oom_marker="$TRAIN_DIR/OOM_DETECTED.json"
  local status
  rm -f "$oom_marker"
  set +e
  env CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_DEVICES" \
    PYTORCH_CUDA_ALLOC_CONF="$TRAIN_CUDA_ALLOC_CONF" \
    python3 -m tools.train_qwen35_dpo_lora \
      --dataset "$dataset" \
      --output-dir "$output_dir" \
      "${common[@]}" 2>&1 | tee -a "$attempt_log"
  status="${PIPESTATUS[0]}"
  set -e
  if [[ "$status" -ne 0 ]] && grep -q "CUDA out of memory" "$attempt_log"; then
    python3 - "$oom_marker" "$arm" "$attempt" "$TRAIN_MAX_MEMORY" \
      "$TRAIN_GPU0_MIN_HEADROOM_MIB" "$attempt_log" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "arm": sys.argv[2],
            "watchdog_attempt": int(sys.argv[3]),
            "max_memory": sys.argv[4],
            "gpu0_minimum_activation_headroom_mib": int(sys.argv[5]),
            "attempt_log": sys.argv[6],
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY
  fi
  return "$status"
}

rewrite_dpo_stage() {
  python3 -m tools.rewrite_mathdial_dpo_context_only \
    --basis-input "$SOURCE_BASIS" \
    --random-input "$SOURCE_RANDOM" \
    --output-dir "$DPO_DIR" \
    --manifest "$DPO_DIR/rewrite_manifest.json" \
    --records-per-arm "$RECORDS_PER_ARM" \
    --basis-gold-records "$BASIS_GOLD_RECORDS" \
    --prompt-mode neutral_conversation
}

train_stage() {
  mkdir -p "$TRAIN_DIR"
  local common=(
    --model-id "$LOCAL_MODEL"
    --num-train-epochs 1
    --learning-rate 5e-6
    --beta 0.1
    --max-length "$TRAIN_MAX_LENGTH"
    --per-device-train-batch-size 1
    --gradient-accumulation-steps 8
    --lora-r 8
    --lora-alpha 16
    --lora-dropout 0.05
    --save-steps 25
    --save-total-limit "$TRAIN_SAVE_TOTAL_LIMIT"
    --warmup-ratio 0.03
    --eval-ratio 0
    --seed "$TRAIN_SEED"
    --no-4bit
    --device-map "$TRAIN_DEVICE_MAP"
    --max-memory "$TRAIN_MAX_MEMORY"
    --resume-from-checkpoint auto
    --require-tokenizer-prefix-match
    --require-no-token-truncation
  )
  python3 -m tools.train_qwen35_dpo_lora \
    --dataset "$CONTEXT_BASIS" \
    --output-dir "$TRAIN_DIR/basis_lora" \
    "${common[@]}" \
    --dry-run
  python3 -m tools.train_qwen35_dpo_lora \
    --dataset "$CONTEXT_RANDOM" \
    --output-dir "$TRAIN_DIR/random_lora" \
    "${common[@]}" \
    --dry-run
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\n' "dry-run completed" > "$TRAIN_DIR/dry_run_SUCCESS"
    return 0
  fi
  gpu_preflight "$TRAIN_CUDA_DEVICES" "BASiS DPO training" || return 20
  train_placement_preflight || return 20
  run_training_arm "basis" "$CONTEXT_BASIS" "$TRAIN_DIR/basis_lora"
  [[ -s "$TRAIN_DIR/basis_lora/adapter_model.safetensors" ]] || return 20
  gpu_preflight "$TRAIN_CUDA_DEVICES" "Random DPO training" || return 20
  train_placement_preflight || return 20
  run_training_arm "random" "$CONTEXT_RANDOM" "$TRAIN_DIR/random_lora"
}

prepare_eval_stage() {
  mkdir -p "$EVAL_DIR"
  local mock=()
  [[ "$DRY_RUN" == "1" ]] && mock+=(--mock)
  python3 -m tools.mathdial_evaluation prepare \
    --samples "$SOURCE_SAMPLES" \
    --conversations "$SOURCE_CONVERSATIONS" \
    --output "$PROMPTS" \
    --errors-output "$EVAL_DIR/translation_errors.jsonl" \
    --skip-sample-errors \
    --count "$EVAL_COUNT" \
    --seed "$EVAL_SEED" \
    --model "$TRANSLATION_MODEL" \
    --exclude-prompts "$PREVIOUS_PROMPTS" \
    --stratify-teacher-moves \
    --candidate-reserve "$EVAL_CANDIDATE_RESERVE" \
    --prompt-id-prefix "mathdial_neutral_v2" \
    --local-prompt-mode neutral_conversation \
    --resume \
    "${mock[@]}"
  python3 - "$PROMPTS" "$PREVIOUS_PROMPTS" "$EVAL_COUNT" <<'PY'
import json
import pathlib
import sys

current = [
    json.loads(line)
    for line in pathlib.Path(sys.argv[1]).open(encoding="utf-8")
    if line.strip()
]
previous = [
    json.loads(line)
    for line in pathlib.Path(sys.argv[2]).open(encoding="utf-8")
    if line.strip()
]
required = int(sys.argv[3])
if len(current) != required:
    raise SystemExit(f"確認評価promptが不足しています: {len(current)}/{required}")
if {str(row["qid"]) for row in current} & {str(row["qid"]) for row in previous}:
    raise SystemExit("v1とneutral-prompt v2の評価qidが重複しています。")
if len({str(row["qid"]) for row in current}) != required:
    raise SystemExit("neutral-prompt v2評価内でqidが重複しています。")
for row in current:
    prompt = str(row.get("model_prompt", ""))
    if row.get("local_prompt_mode") != "neutral_conversation":
        raise SystemExit("評価promptがneutral_conversationではありません。")
    if not prompt.endswith("AI:\n"):
        raise SystemExit("評価promptが末尾のAI:と改行で終わっていません。")
    instruction = "以下の会話の文脈に沿って、次のAIの応答を自然な日本語で生成してください。"
    if not prompt.startswith(instruction + "\n\nUser:"):
        raise SystemExit("評価promptの中立的な会話指示が不正です。")
    if str(row["problem_ja"]).strip() not in prompt:
        raise SystemExit("評価promptから問題文が失われています。")
    forbidden = (
        "個別指導", "教師返答", "段階的ヒント", "理解確認",
        "equitable_tutoring", "BASiS", "ground_truth",
    )
    header = prompt.split("\n\n", 1)[0]
    if any(token in header for token in forbidden):
        raise SystemExit(f"評価生成promptに禁止された指示が含まれます: {header}")
PY
}

generate_responses_stage() {
  local mock=()
  [[ "$DRY_RUN" == "1" ]] && mock+=(--mock)
  if [[ "$DRY_RUN" != "1" ]]; then
    gpu_preflight "$EVAL_CUDA_DEVICES" \
      "neutral-prompt evaluation generation" || return 20
  fi
  env CUDA_VISIBLE_DEVICES="$EVAL_CUDA_DEVICES" \
    DPO_COMPARE_MAX_MEMORY="$EVAL_MAX_MEMORY" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    python3 -m tools.mathdial_evaluation generate \
      --input "$PROMPTS" \
      --output "$RESPONSES" \
      --errors-output "$EVAL_DIR/generation_errors.jsonl" \
      --skip-sample-errors \
      --oracle-output "$ORACLE_INPUT" \
      --base-model "$LOCAL_MODEL" \
      --basis-lora "$TRAIN_DIR/basis_lora" \
      --random-lora "$TRAIN_DIR/random_lora" \
      --seed "$EVAL_SEED" \
      --local-prompt-mode neutral_conversation \
      "${mock[@]}"
  python3 - "$RESPONSES" "$ORACLE_INPUT" "$EVAL_COUNT" <<'PY'
import json
import pathlib
import sys
responses = [
    json.loads(line)
    for line in pathlib.Path(sys.argv[1]).open(encoding="utf-8")
    if line.strip()
]
oracle = [
    json.loads(line)
    for line in pathlib.Path(sys.argv[2]).open(encoding="utf-8")
    if line.strip()
]
required = int(sys.argv[3])
if len(responses) != required or len(oracle) != required * 3:
    raise SystemExit(
        f"評価応答不足 responses={len(responses)}/{required} "
        f"oracle={len(oracle)}/{required * 3}"
    )
for row in responses:
    if row.get("local_prompt_mode") != "neutral_conversation":
        raise SystemExit("異なるlocal_prompt_modeの応答が混入しています。")
    if not all(str(row.get(key, "")).strip() for key in (
        "base_response", "basis_response", "random_dpo_response"
    )):
        raise SystemExit("3モデルのいずれかに空応答があります。")
PY
}

oracle_v2_stage() {
  local dry=()
  [[ "$DRY_RUN" == "1" ]] && dry+=(--dry-run)
  python3 scripts/eval_oracle_mathdial_v2.py \
    --input "$ORACLE_INPUT" \
    --output_dir "$PEDAGOGICAL_DIR" \
    --category pedagogical \
    --judge_model "$JUDGE_MODEL" \
    --score-scale 10 \
    --oracle-workers "$WORKERS" \
    --seed "$EVAL_SEED" \
    --resume \
    "${dry[@]}"
  python3 scripts/eval_oracle_mathdial_v2.py \
    --input "$ORACLE_INPUT" \
    --output_dir "$GENERAL_DIR" \
    --category general \
    --judge_model "$JUDGE_MODEL" \
    --score-scale 10 \
    --oracle-workers "$WORKERS" \
    --seed "$EVAL_SEED" \
    --resume \
    "${dry[@]}"
  local required=$((EVAL_COUNT * 3))
  python3 - "$PEDAGOGICAL_DIR/raw.jsonl" "$GENERAL_DIR/raw.jsonl" "$required" <<'PY'
import pathlib
import sys
required = int(sys.argv[3])
for value in sys.argv[1:3]:
    path = pathlib.Path(value)
    actual = sum(bool(line.strip()) for line in path.open(encoding="utf-8"))
    if actual != required:
        raise SystemExit(f"Oracle評価件数が不足しています: {path} {actual}/{required}")
PY
}

statistics_stage() {
  local permutations=10000 bootstrap=2000
  if [[ "$DRY_RUN" == "1" ]]; then
    permutations=100
    bootstrap=100
  fi
  python3 scripts/run_mathdial_statistics.py \
    --raw "$PEDAGOGICAL_DIR/raw.jsonl" \
    --raw "$GENERAL_DIR/raw.jsonl" \
    --output-dir "$STATISTICS_DIR" \
    --permutations "$permutations" \
    --bootstrap "$bootstrap" \
    --seed "$EVAL_SEED"
}

report_stage() {
  python3 - "$OUTPUT_ROOT" "$SOURCE_RUN" "$EXPERIMENT_FINGERPRINT" \
    "$EVAL_COUNT" "$TRAIN_SEED" "$EVAL_SEED" "$TRANSLATION_MODEL" \
    "$JUDGE_MODEL" "$RECORDS_PER_ARM" "$BASIS_GOLD_RECORDS" \
    "$TRAIN_MAX_LENGTH" "$TRAIN_DEVICE_MAP" "$TRAIN_MAX_MEMORY" \
    "$TRAIN_GPU0_MIN_HEADROOM_MIB" "$TRAIN_CUDA_ALLOC_CONF" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

root = pathlib.Path(sys.argv[1])
runtime_attempts_path = root / "training_runtime_attempts.jsonl"
runtime_attempts = []
if runtime_attempts_path.is_file():
    runtime_attempts = [
        json.loads(line)
        for line in runtime_attempts_path.open(encoding="utf-8")
        if line.strip()
    ]
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "confirmatory_axes_frozen_before_scoring",
    "source_run": sys.argv[2],
    "experiment_fingerprint": sys.argv[3],
    "evaluation_count": int(sys.argv[4]),
    "training_seed": int(sys.argv[5]),
    "evaluation_seed": int(sys.argv[6]),
    "translation_model": sys.argv[7],
    "judge_model": sys.argv[8],
    "local_prompt_mode": "neutral_conversation",
    "prompt_template_version": "dpo_user_ai_neutral_instruction.v3",
    "prompt_overlap_with_v1": 0,
    "training": {
        "max_length": int(sys.argv[11]),
        "token_truncation_policy": "reject_if_exceeds",
        "device_map": sys.argv[12],
        "max_memory": sys.argv[13],
        "gpu0_minimum_activation_headroom_mib": int(sys.argv[14]),
        "cuda_allocator_configuration": sys.argv[15],
        "runtime_attempts": runtime_attempts,
    },
    "training_data_policy": {
        "chosen_rejected": "unchanged_from_source_run",
        "basis_records": int(sys.argv[9]),
        "basis_selected_records": int(sys.argv[9]) - int(sys.argv[10]),
        "basis_gold_records": int(sys.argv[10]),
        "random_records": int(sys.argv[9]),
        "random_gold_records": 0,
    },
    "rewrite_manifest": json.loads(
        (root / "dpo_neutral_conversation/rewrite_manifest.json").read_text(encoding="utf-8")
    ),
}
(root / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
sections = [
    "# MathDial neutral-prompt DPO / Oracle v2 confirmation",
    "",
    "v1で未使用のtest qidを使い、生成モデルへスタイル指示を与えずに比較した主評価。",
    "",
    "## Manifest",
    "",
    "```json",
    json.dumps(manifest, ensure_ascii=False, indent=2),
    "```",
]
for title, relative in (
    ("Model summary", "evaluation/statistics/model_summary.csv"),
    ("Friedman", "evaluation/statistics/omnibus_friedman.csv"),
    ("Holm-adjusted post-hoc", "evaluation/statistics/posthoc_pairwise.csv"),
):
    sections.extend([
        "",
        f"## {title}",
        "",
        "```csv",
        (root / relative).read_text(encoding="utf-8").strip(),
        "```",
    ])
(root / "report.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
PY
}

trap 'status=$?; if [[ $status -ne 0 ]]; then write_status incomplete "${CURRENT_STAGE:-startup}" "pipeline exited status=$status" || true; fi' EXIT

CURRENT_STAGE="rewrite_dpo"
run_stage rewrite_dpo rewrite_dpo_stage
CURRENT_STAGE="train"
run_stage train train_stage
CURRENT_STAGE="prepare_eval"
run_stage prepare_eval prepare_eval_stage
CURRENT_STAGE="generate_responses"
run_stage generate_responses generate_responses_stage
CURRENT_STAGE="oracle_v2"
run_stage oracle_v2 oracle_v2_stage
CURRENT_STAGE="statistics"
run_stage statistics statistics_stage
CURRENT_STAGE="report"
run_stage report report_stage
write_status "success" "completed" "pipeline completed"
trap - EXIT

echo "MathDial neutral-prompt v2 pipeline completed: $OUTPUT_ROOT"
echo "Report: $OUTPUT_ROOT/report.md"
echo "Log: $LOG_FILE"
