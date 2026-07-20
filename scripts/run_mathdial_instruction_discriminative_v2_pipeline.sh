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
EXCLUDE_NEUTRAL_RUN="${EXCLUDE_NEUTRAL_RUN:-artifacts/mathdial_wildchat/runs/mathdial_wildchat_gpt56_v11_neutral_prompt_v6_length}"
RUN_TAG="${RUN_TAG:-mathdial_v6_instruction_discriminative_followup_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/mathdial_wildchat/evaluation_rechecks/${RUN_TAG}}"
START_STAGE="${START_STAGE:-prepare_eval}"
END_STAGE="${END_STAGE:-report}"
FORCE_STAGE="${FORCE_STAGE:-}"
DRY_RUN="${DRY_RUN:-0}"
WORKERS="${WORKERS:-4}"
EVAL_COUNT="${EVAL_COUNT:-150}"
EVAL_SEED="${EVAL_SEED:-20260720}"
LOCAL_MODEL="${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"
TRANSLATION_MODEL="${MATHDIAL_SCORING_MODEL:-${AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT:-gpt-5.6-terra}}"
JUDGE_MODEL="${MATHDIAL_JUDGE_MODEL:-${AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT:-gpt-5.6-terra}}"
EVAL_CUDA_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1}"
EVAL_MAX_MEMORY="${EVAL_MAX_MEMORY:-0=46GiB,1=46GiB,cpu=0GiB}"
EVAL_MIN_FREE_MEMORY_MIB="${EVAL_MIN_FREE_MEMORY_MIB:-36000}"
PIPELINE_MIN_FREE_GB="${PIPELINE_MIN_FREE_GB:-4}"
QUOTA_CONFIG="${MATHDIAL_DISCRIMINATIVE_QUOTA_CONFIG:-configs/evaluations/mathdial_discriminative_followup_v1.yaml}"

STAGES=(prepare_eval generate_responses oracle_v2 statistics report)
STATE_DIR="$OUTPUT_ROOT/stage_state"
LOG_DIR="$OUTPUT_ROOT/logs"
STATUS_FILE="$OUTPUT_ROOT/pipeline_status.json"
EVAL_DIR="$OUTPUT_ROOT/evaluation"
PROMPTS="$EVAL_DIR/prompts_ja.jsonl"
PROMPT_CANDIDATES="$EVAL_DIR/prompt_candidates_ja.jsonl"
SELECTION_MANIFEST="$EVAL_DIR/selection_manifest.json"
RESPONSES="$EVAL_DIR/responses.jsonl"
ORACLE_INPUT="$EVAL_DIR/oracle_input.jsonl"
PEDAGOGICAL_DIR="$EVAL_DIR/oracle/pedagogical_v2"
GENERAL_DIR="$EVAL_DIR/oracle/general"
STATISTICS_DIR="$EVAL_DIR/statistics"

SOURCE_SAMPLES="$SOURCE_RUN/mathdial/data/mathdial_assistant_samples.jsonl"
SOURCE_CONVERSATIONS="$SOURCE_RUN/mathdial/data/mathdial_conversations.jsonl"
V6_PROMPTS="$SOURCE_RUN/evaluation/prompts_ja.jsonl"
V11_PROMPTS="$EXCLUDE_NEUTRAL_RUN/evaluation/prompts_ja.jsonl"
BASIS_LORA="$SOURCE_RUN/training/basis_lora"
RANDOM_LORA="$SOURCE_RUN/training/random_lora"

mkdir -p "$STATE_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/instruction_discriminative_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

for required in "$SOURCE_SAMPLES" "$SOURCE_CONVERSATIONS" "$V6_PROMPTS" \
  "$V11_PROMPTS" "$QUOTA_CONFIG"; do
  [[ -f "$required" ]] || {
    echo "必要なsource成果物がありません: $required" >&2
    exit 20
  }
done
if [[ "$DRY_RUN" != "1" ]]; then
  for required in "$BASIS_LORA/adapter_model.safetensors" \
    "$RANDOM_LORA/adapter_model.safetensors"; do
    [[ -f "$required" ]] || {
      echo "必要なv6 adapterがありません: $required" >&2
      exit 20
    }
  done
fi

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

EXPERIMENT_FINGERPRINT="$(
  python3 - "$SOURCE_SAMPLES" "$SOURCE_CONVERSATIONS" "$V6_PROMPTS" \
    "$V11_PROMPTS" "$QUOTA_CONFIG" "$BASIS_LORA/adapter_config.json" \
    "$RANDOM_LORA/adapter_config.json" "$BASIS_LORA/adapter_model.safetensors" \
    "$RANDOM_LORA/adapter_model.safetensors" "$EVAL_COUNT" "$EVAL_SEED" \
    "$LOCAL_MODEL" "$TRANSLATION_MODEL" "$JUDGE_MODEL" <<'PY'
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

files = {path: sha256(path) for path in sys.argv[1:10]}
for path in (
    "core/dpo_prompting.py",
    "tools/mathdial_evaluation.py",
    "scripts/eval_oracle_mathdial_v2.py",
    "scripts/run_mathdial_statistics.py",
    "scripts/run_mathdial_instruction_discriminative_v2_pipeline.sh",
    "configs/evaluations/mathdial_oracle_v2.yaml",
):
    files[path] = sha256(path)
payload = {
    "files": files,
    "values": sys.argv[10:],
    "sampling_preset": "discriminative_followup",
    "local_prompt_mode": "mathdial_instruction",
    "prompt_template_version": "dpo_user_ai_instruction.v1",
    "adapter_policy": "reuse_v6_without_retraining",
}
print(hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest())
PY
)"

if [[ -s "$OUTPUT_ROOT/run_metadata.json" ]]; then
  existing_fingerprint="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment_fingerprint"])' \
      "$OUTPUT_ROOT/run_metadata.json"
  )"
  if [[ "$existing_fingerprint" != "$EXPERIMENT_FINGERPRINT" ]]; then
    echo "同じRUN_TAGの実験条件またはコードが変わっています。新しいRUN_TAGを使ってください。" >&2
    exit 20
  fi
else
  python3 - "$OUTPUT_ROOT/run_metadata.json" "$RUN_TAG" "$SOURCE_RUN" \
    "$EXCLUDE_NEUTRAL_RUN" "$EXPERIMENT_FINGERPRINT" "$EVAL_COUNT" \
    "$EVAL_SEED" "$LOCAL_MODEL" "$TRANSLATION_MODEL" "$JUDGE_MODEL" \
    "$QUOTA_CONFIG" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_tag": sys.argv[2],
    "source_run": sys.argv[3],
    "exclude_neutral_run": sys.argv[4],
    "experiment_fingerprint": sys.argv[5],
    "evaluation_count": int(sys.argv[6]),
    "evaluation_seed": int(sys.argv[7]),
    "models": {
        "local": sys.argv[8],
        "translation": sys.argv[9],
        "judge": sys.argv[10],
    },
    "quota_config": sys.argv[11],
    "status": "prospective_targeted_followup_after_subgroup_analysis",
    "adapter_policy": "reuse_v6_without_retraining",
}
path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
fi

write_status() {
  local state="$1" stage="$2" message="$3"
  python3 - "$STATUS_FILE" "$RUN_TAG" "$state" "$stage" "$message" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone
path = pathlib.Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_tag": sys.argv[2],
            "state": sys.argv[3],
            "stage": sys.argv[4],
            "message": sys.argv[5],
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY
}

stage_outputs_valid() {
  local expected_oracle=$((EVAL_COUNT * 3))
  case "$1" in
    prepare_eval)
      [[ -s "$PROMPTS" && -s "$SELECTION_MANIFEST" ]] &&
        [[ "$(wc -l < "$PROMPTS")" -eq "$EVAL_COUNT" ]]
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
        -f "$STATISTICS_DIR/posthoc_pairwise.csv" &&
        -s "$STATISTICS_DIR/stratum_model_summary.csv" &&
        -s "$STATISTICS_DIR/stratum_pairwise_summary.csv" ]]
      ;;
    report)
      [[ -s "$OUTPUT_ROOT/report.md" && -s "$OUTPUT_ROOT/manifest.json" ]]
      ;;
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
  [[ "$DRY_RUN" == "1" ]] && return 0
  command -v nvidia-smi >/dev/null || {
    echo "nvidia-smiが見つかりません。" >&2
    return 20
  }
  python3 - "$EVAL_CUDA_DEVICES" "$EVAL_MIN_FREE_MEMORY_MIB" <<'PY'
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
        f"評価生成用GPU空きメモリ不足 devices={missing} "
        f"required={minimum}MiB actual={free}"
    )
PY
}

prepare_eval_stage() {
  mkdir -p "$EVAL_DIR"
  local mock=()
  [[ "$DRY_RUN" == "1" ]] && mock+=(--mock)
  python3 - "$SOURCE_SAMPLES" "$SOURCE_CONVERSATIONS" "$V6_PROMPTS" \
    "$V11_PROMPTS" "$QUOTA_CONFIG" "$EVAL_COUNT" "$EVAL_SEED" <<'PY' || return 20
import sys
from pathlib import Path
from tools.mathdial_evaluation import (
    exclusion_ids_from_prompts,
    load_discriminative_quota_config,
    read_jsonl,
    select_discriminative_followup_prompts,
)
samples, conversations = read_jsonl(sys.argv[1]), read_jsonl(sys.argv[2])
excluded_samples, excluded_qids = exclusion_ids_from_prompts(
    [Path(sys.argv[3]), Path(sys.argv[4])]
)
quotas, reserve, _ = load_discriminative_quota_config(sys.argv[5])
if sum(quotas.values()) != int(sys.argv[6]):
    raise SystemExit("EVAL_COUNTとquota合計が一致しません。")
selected, _ = select_discriminative_followup_prompts(
    samples,
    conversations,
    quotas=quotas,
    reserve_per_stratum=reserve,
    seed=int(sys.argv[7]),
    excluded_sample_ids=excluded_samples,
    excluded_qids=excluded_qids,
)
print(
    f"[preflight] discriminative candidates={len(selected)} "
    f"primary={sum(quotas.values())} reserve={reserve * len(quotas)}"
)
PY
  python3 -m tools.mathdial_evaluation prepare \
    --samples "$SOURCE_SAMPLES" \
    --conversations "$SOURCE_CONVERSATIONS" \
    --output "$PROMPTS" \
    --candidate-output "$PROMPT_CANDIDATES" \
    --selection-manifest "$SELECTION_MANIFEST" \
    --errors-output "$EVAL_DIR/translation_errors.jsonl" \
    --skip-sample-errors \
    --count "$EVAL_COUNT" \
    --seed "$EVAL_SEED" \
    --model "$TRANSLATION_MODEL" \
    --exclude-prompts "$V6_PROMPTS" \
    --exclude-prompts "$V11_PROMPTS" \
    --sampling-preset discriminative_followup \
    --sampling-quota-config "$QUOTA_CONFIG" \
    --prompt-id-prefix "mathdial_instruction_discriminative_v1" \
    --local-prompt-mode mathdial_instruction \
    --resume \
    "${mock[@]}"
  python3 - "$PROMPTS" "$V6_PROMPTS" "$V11_PROMPTS" \
    "$SELECTION_MANIFEST" "$EVAL_COUNT" <<'PY'
import json
import pathlib
import sys
current = [
    json.loads(line)
    for line in pathlib.Path(sys.argv[1]).open(encoding="utf-8")
    if line.strip()
]
excluded = set()
for value in sys.argv[2:4]:
    excluded.update(
        str(json.loads(line)["qid"])
        for line in pathlib.Path(value).open(encoding="utf-8")
        if line.strip()
    )
manifest = json.loads(pathlib.Path(sys.argv[4]).read_text(encoding="utf-8"))
required = int(sys.argv[5])
if len(current) != required:
    raise SystemExit(f"識別力評価promptが不足しています: {len(current)}/{required}")
qids = {str(row["qid"]) for row in current}
conversations = {str(row["conversation_id"]) for row in current}
if len(qids) != required or len(conversations) != required:
    raise SystemExit("識別力評価内でqidまたはconversationが重複しています。")
if qids & excluded:
    raise SystemExit("v6/v11評価済みqidが識別力評価へ混入しています。")
if manifest.get("final_count") != required:
    raise SystemExit("selection manifestの最終件数が一致しません。")
for row in current:
    prompt = str(row.get("model_prompt", ""))
    if row.get("local_prompt_mode") != "mathdial_instruction":
        raise SystemExit("評価promptが旧MathDial instructionではありません。")
    if not prompt.startswith("以下の個別指導対話の次の教師返答を生成してください。"):
        raise SystemExit("旧MathDial instructionの先頭が変わっています。")
    if not prompt.endswith("AI:") or prompt.endswith("AI:\n"):
        raise SystemExit("v6互換prompt末尾がAI:ではありません。")
    if str(row["problem_ja"]).strip() not in prompt:
        raise SystemExit("評価promptから問題文が失われています。")
    forbidden = (
        "正解参照（日本語）:",
        "Reference answer:",
        "equitable_tutoring",
        "teacher_move_stage_alignment",
        "BASiS",
    )
    if any(token and token in prompt for token in forbidden):
        raise SystemExit("生成promptに評価・正解情報が混入しています。")
PY
}

generate_responses_stage() {
  local mock=()
  [[ "$DRY_RUN" == "1" ]] && mock+=(--mock)
  gpu_preflight || return 20
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
      --basis-lora "$BASIS_LORA" \
      --random-lora "$RANDOM_LORA" \
      --seed "$EVAL_SEED" \
      --local-prompt-mode mathdial_instruction \
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
    if row.get("local_prompt_mode") != "mathdial_instruction":
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
    --prompt-metadata "$PROMPTS" \
    --permutations "$permutations" \
    --bootstrap "$bootstrap" \
    --seed "$EVAL_SEED"
}

report_stage() {
  python3 - "$OUTPUT_ROOT" "$SOURCE_RUN" "$EXCLUDE_NEUTRAL_RUN" \
    "$EXPERIMENT_FINGERPRINT" "$EVAL_COUNT" "$EVAL_SEED" \
    "$TRANSLATION_MODEL" "$JUDGE_MODEL" "$LOCAL_MODEL" "$BASIS_LORA" \
    "$RANDOM_LORA" <<'PY'
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

root = pathlib.Path(sys.argv[1])
def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

selection = json.loads(
    (root / "evaluation/selection_manifest.json").read_text(encoding="utf-8")
)
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "prospective_targeted_followup_after_subgroup_analysis",
    "source_run": sys.argv[2],
    "exclude_neutral_run": sys.argv[3],
    "experiment_fingerprint": sys.argv[4],
    "evaluation_count": int(sys.argv[5]),
    "evaluation_seed": int(sys.argv[6]),
    "translation_model": sys.argv[7],
    "judge_model": sys.argv[8],
    "local_model": sys.argv[9],
    "local_prompt_mode": "mathdial_instruction",
    "prompt_template_version": "dpo_user_ai_instruction.v1",
    "adapter_policy": "reuse_v6_without_retraining",
    "known_limitation": (
        "v6学習時のtokenizer prompt/completion境界Mismatchを保持したadapterを"
        "再利用している。今回の評価処理では修復していない。"
    ),
    "basis_lora_path": sys.argv[10],
    "random_lora_path": sys.argv[11],
    "basis_adapter_sha256": sha256(
        pathlib.Path(sys.argv[10]) / "adapter_model.safetensors"
    ),
    "random_adapter_sha256": sha256(
        pathlib.Path(sys.argv[11]) / "adapter_model.safetensors"
    ),
    "selection": selection,
}
(root / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
sections = [
    "# MathDial v6 instruction discriminative follow-up",
    "",
    "過去の群別結果から立てた仮説を、v6/v11で未使用のtest qidで検証する"
    "識別力重視の追試。MathDial全体の無条件な主評価とは区別する。",
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
    ("Exploratory stratum means", "evaluation/statistics/stratum_model_summary.csv"),
    ("Exploratory stratum differences", "evaluation/statistics/stratum_pairwise_summary.csv"),
):
    sections.extend([
        "",
        f"## {title}",
        "",
        "```csv",
        (root / relative).read_text(encoding="utf-8").strip(),
        "```",
    ])
(root / "report.md").write_text(
    "\n".join(sections) + "\n",
    encoding="utf-8",
)
PY
}

trap 'status=$?; if [[ $status -ne 0 ]]; then write_status incomplete "${CURRENT_STAGE:-startup}" "pipeline exited status=$status" || true; fi' EXIT

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

echo "MathDial instruction discriminative follow-up completed: $OUTPUT_ROOT"
echo "Report: $OUTPUT_ROOT/report.md"
echo "Log: $LOG_FILE"
