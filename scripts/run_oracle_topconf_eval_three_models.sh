#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-esconv_topconf_three_model}"
BASE_VS_BAYES_RESPONSES="${BASE_VS_BAYES_RESPONSES:-artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy/responses.jsonl}"
BAYES_VS_RANDOM_RESPONSES="${BAYES_VS_RANDOM_RESPONSES:-artifacts/evaluations/oracle_eval_runs/esconv_5000_to_2000_bayes_vs_random2500_oracle_esconv_v3_strategy/responses.jsonl}"
JUDGE_MODEL="${JUDGE_MODEL:-${ORACLE_MODEL:-gpt-5.4}}"
ORACLE_WORKERS="${ORACLE_WORKERS:-4}"
SCORE_SCALE="${SCORE_SCALE:-5}"
PAIRWISE_TIE_THRESHOLD="${PAIRWISE_TIE_THRESHOLD:-}"
CATEGORY_DIR_SUFFIX="${CATEGORY_DIR_SUFFIX:-}"
CONVERSATION_STYLE_SCRIPT="${CONVERSATION_STYLE_SCRIPT:-scripts/eval_oracle_conversation_style.py}"
STRATEGY_TRANSITION_SCRIPT="${STRATEGY_TRANSITION_SCRIPT:-scripts/eval_oracle_strategy_transition.py}"
TST_SCRIPT="${TST_SCRIPT:-scripts/eval_oracle_tst.py}"
USR_QUALITY_SCRIPT="${USR_QUALITY_SCRIPT:-scripts/eval_oracle_usr_quality.py}"
CONVERSATION_STYLE_EVAL_NAME="${CONVERSATION_STYLE_EVAL_NAME:-conversation_style}"
STRATEGY_TRANSITION_EVAL_NAME="${STRATEGY_TRANSITION_EVAL_NAME:-strategy_transition}"
TST_EVAL_NAME="${TST_EVAL_NAME:-text_style_transfer}"
USR_QUALITY_EVAL_NAME="${USR_QUALITY_EVAL_NAME:-usr_quality}"
CONVERSATION_STYLE_OUTPUT_BASENAME="${CONVERSATION_STYLE_OUTPUT_BASENAME:-oracle_conversation_style}"
STRATEGY_TRANSITION_OUTPUT_BASENAME="${STRATEGY_TRANSITION_OUTPUT_BASENAME:-oracle_strategy_transition}"
TST_OUTPUT_BASENAME="${TST_OUTPUT_BASENAME:-oracle_tst}"
USR_QUALITY_OUTPUT_BASENAME="${USR_QUALITY_OUTPUT_BASENAME:-oracle_usr_quality}"
RUN_CONVERSATION_STYLE="${RUN_CONVERSATION_STYLE:-1}"
RUN_STRATEGY_TRANSITION="${RUN_STRATEGY_TRANSITION:-1}"
RUN_TST="${RUN_TST:-1}"
RUN_USR_QUALITY="${RUN_USR_QUALITY:-1}"
RUN_SIGNIFICANCE="${RUN_SIGNIFICANCE:-0}"
SIGNIFICANCE_N_PERMUTATIONS="${SIGNIFICANCE_N_PERMUTATIONS:-10000}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"
if [ -z "${OUTPUT_ROOT:-}" ]; then
  if [ "$DRY_RUN" = "1" ]; then
    OUTPUT_ROOT="artifacts/evaluations/oracle_eval_runs/${RUN_TAG}_dry_run"
  else
    OUTPUT_ROOT="artifacts/evaluations/oracle_eval_runs/${RUN_TAG}"
  fi
fi
SIGNIFICANCE_OUTPUT_DIR="${SIGNIFICANCE_OUTPUT_DIR:-${OUTPUT_ROOT}/significance_tests}"
MERGED_INPUT="${MERGED_INPUT:-${OUTPUT_ROOT}/three_model_responses.jsonl}"
EVAL_INPUT="$MERGED_INPUT"
MAX_RETRIES="${MAX_RETRIES:-5}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
LOG_DIR="${ORACLE_LOG_DIR:-logs/oracle_evaluation/topconf_three_model}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Top conference Oracle three-model evaluation started at $(date)"
echo "run_tag: $RUN_TAG"
echo "base_vs_bayes_responses: $BASE_VS_BAYES_RESPONSES"
echo "bayes_vs_random_responses: $BAYES_VS_RANDOM_RESPONSES"
echo "merged_input: $MERGED_INPUT"
echo "output_root: $OUTPUT_ROOT"
echo "judge_model: $JUDGE_MODEL"
echo "oracle_workers: $ORACLE_WORKERS"
echo "score_scale: $SCORE_SCALE"
echo "pairwise_tie_threshold: ${PAIRWISE_TIE_THRESHOLD:-auto}"
echo "run_conversation_style: $RUN_CONVERSATION_STYLE ($CONVERSATION_STYLE_SCRIPT)"
echo "run_strategy_transition: $RUN_STRATEGY_TRANSITION ($STRATEGY_TRANSITION_SCRIPT)"
echo "run_tst: $RUN_TST ($TST_SCRIPT)"
echo "run_usr_quality: $RUN_USR_QUALITY ($USR_QUALITY_SCRIPT)"
echo "run_significance: $RUN_SIGNIFICANCE"
echo "limit: ${LIMIT:-all}"
echo "dry_run: $DRY_RUN"
echo "log_file: $LOG_FILE"
echo "========================================"

python3 - "$BASE_VS_BAYES_RESPONSES" "$BAYES_VS_RANDOM_RESPONSES" "$MERGED_INPUT" "$RUN_TAG" <<'PY'
import json
import sys
from pathlib import Path
from typing import Any


base_vs_bayes_path = Path(sys.argv[1])
bayes_vs_random_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
run_tag = sys.argv[4]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"入力ファイルがありません: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} はJSON objectである必要があります。")
            records.append(payload)
    if not records:
        raise ValueError(f"入力ファイルに有効なレコードがありません: {path}")
    return records


def prompt_id(record: dict[str, Any]) -> str:
    value = str(record.get("prompt_id") or record.get("sample_id") or record.get("id") or "").strip()
    if not value:
        raise ValueError(f"prompt_idを取得できません: {record}")
    return value


def by_prompt_id(records: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        key = prompt_id(record)
        if key in indexed:
            raise ValueError(f"{label}でprompt_idが重複しています: {key}")
        indexed[key] = record
    return indexed


def normalized_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_text(value: Any) -> str:
    return str(value or "").strip()


def assert_same(left: dict[str, Any], right: dict[str, Any], key: str, prompt_key: str) -> None:
    if normalized_json(left.get(key)) != normalized_json(right.get(key)):
        raise ValueError(f"{prompt_key}: {key} が統合元2ファイルで一致しません。")


base_vs_bayes = by_prompt_id(read_jsonl(base_vs_bayes_path), label="Base vs Bayes")
bayes_vs_random = by_prompt_id(read_jsonl(bayes_vs_random_path), label="Bayes vs Random")

left_ids = set(base_vs_bayes)
right_ids = set(bayes_vs_random)
if left_ids != right_ids:
    only_left = sorted(left_ids - right_ids)[:20]
    only_right = sorted(right_ids - left_ids)[:20]
    raise ValueError(
        "prompt_id集合が一致しません。"
        f"Base vs Bayesのみ: {only_left} / Bayes vs Randomのみ: {only_right}"
    )

merged: list[dict[str, Any]] = []
for key in sorted(left_ids):
    left = base_vs_bayes[key]
    right = bayes_vs_random[key]
    for common_key in ("prompt", "history", "category", "axis_focus"):
        assert_same(left, right, common_key, key)

    base_response = normalized_text(left.get("base_response"))
    bayes_response_from_left = normalized_text(left.get("dpo_response"))
    bayes_response_from_right = normalized_text(right.get("base_response"))
    random_response = normalized_text(right.get("dpo_response"))
    if not base_response:
        raise ValueError(f"{key}: Base応答が空です。")
    if not bayes_response_from_left:
        raise ValueError(f"{key}: Base vs Bayes側のBayes-DPO応答が空です。")
    if not bayes_response_from_right:
        raise ValueError(f"{key}: Bayes vs Random側のBayes-DPO応答が空です。")
    if bayes_response_from_left != bayes_response_from_right:
        raise ValueError(f"{key}: Bayes-DPO応答が統合元2ファイルで一致しません。")
    if not random_response:
        raise ValueError(f"{key}: Random-DPO応答が空です。")

    merged.append(
        {
            "prompt_id": key,
            "category": left.get("category", ""),
            "prompt": left.get("prompt", ""),
            "history": left.get("history", []),
            "axis_focus": left.get("axis_focus", []),
            "base_response": base_response,
            "bayes_dpo_response": bayes_response_from_left,
            "random_dpo_response": random_response,
            "response_source_mapping": {
                "base_response": "Base",
                "bayes_dpo_response": "BASiS/Bayes-DPO",
                "random_dpo_response": "Random-DPO",
            },
            "source_files": {
                "base_vs_bayes_responses": base_vs_bayes_path.as_posix(),
                "bayes_vs_random_responses": bayes_vs_random_path.as_posix(),
            },
            "run_tag": run_tag,
        }
    )

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as file:
    for record in merged:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"3モデル評価入力を書き出しました: {output_path} ({len(merged)} records)")
PY

if [ -n "$LIMIT" ]; then
  EVAL_INPUT="${OUTPUT_ROOT}/three_model_responses_limit${LIMIT}.jsonl"
  python3 - "$MERGED_INPUT" "$EVAL_INPUT" "$LIMIT" <<'PY'
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
limit = int(sys.argv[3])
if limit <= 0:
    raise ValueError("LIMITは1以上にしてください。")

lines = [line for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
selected = lines[:limit]
if len(selected) < limit:
    raise ValueError(f"LIMIT={limit} に対して入力は {len(selected)} records しかありません。")
output_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
print(f"LIMIT={limit} の評価入力を書き出しました: {output_path} ({len(selected)} prompt records)")
PY
fi

COMMON_ARGS=(
  --input "$EVAL_INPUT"
  --judge_model "$JUDGE_MODEL"
  --oracle-workers "$ORACLE_WORKERS"
  --max_retries "$MAX_RETRIES"
  --max-output-tokens "$MAX_OUTPUT_TOKENS"
  --temperature "$TEMPERATURE"
  --seed "$SEED"
  --score-scale "$SCORE_SCALE"
  --resume
)
if [ -n "$PAIRWISE_TIE_THRESHOLD" ]; then
  COMMON_ARGS+=(--pairwise-tie-threshold "$PAIRWISE_TIE_THRESHOLD")
fi
if [ "$DRY_RUN" = "1" ]; then
  COMMON_ARGS+=(--dry-run)
fi

update_metadata() {
  local metadata_path="$1"
  local eval_name="$2"
  python3 - "$metadata_path" "$eval_name" "$RUN_TAG" "$LIMIT" "$DRY_RUN" "$JUDGE_MODEL" "$ORACLE_WORKERS" "$SCORE_SCALE" "$PAIRWISE_TIE_THRESHOLD" "$BASE_VS_BAYES_RESPONSES" "$BAYES_VS_RANDOM_RESPONSES" "$MERGED_INPUT" "$EVAL_INPUT" "$LOG_FILE" <<'PY'
import json
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
eval_name = sys.argv[2]
run_tag = sys.argv[3]
limit = sys.argv[4]
dry_run = sys.argv[5]
judge_model = sys.argv[6]
oracle_workers = int(sys.argv[7])
score_scale = int(sys.argv[8])
pairwise_tie_threshold = sys.argv[9]
base_vs_bayes = sys.argv[10]
bayes_vs_random = sys.argv[11]
merged_input = sys.argv[12]
eval_input = sys.argv[13]
log_file = sys.argv[14]

payload = json.loads(metadata_path.read_text(encoding="utf-8"))
resolved_threshold = payload.get("pairwise_tie_threshold")
if pairwise_tie_threshold != "":
    resolved_threshold = float(pairwise_tie_threshold)
payload.update(
    {
        "run_tag": run_tag,
        "topconf_eval_name": eval_name,
        "limit": None if limit == "" else int(limit),
        "dry_run": dry_run == "1",
        "judge_model": judge_model,
        "oracle_workers": oracle_workers,
        "score_scale": score_scale,
        "score_min": 1,
        "score_max": score_scale,
        "pairwise_tie_threshold": resolved_threshold,
        "evaluation_scale_name": "10-point Oracle evaluation" if score_scale == 10 else "5-point Oracle evaluation",
        "is_10_point_evaluation": score_scale == 10,
        "separate_from_5_point_evaluation": score_scale == 10,
        "three_model_response_mapping": {
            "BASiS": "Bayes-DPO",
            "Base": "base_response",
            "Random-DPO": "random_dpo_response",
            "bayes_dpo_response": "BASiS/Bayes-DPO",
        },
        "source_files": {
            "base_vs_bayes_responses": base_vs_bayes,
            "bayes_vs_random_responses": bayes_vs_random,
            "merged_input": merged_input,
            "eval_input": eval_input,
            "log_file": log_file,
        },
    }
)
metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"metadataを更新しました: {metadata_path}")
PY
}

run_eval() {
  local eval_name="$1"
  local script_path="$2"
  local output_dir="$3"
  echo "----------------------------------------"
  echo "Running ${eval_name}: ${script_path}"
  echo "output_dir: ${output_dir}"
  python3 "$script_path" "${COMMON_ARGS[@]}" --output_dir "$output_dir"
  update_metadata "${output_dir}/metadata.json" "$eval_name"
}

SIGNIFICANCE_CATEGORIES=()

if [ "$RUN_CONVERSATION_STYLE" = "1" ]; then
  run_eval "$CONVERSATION_STYLE_EVAL_NAME" "$CONVERSATION_STYLE_SCRIPT" "${OUTPUT_ROOT}/${CONVERSATION_STYLE_OUTPUT_BASENAME}${CATEGORY_DIR_SUFFIX}"
  SIGNIFICANCE_CATEGORIES+=(--category "${CONVERSATION_STYLE_EVAL_NAME}=${CONVERSATION_STYLE_OUTPUT_BASENAME}${CATEGORY_DIR_SUFFIX}")
fi
if [ "$RUN_STRATEGY_TRANSITION" = "1" ]; then
  run_eval "$STRATEGY_TRANSITION_EVAL_NAME" "$STRATEGY_TRANSITION_SCRIPT" "${OUTPUT_ROOT}/${STRATEGY_TRANSITION_OUTPUT_BASENAME}${CATEGORY_DIR_SUFFIX}"
  SIGNIFICANCE_CATEGORIES+=(--category "${STRATEGY_TRANSITION_EVAL_NAME}=${STRATEGY_TRANSITION_OUTPUT_BASENAME}${CATEGORY_DIR_SUFFIX}")
fi
if [ "$RUN_TST" = "1" ]; then
  run_eval "$TST_EVAL_NAME" "$TST_SCRIPT" "${OUTPUT_ROOT}/${TST_OUTPUT_BASENAME}${CATEGORY_DIR_SUFFIX}"
  SIGNIFICANCE_CATEGORIES+=(--category "${TST_EVAL_NAME}=${TST_OUTPUT_BASENAME}${CATEGORY_DIR_SUFFIX}")
fi
if [ "$RUN_USR_QUALITY" = "1" ]; then
  run_eval "$USR_QUALITY_EVAL_NAME" "$USR_QUALITY_SCRIPT" "${OUTPUT_ROOT}/${USR_QUALITY_OUTPUT_BASENAME}${CATEGORY_DIR_SUFFIX}"
  SIGNIFICANCE_CATEGORIES+=(--category "${USR_QUALITY_EVAL_NAME}=${USR_QUALITY_OUTPUT_BASENAME}${CATEGORY_DIR_SUFFIX}")
fi

if [ "$RUN_SIGNIFICANCE" = "1" ]; then
  if [ "${#SIGNIFICANCE_CATEGORIES[@]}" -eq 0 ]; then
    echo "RUN_SIGNIFICANCE=1 ですが、実行対象カテゴリがありません。"
  else
    echo "----------------------------------------"
    echo "Running three-model significance tests"
    echo "output_dir: $SIGNIFICANCE_OUTPUT_DIR"
    python3 scripts/analyze_oracle_three_model_significance.py \
      --root "$OUTPUT_ROOT" \
      --output_dir "$SIGNIFICANCE_OUTPUT_DIR" \
      --tie_threshold "${PAIRWISE_TIE_THRESHOLD:-0.25}" \
      --n_permutations "$SIGNIFICANCE_N_PERMUTATIONS" \
      --seed "$SEED" \
      "${SIGNIFICANCE_CATEGORIES[@]}"
  fi
fi

echo "========================================"
echo "Top conference Oracle three-model evaluation completed at $(date)"
echo "merged_input: $MERGED_INPUT"
echo "output_root: $OUTPUT_ROOT"
echo "========================================"
