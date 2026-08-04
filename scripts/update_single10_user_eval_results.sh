#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ESCONV_DATABASE="${ESCONV_DATABASE:-artifacts/user_eval/web/esconv_likert_single10_responses.sqlite3}"
MATHDIAL_DATABASE="${MATHDIAL_DATABASE:-artifacts/user_eval/web/mathdial_likert_v3_single10_responses.sqlite3}"
MEDITOD_DATABASE="${MEDITOD_DATABASE:-artifacts/user_eval/web/meditod_likert_v3_single10_responses.sqlite3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/user_eval/results}"

for database in "$ESCONV_DATABASE" "$MATHDIAL_DATABASE" "$MEDITOD_DATABASE"; do
  if [[ ! -s "$database" ]]; then
    echo "回答DBが存在しないか空です: $database" >&2
    exit 2
  fi
done

python3 -m tools.analyze_esconv_likert_responses \
  --database "$ESCONV_DATABASE" \
  --private-answer-key artifacts/user_eval/google_forms/esconv_human_reviewed_likert_single10_v8/experiment_a/answer_key_private.csv \
  --output-dir "$OUTPUT_ROOT/esconv_single10"

python3 -m tools.analyze_three_model_likert_responses \
  --database "$MATHDIAL_DATABASE" \
  --definition configs/user_evaluations/mathdial_likert_v3_single10.yaml \
  --private-answer-key artifacts/mathdial_wildchat/evaluation_rechecks/mathdial_v6_instruction_outcome_selected_top100_v1/user_eval_v3_single10/private_answer_key.jsonl \
  --output-dir "$OUTPUT_ROOT/mathdial_single10"

python3 -m tools.analyze_three_model_likert_responses \
  --database "$MEDITOD_DATABASE" \
  --definition configs/user_evaluations/meditod_likert_v3_single10.yaml \
  --private-answer-key artifacts/meditod_wildchat/runs/meditod_wildchat_gpt56_v2/user_eval_v3_single10/private_answer_key.jsonl \
  --output-dir "$OUTPUT_ROOT/meditod_single10"

echo "3データセットの最新集計を書き出しました: $OUTPUT_ROOT"
