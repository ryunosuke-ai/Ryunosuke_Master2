#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOURCE_RUN="${SOURCE_RUN:-artifacts/meditod_wildchat/runs/meditod_wildchat_gpt56_v2}"
RESPONSES="${RESPONSES:-$SOURCE_RUN/evaluation/responses.jsonl}"
HISTORY_ORACLE_RAW="${HISTORY_ORACLE_RAW:-$SOURCE_RUN/evaluation/oracle/history/raw.jsonl}"
SAFETY_ORACLE_RAW="${SAFETY_ORACLE_RAW:-$SOURCE_RUN/evaluation/oracle/safety/raw.jsonl}"
DEFINITION="${DEFINITION:-configs/user_evaluations/meditod_likert_v2.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SOURCE_RUN/user_eval_v2_posthoc_axes}"
COUNT="${COUNT:-20}"
SEED="${SEED:-42}"

for path in "$RESPONSES" "$HISTORY_ORACLE_RAW" "$SAFETY_ORACLE_RAW" "$DEFINITION"; do
  [[ -s "$path" ]] || {
    echo "MediTOD人手評価入力が見つかりません: $path" >&2
    exit 2
  }
done

python3 -m tools.prepare_three_model_likert_eval \
  --dataset meditod \
  --responses "$RESPONSES" \
  --oracle-raw "$HISTORY_ORACLE_RAW" \
  --oracle-raw "$SAFETY_ORACLE_RAW" \
  --definition "$DEFINITION" \
  --output-root "$OUTPUT_ROOT" \
  --count "$COUNT" \
  --seed "$SEED"
