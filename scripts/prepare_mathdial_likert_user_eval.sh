#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOURCE_RUN="${SOURCE_RUN:-artifacts/mathdial_wildchat/evaluation_rechecks/mathdial_v6_instruction_discriminative_followup_v1}"
RESPONSES="${RESPONSES:-$SOURCE_RUN/evaluation/responses.jsonl}"
ORACLE_RAW="${ORACLE_RAW:-$SOURCE_RUN/evaluation/oracle/pedagogical_v2/raw.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SOURCE_RUN/user_eval}"
COUNT="${COUNT:-20}"
SEED="${SEED:-42}"

[[ -s "$RESPONSES" ]] || { echo "MathDial 3モデル応答が見つかりません: $RESPONSES" >&2; exit 2; }
[[ -s "$ORACLE_RAW" ]] || { echo "MathDial Oracle rawが見つかりません: $ORACLE_RAW" >&2; exit 2; }
python3 -m tools.prepare_three_model_likert_eval \
  --dataset mathdial --responses "$RESPONSES" --oracle-raw "$ORACLE_RAW" \
  --output-root "$OUTPUT_ROOT" --count "$COUNT" --seed "$SEED"
