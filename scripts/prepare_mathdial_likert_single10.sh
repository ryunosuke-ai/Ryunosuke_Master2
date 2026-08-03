#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOURCE_RUN="${SOURCE_RUN:-artifacts/mathdial_wildchat/evaluation_rechecks/mathdial_v6_instruction_outcome_selected_top100_v1}"
RESPONSES="${RESPONSES:-$SOURCE_RUN/evaluation/responses.jsonl}"
ORACLE_RAW="${ORACLE_RAW:-$SOURCE_RUN/evaluation/oracle/pedagogical_v2/raw.jsonl}"
DEFINITION="${DEFINITION:-configs/user_evaluations/mathdial_likert_v3_single10.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SOURCE_RUN/user_eval_v3_single10}"
SEED="${SEED:-42}"

python3 -m tools.prepare_three_model_likert_eval \
  --dataset mathdial \
  --responses "$RESPONSES" \
  --oracle-raw "$ORACLE_RAW" \
  --definition "$DEFINITION" \
  --output-root "$OUTPUT_ROOT" \
  --count 10 \
  --single-form \
  --seed "$SEED"
