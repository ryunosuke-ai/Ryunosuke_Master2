#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET=mathdial
export FORM_ROOT="${FORM_ROOT:-artifacts/mathdial_wildchat/evaluation_rechecks/mathdial_v6_instruction_outcome_selected_top100_v1/user_eval_v3_single10}"
export DEFINITION="${DEFINITION:-configs/user_evaluations/mathdial_likert_v3_single10.yaml}"
export DATABASE="${DATABASE:-artifacts/user_eval/web/mathdial_likert_v3_single10_responses.sqlite3}"
export PORT="${PORT:-8504}"

if [[ ! -s "$FORM_ROOT/experiment_a/form_items_public.jsonl" ]]; then
  OUTPUT_ROOT="$FORM_ROOT" DEFINITION="$DEFINITION" \
    "$SCRIPT_DIR/prepare_mathdial_likert_single10.sh"
fi
exec "$SCRIPT_DIR/run_three_model_likert_user_eval_web.sh"
