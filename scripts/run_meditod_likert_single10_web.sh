#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET=meditod
export FORM_ROOT="${FORM_ROOT:-artifacts/meditod_wildchat/runs/meditod_wildchat_gpt56_v2/user_eval_v3_single10}"
export DEFINITION="${DEFINITION:-configs/user_evaluations/meditod_likert_v3_single10.yaml}"
export DATABASE="${DATABASE:-artifacts/user_eval/web/meditod_likert_v3_single10_responses.sqlite3}"
export PORT="${PORT:-8505}"

if [[ ! -s "$FORM_ROOT/experiment_a/form_items_public.jsonl" ]]; then
  OUTPUT_ROOT="$FORM_ROOT" DEFINITION="$DEFINITION" \
    "$SCRIPT_DIR/prepare_meditod_likert_single10.sh"
fi
exec "$SCRIPT_DIR/run_three_model_likert_user_eval_web.sh"
