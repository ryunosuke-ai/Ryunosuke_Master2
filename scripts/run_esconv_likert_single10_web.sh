#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FORM_ROOT="${FORM_ROOT:-artifacts/user_eval/google_forms/esconv_human_reviewed_likert_single10_v8}"
export DATABASE="${DATABASE:-artifacts/user_eval/web/esconv_likert_single10_responses.sqlite3}"
export PORT="${PORT:-8503}"

if [[ ! -s "$FORM_ROOT/experiment_a/form_items_public.jsonl" ]]; then
  OUTPUT_ROOT="$FORM_ROOT" "$SCRIPT_DIR/prepare_esconv_likert_single10.sh"
fi
exec "$SCRIPT_DIR/run_esconv_likert_user_eval_web.sh"
