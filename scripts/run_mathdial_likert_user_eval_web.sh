#!/usr/bin/env bash
set -euo pipefail
export DATASET=mathdial
export PORT="${PORT:-8504}"
export DEFINITION="${DEFINITION:-configs/user_evaluations/mathdial_likert_v2.yaml}"
export FORM_ROOT="${FORM_ROOT:-artifacts/mathdial_wildchat/evaluation_rechecks/mathdial_v6_instruction_outcome_selected_top100_v1/user_eval_v2_posthoc_axes}"
export DATABASE="${DATABASE:-artifacts/user_eval/web/mathdial_likert_v2_responses.sqlite3}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_three_model_likert_user_eval_web.sh"
