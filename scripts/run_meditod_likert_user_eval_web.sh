#!/usr/bin/env bash
set -euo pipefail
export DATASET=meditod
export PORT="${PORT:-8505}"
export DEFINITION="${DEFINITION:-configs/user_evaluations/meditod_likert_v2.yaml}"
export FORM_ROOT="${FORM_ROOT:-artifacts/meditod_wildchat/runs/meditod_wildchat_gpt56_v2/user_eval_v2_posthoc_axes}"
export DATABASE="${DATABASE:-artifacts/user_eval/web/meditod_likert_v2_responses.sqlite3}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_three_model_likert_user_eval_web.sh"
