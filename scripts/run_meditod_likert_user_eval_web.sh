#!/usr/bin/env bash
set -euo pipefail
export DATASET=meditod
export PORT="${PORT:-8505}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_three_model_likert_user_eval_web.sh"
