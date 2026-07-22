#!/usr/bin/env bash
set -euo pipefail
export DATASET=mathdial
export PORT="${PORT:-8504}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_three_model_likert_user_eval_web.sh"
