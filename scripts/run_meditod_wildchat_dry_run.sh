#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export RUN_TAG="${RUN_TAG:-meditod_wildchat_dry_run}"
export DRY_RUN=1
export START_STAGE="${START_STAGE:-preprocess}"
export END_STAGE="${END_STAGE:-prepare_user_eval}"
export WORKERS="${WORKERS:-2}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
exec ./scripts/run_meditod_wildchat_pipeline.sh
