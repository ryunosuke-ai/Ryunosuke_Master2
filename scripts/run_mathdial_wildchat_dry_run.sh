#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DRY_RUN=1
export RUN_TAG="${RUN_TAG:-mathdial_wildchat_v3_dry_run}"
export LIMIT="${LIMIT:-20}"
export WORKERS="${WORKERS:-1}"
exec "$SCRIPT_DIR/run_mathdial_wildchat_pipeline.sh" "$@"
