#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_TAG="${RUN_TAG:-esconv_topconf_three_model_10pt}"
DRY_RUN="${DRY_RUN:-0}"
if [ -z "${OUTPUT_ROOT:-}" ]; then
  if [ "$DRY_RUN" = "1" ]; then
    OUTPUT_ROOT="artifacts/evaluations/oracle_eval_runs/${RUN_TAG}_topconf_three_model_10pt_dry_run"
  else
    OUTPUT_ROOT="artifacts/evaluations/oracle_eval_runs/${RUN_TAG}_topconf_three_model_10pt"
  fi
fi

export RUN_TAG
export DRY_RUN
export OUTPUT_ROOT
export SCORE_SCALE="${SCORE_SCALE:-10}"
export PAIRWISE_TIE_THRESHOLD="${PAIRWISE_TIE_THRESHOLD:-0.25}"
export CATEGORY_DIR_SUFFIX="${CATEGORY_DIR_SUFFIX:-_10pt}"
export JUDGE_MODEL="${JUDGE_MODEL:-${ORACLE_MODEL:-gpt-5.4}}"
export ORACLE_WORKERS="${ORACLE_WORKERS:-4}"
export ORACLE_LOG_DIR="${ORACLE_LOG_DIR:-logs/oracle_evaluation/topconf_three_model_10pt}"

exec "${PROJECT_ROOT}/scripts/run_oracle_topconf_eval_three_models.sh"
