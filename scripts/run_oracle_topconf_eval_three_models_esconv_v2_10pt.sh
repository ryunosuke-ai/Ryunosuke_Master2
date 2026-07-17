#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_TAG="${RUN_TAG:-esconv_topconf_three_model_esconv_v2_10pt}"
DRY_RUN="${DRY_RUN:-0}"
if [ -z "${OUTPUT_ROOT:-}" ]; then
  if [ "$DRY_RUN" = "1" ]; then
    OUTPUT_ROOT="artifacts/evaluations/oracle_eval_runs/${RUN_TAG}_topconf_three_model_esconv_v2_10pt_dry_run"
  else
    OUTPUT_ROOT="artifacts/evaluations/oracle_eval_runs/${RUN_TAG}_topconf_three_model_esconv_v2_10pt"
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
export ORACLE_LOG_DIR="${ORACLE_LOG_DIR:-logs/oracle_evaluation/topconf_three_model_esconv_v2_10pt}"

export CONVERSATION_STYLE_SCRIPT="${CONVERSATION_STYLE_SCRIPT:-scripts/eval_oracle_conversation_style_esconv_v2.py}"
export STRATEGY_TRANSITION_SCRIPT="${STRATEGY_TRANSITION_SCRIPT:-scripts/eval_oracle_strategy_transition_esconv_v2.py}"
export CONVERSATION_STYLE_EVAL_NAME="${CONVERSATION_STYLE_EVAL_NAME:-conversation_style_esconv_v2}"
export STRATEGY_TRANSITION_EVAL_NAME="${STRATEGY_TRANSITION_EVAL_NAME:-strategy_transition_esconv_v2}"
export CONVERSATION_STYLE_OUTPUT_BASENAME="${CONVERSATION_STYLE_OUTPUT_BASENAME:-oracle_conversation_style_esconv_v2}"
export STRATEGY_TRANSITION_OUTPUT_BASENAME="${STRATEGY_TRANSITION_OUTPUT_BASENAME:-oracle_strategy_transition_esconv_v2}"

export RUN_CONVERSATION_STYLE="${RUN_CONVERSATION_STYLE:-1}"
export RUN_STRATEGY_TRANSITION="${RUN_STRATEGY_TRANSITION:-1}"
export RUN_TST="${RUN_TST:-0}"
export RUN_USR_QUALITY="${RUN_USR_QUALITY:-0}"
export RUN_SIGNIFICANCE="${RUN_SIGNIFICANCE:-1}"
export SIGNIFICANCE_OUTPUT_DIR="${SIGNIFICANCE_OUTPUT_DIR:-${OUTPUT_ROOT}/significance_tests_esconv_v2_10pt}"
export SIGNIFICANCE_N_PERMUTATIONS="${SIGNIFICANCE_N_PERMUTATIONS:-10000}"

exec "${PROJECT_ROOT}/scripts/run_oracle_topconf_eval_three_models.sh"
