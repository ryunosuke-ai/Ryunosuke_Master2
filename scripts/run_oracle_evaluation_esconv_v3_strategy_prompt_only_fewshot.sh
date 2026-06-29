#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-esconv_prompt_only_fewshot}"
PROMPTS="${PROMPTS:-configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl}"
SMALL_CORPUS="${SMALL_CORPUS:-data/esconv_analysis_corpus_reminiscence_5000_to_2000.jsonl}"
BAYES_MODEL="${BAYES_MODEL:-artifacts/bayes_models/generated_transition_bayes_model_esconv_reminiscence_5000_to_2000.json}"
FEWSHOT_EXAMPLES="${FEWSHOT_EXAMPLES:-artifacts/datasets/esconv_gold_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl}"
FEWSHOT_COUNT="${FEWSHOT_COUNT:-8}"
BASE_MODEL_ID="${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"
ORACLE_MODEL="${ORACLE_MODEL:-gpt-5.4}"
ORACLE_WORKERS="${ORACLE_WORKERS:-2}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy_gpt54/summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/evaluations/oracle_eval_runs/${RUN_TAG}_oracle_esconv_v3_strategy_gpt54}"
MAX_PROMPTS="${MAX_PROMPTS:-}"
SKIP_PROMPTS="${SKIP_PROMPTS:-}"
CATEGORIES="${CATEGORIES:-}"
ORACLE_CUDA_VISIBLE_DEVICES="${ORACLE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
DPO_COMPARE_MAX_MEMORY="${DPO_COMPARE_MAX_MEMORY:-0=46GiB,1=46GiB,cpu=0GiB}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="${ORACLE_LOG_DIR:-logs/oracle_evaluation/prompt_only_fewshot}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="${LOG_DIR}/oracle_eval_v3_strategy_prompt_only_fewshot_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

export DPO_COMPARE_MAX_MEMORY
export PYTORCH_CUDA_ALLOC_CONF
if [ -n "$ORACLE_CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES="$ORACLE_CUDA_VISIBLE_DEVICES"
fi

echo "========================================"
echo "Prompt-only few-shot ESConv strategy Oracle v3 evaluation started at $(date)"
echo "run_tag: $RUN_TAG"
echo "prompts: $PROMPTS"
echo "small_corpus: $SMALL_CORPUS"
echo "bayes_model: $BAYES_MODEL"
echo "fewshot_examples: $FEWSHOT_EXAMPLES"
echo "fewshot_count: $FEWSHOT_COUNT"
echo "base_model_id: $BASE_MODEL_ID"
echo "oracle_model: $ORACLE_MODEL"
echo "oracle_workers: $ORACLE_WORKERS"
echo "baseline_summary: $BASELINE_SUMMARY"
echo "output_dir: $OUTPUT_DIR"
echo "skip_prompts: ${SKIP_PROMPTS:-0}"
echo "categories: ${CATEGORIES:-all}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-}"
echo "device_map: auto"
echo "max_memory: ${DPO_COMPARE_MAX_MEMORY:-}"
echo "pytorch_cuda_alloc_conf: ${PYTORCH_CUDA_ALLOC_CONF:-}"
echo "log_file: $LOG_FILE"
echo "========================================"

args=(
  python3 -m tools.run_oracle_evaluation_prompt_only
  --prompts "$PROMPTS"
  --small-corpus "$SMALL_CORPUS"
  --bayes-model "$BAYES_MODEL"
  --fewshot-examples "$FEWSHOT_EXAMPLES"
  --fewshot-count "$FEWSHOT_COUNT"
  --base-model-id "$BASE_MODEL_ID"
  --oracle-model "$ORACLE_MODEL"
  --oracle-workers "$ORACLE_WORKERS"
  --style-preset esconv_strategy_v3
  --baseline-summary "$BASELINE_SUMMARY"
  --output-dir "$OUTPUT_DIR"
  --seed 42
  --max-new-tokens 192
  --temperature 0.7
  --top-p 0.8
  --repetition-penalty 1.0
  --no-4bit
)

if [ -n "$MAX_PROMPTS" ]; then
  args+=(--max-prompts "$MAX_PROMPTS")
fi
if [ -n "$SKIP_PROMPTS" ]; then
  args+=(--skip-prompts "$SKIP_PROMPTS")
fi
if [ -n "$CATEGORIES" ]; then
  args+=(--categories "$CATEGORIES")
fi

"${args[@]}"

echo "========================================"
echo "Prompt-only few-shot ESConv strategy Oracle v3 evaluation completed at $(date)"
echo "summary: ${OUTPUT_DIR}/summary.json"
echo "responses: ${OUTPUT_DIR}/responses.jsonl"
echo "judgments: ${OUTPUT_DIR}/judgments.jsonl"
echo "========================================"
