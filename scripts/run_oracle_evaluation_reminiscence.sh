#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-reminiscence_5000_to_2000}"
PROMPTS="${PROMPTS:-configs/evaluation_prompts/reminiscence_oracle_eval_v1.jsonl}"
SMALL_CORPUS="${SMALL_CORPUS:-data/small_corpus.jsonl}"
BAYES_MODEL="${BAYES_MODEL:-artifacts/bayes_models/generated_transition_bayes_model.json}"
BASE_MODEL_ID="${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"
LORA_PATH="${DPO_COMPARE_LORA_PATH:-artifacts/training_runs/qwen35_bayes_dpo_lora_${RUN_TAG}_ep1_lr5e-6_r8_a16_no4bit}"
ORACLE_MODEL="${ORACLE_MODEL:-gpt-5.4-pro}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/evaluations/oracle_eval_runs/${RUN_TAG}_oracle_v1}"
MAX_PROMPTS="${MAX_PROMPTS:-}"

LOG_DIR="${ORACLE_LOG_DIR:-logs/oracle_evaluation}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="${LOG_DIR}/oracle_eval_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Oracle evaluation started at $(date)"
echo "run_tag: $RUN_TAG"
echo "prompts: $PROMPTS"
echo "small_corpus: $SMALL_CORPUS"
echo "bayes_model: $BAYES_MODEL"
echo "base_model_id: $BASE_MODEL_ID"
echo "lora_path: $LORA_PATH"
echo "oracle_model: $ORACLE_MODEL"
echo "output_dir: $OUTPUT_DIR"
echo "log_file: $LOG_FILE"
echo "========================================"

args=(
  python3 -m tools.run_oracle_evaluation
  --prompts "$PROMPTS"
  --small-corpus "$SMALL_CORPUS"
  --bayes-model "$BAYES_MODEL"
  --base-model-id "$BASE_MODEL_ID"
  --lora-path "$LORA_PATH"
  --oracle-model "$ORACLE_MODEL"
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

"${args[@]}"

echo "========================================"
echo "Oracle evaluation completed at $(date)"
echo "summary: ${OUTPUT_DIR}/summary.json"
echo "responses: ${OUTPUT_DIR}/responses.jsonl"
echo "judgments: ${OUTPUT_DIR}/judgments.jsonl"
echo "========================================"
