#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# 発表用の既存ESConv成果物は過去RUN_TAGを引き継いでいる。
# 名前にreminiscenceを含むが、実体はESConv支援対話スタイル学習実験。
RUN_TAG="${RUN_TAG:-reminiscence_5000_to_2000}"
PROMPTS="${PROMPTS:-configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl}"
SMALL_CORPUS="${SMALL_CORPUS:-data/esconv_analysis_corpus_${RUN_TAG}.jsonl}"
BAYES_MODEL="${BAYES_MODEL:-artifacts/bayes_models/generated_transition_bayes_model_esconv_${RUN_TAG}.json}"
BASE_MODEL_ID="${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"
LORA_PATH="${DPO_COMPARE_LORA_PATH:-artifacts/training_runs/qwen35_bayes_dpo_lora_${RUN_TAG}_ep1_lr5e-6_r8_a16_no4bit}"
ORACLE_MODEL="${ORACLE_MODEL:-gpt-5.4-pro}"
ORACLE_WORKERS="${ORACLE_WORKERS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/evaluations/oracle_eval_runs/${RUN_TAG}_oracle_esconv_v3_strategy}"
MAX_PROMPTS="${MAX_PROMPTS:-}"
SKIP_PROMPTS="${SKIP_PROMPTS:-}"
CATEGORIES="${CATEGORIES:-}"
LOCAL_PROMPT_MODE="${LOCAL_PROMPT_MODE:-instruction}"
ORACLE_CUDA_VISIBLE_DEVICES="${ORACLE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
DPO_COMPARE_MAX_MEMORY="${DPO_COMPARE_MAX_MEMORY:-0=46GiB,1=46GiB,cpu=0GiB}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="${ORACLE_LOG_DIR:-logs/oracle_evaluation/esconv}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="${LOG_DIR}/oracle_eval_v3_strategy_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

export DPO_COMPARE_MAX_MEMORY
export PYTORCH_CUDA_ALLOC_CONF
if [ -n "$ORACLE_CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES="$ORACLE_CUDA_VISIBLE_DEVICES"
fi

echo "========================================"
echo "ESConv strategy Oracle v3 evaluation started at $(date)"
echo "run_tag: $RUN_TAG"
echo "prompts: $PROMPTS"
echo "small_corpus: $SMALL_CORPUS"
echo "bayes_model: $BAYES_MODEL"
echo "base_model_id: $BASE_MODEL_ID"
echo "lora_path: $LORA_PATH"
echo "oracle_model: $ORACLE_MODEL"
echo "oracle_workers: $ORACLE_WORKERS"
echo "output_dir: $OUTPUT_DIR"
echo "skip_prompts: ${SKIP_PROMPTS:-0}"
echo "categories: ${CATEGORIES:-all}"
echo "local_prompt_mode: $LOCAL_PROMPT_MODE"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-}"
echo "device_map: auto"
echo "max_memory: ${DPO_COMPARE_MAX_MEMORY:-}"
echo "pytorch_cuda_alloc_conf: ${PYTORCH_CUDA_ALLOC_CONF:-}"
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
  --oracle-workers "$ORACLE_WORKERS"
  --style-preset esconv_strategy_v3
  --output-dir "$OUTPUT_DIR"
  --seed 42
  --max-new-tokens 192
  --temperature 0.7
  --top-p 0.8
  --repetition-penalty 1.0
  --local-prompt-mode "$LOCAL_PROMPT_MODE"
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
echo "ESConv strategy Oracle v3 evaluation completed at $(date)"
echo "summary: ${OUTPUT_DIR}/summary.json"
echo "responses: ${OUTPUT_DIR}/responses.jsonl"
echo "judgments: ${OUTPUT_DIR}/judgments.jsonl"
echo "========================================"
