#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-esconv_5000_to_2000_bayes_vs_random2500}"
PROMPTS="${PROMPTS:-configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl}"
SMALL_CORPUS="${SMALL_CORPUS:-data/esconv_analysis_corpus_reminiscence_5000_to_2000.jsonl}"
BAYES_MODEL="${BAYES_MODEL:-artifacts/bayes_models/generated_transition_bayes_model_esconv_reminiscence_5000_to_2000.json}"
BASE_MODEL_ID="${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"
BAYES_DPO_LORA_PATH="${BAYES_DPO_LORA_PATH:-artifacts/training_runs/qwen35_bayes_dpo_lora_reminiscence_5000_to_2000_ep1_lr5e-6_r8_a16_no4bit}"
RANDOM_DPO_RUN_TAG="${RANDOM_DPO_RUN_TAG:-esconv_5000_to_2000_random2500}"
RANDOM_DPO_LORA_PATH="${RANDOM_DPO_LORA_PATH:-artifacts/training_runs/qwen35_random2500_dailydialog_dpo_lora_${RANDOM_DPO_RUN_TAG}_ep1_lr5e-6_r8_a16_no4bit}"
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

require_lora() {
  local label="$1"
  local path="$2"
  local next_command="$3"

  if [ ! -d "$path" ] || [ ! -f "$path/adapter_config.json" ]; then
    echo "${label} LoRA not found:"
    echo "$path"
    if [ -n "$next_command" ]; then
      echo
      echo "Please run this first after the current Oracle evaluation finishes:"
      echo "$next_command"
    fi
    exit 1
  fi
}

require_lora "Bayes-DPO" "$BAYES_DPO_LORA_PATH" ""
require_lora "Random-DPO" "$RANDOM_DPO_LORA_PATH" "./scripts/run_dpo_pipeline_esconv_random_2500.sh"

LOG_DIR="${ORACLE_LOG_DIR:-logs/oracle_evaluation/bayes_vs_random}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="${LOG_DIR}/oracle_eval_v3_strategy_bayes_vs_random_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

export DPO_COMPARE_MAX_MEMORY
export PYTORCH_CUDA_ALLOC_CONF
if [ -n "$ORACLE_CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES="$ORACLE_CUDA_VISIBLE_DEVICES"
fi

echo "========================================"
echo "ESConv strategy Oracle v3 LoRA pair evaluation started at $(date)"
echo "comparison_kind: lora_pair"
echo "base field = bayes_dpo"
echo "dpo field = random_dpo"
echo "bayes_dpo_win_rate = base_win_rate"
echo "random_dpo_win_rate = dpo_win_rate"
echo "run_tag: $RUN_TAG"
echo "prompts: $PROMPTS"
echo "small_corpus: $SMALL_CORPUS"
echo "bayes_model: $BAYES_MODEL"
echo "base_model_id: $BASE_MODEL_ID"
echo "bayes_dpo_lora_path: $BAYES_DPO_LORA_PATH"
echo "random_dpo_lora_path: $RANDOM_DPO_LORA_PATH"
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
  python3 -m tools.run_oracle_evaluation_lora_pair
  --prompts "$PROMPTS"
  --small-corpus "$SMALL_CORPUS"
  --bayes-model "$BAYES_MODEL"
  --base-model-id "$BASE_MODEL_ID"
  --base-lora-path "$BAYES_DPO_LORA_PATH"
  --dpo-lora-path "$RANDOM_DPO_LORA_PATH"
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
echo "ESConv strategy Oracle v3 LoRA pair evaluation completed at $(date)"
echo "summary: ${OUTPUT_DIR}/summary.json"
echo "responses: ${OUTPUT_DIR}/responses.jsonl"
echo "judgments: ${OUTPUT_DIR}/judgments.jsonl"
echo "base field = bayes_dpo"
echo "dpo field = random_dpo"
echo "bayes_dpo_win_rate = base_win_rate"
echo "random_dpo_win_rate = dpo_win_rate"
echo "========================================"
