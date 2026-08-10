#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

DATASET="${DATASET:?DATASET=esconv|mathdial|meditod を指定してください}"
case "$DATASET" in esconv|mathdial|meditod) ;; *) echo "未対応dataset: $DATASET" >&2; exit 20 ;; esac
RUN_TAG="${RUN_TAG:-gold_only_dpo500_v1}"
CONFIG="${GOLD_ONLY_CONFIG:-configs/experiments/gold_only_dpo500_v1.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/gold_only_dpo/runs/$RUN_TAG/$DATASET}"
START_STAGE="${START_STAGE:-prepare_data}"
END_STAGE="${END_STAGE:-report}"
STAGE="${STAGE:-}"
[[ -n "$STAGE" ]] && { START_STAGE="$STAGE"; END_STAGE="$STAGE"; }
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1}"
TRAIN_DEVICE_MAP="${TRAIN_DEVICE_MAP:-auto}"
TRAIN_MAX_MEMORY="${TRAIN_MAX_MEMORY:-0=46GiB,1=46GiB,cpu=0GiB}"
TRAIN_SAVE_TOTAL_LIMIT="${TRAIN_SAVE_TOTAL_LIMIT:-2}"
TRAIN_GPU_MIN_FREE_MIB="${TRAIN_GPU_MIN_FREE_MIB:-40000}"
LOCAL_MODEL="${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

DATA_DIR="$OUTPUT_ROOT/data"
TRAIN_DIR="$OUTPUT_ROOT/training/gold_only_lora"
EVAL_DIR="$OUTPUT_ROOT/evaluation"
MARKER_DIR="$OUTPUT_ROOT/stage_markers"
LOG_DIR="$OUTPUT_ROOT/logs"
STATUS_FILE="$OUTPUT_ROOT/pipeline_status.json"
mkdir -p "$DATA_DIR" "$EVAL_DIR" "$MARKER_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

STAGES=(prepare_data train generate_responses oracle_eval statistics report)
stage_index() {
  local wanted="$1" index
  for index in "${!STAGES[@]}"; do [[ "${STAGES[$index]}" == "$wanted" ]] && { echo "$index"; return; }; done
  echo "未知のstage: $wanted" >&2; exit 20
}
START_INDEX="$(stage_index "$START_STAGE")"
END_INDEX="$(stage_index "$END_STAGE")"
(( START_INDEX <= END_INDEX )) || { echo "START_STAGEはEND_STAGE以前にしてください。" >&2; exit 20; }

BASE_FINGERPRINT="$(python3 - "$CONFIG" "$DATASET" "$DRY_RUN" "$LOCAL_MODEL" "$SEED" "$TRAIN_DEVICE_MAP" "$TRAIN_MAX_MEMORY" <<'PY'
import hashlib,json,pathlib,sys,yaml
config_path=pathlib.Path(sys.argv[1]); dataset=sys.argv[2]
cfg=yaml.safe_load(config_path.read_text(encoding='utf-8')); ds=cfg['datasets'][dataset]
paths=[config_path,pathlib.Path(ds['gold_source']),pathlib.Path(ds['basis_train_source']),pathlib.Path(ds['evaluation_source'])]
if ds.get('oracle_template_source'): paths.append(pathlib.Path(ds['oracle_template_source']))
for item in ds['oracle_categories'].values():
 for key in ('existing_raw','existing_ood_raw'):
  if item.get(key): paths.append(pathlib.Path(item[key]))
paths += [
 pathlib.Path('tools/gold_only_dpo.py'),pathlib.Path('tools/gold_only_dpo_report.py'),
 pathlib.Path('tools/run_oracle_evaluation_lora_pair.py'),
 pathlib.Path('scripts/run_gold_only_four_model_statistics.py'),
 pathlib.Path('scripts/run_gold_only_dpo_dataset_pipeline.sh'),
 pathlib.Path('scripts/eval_oracle_tst.py'),
 pathlib.Path('scripts/eval_oracle_conversation_style_esconv_v2.py'),
 pathlib.Path('scripts/eval_oracle_strategy_transition_esconv_v2.py'),
 pathlib.Path('scripts/eval_oracle_mathdial_v2.py'),pathlib.Path('scripts/eval_oracle_meditod.py'),
]
h=hashlib.sha256()
for path in paths:
 if not path.is_file(): raise SystemExit(f'必須入力がありません: {path}')
 h.update(path.as_posix().encode()); h.update(hashlib.sha256(path.read_bytes()).digest())
h.update(json.dumps({'dataset':dataset,'dry_run':sys.argv[3],'local_model':sys.argv[4],'seed':sys.argv[5],'device_map':sys.argv[6],'max_memory':sys.argv[7]},sort_keys=True).encode())
print(h.hexdigest())
PY
)"

RUN_MANIFEST="$OUTPUT_ROOT/run_manifest.json"
python3 - "$RUN_MANIFEST" "$BASE_FINGERPRINT" "$DATASET" "$RUN_TAG" "$CONFIG" <<'PY'
import datetime,json,pathlib,sys
path=pathlib.Path(sys.argv[1]); fp=sys.argv[2]
if path.exists():
 old=json.loads(path.read_text(encoding='utf-8'))
 if old.get('fingerprint') != fp: raise SystemExit('同じRUN_TAGの入力・config・コードhashが変わっています。新しいRUN_TAGを使用してください。')
else:
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps({'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fingerprint':fp,'dataset':sys.argv[3],'run_tag':sys.argv[4],'config':sys.argv[5]},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY

write_status() {
  python3 - "$STATUS_FILE" "$DATASET" "$RUN_TAG" "$1" "$2" "$3" <<'PY'
import datetime,json,pathlib,sys
path=pathlib.Path(sys.argv[1]); path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps({'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':sys.argv[2],'run_tag':sys.argv[3],'state':sys.argv[4],'stage':sys.argv[5],'message':sys.argv[6]},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY
}

marker_valid() {
  python3 - "$MARKER_DIR/$1.SUCCESS.json" "$BASE_FINGERPRINT" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1])
try: x=json.loads(p.read_text(encoding='utf-8'))
except Exception: raise SystemExit(1)
if x.get('fingerprint') != sys.argv[2]: raise SystemExit(1)
for name,expected in x.get('outputs',{}).items():
 q=pathlib.Path(name)
 if not q.is_file() or hashlib.sha256(q.read_bytes()).hexdigest()!=expected: raise SystemExit(1)
PY
}

write_marker() {
  local stage="$1"; shift
  python3 - "$MARKER_DIR/$stage.SUCCESS.json" "$BASE_FINGERPRINT" "$stage" "$@" <<'PY'
import datetime,hashlib,json,pathlib,sys
path=pathlib.Path(sys.argv[1]); outputs={}
for name in sys.argv[4:]:
 p=pathlib.Path(name)
 if not p.is_file(): raise SystemExit(f'stage出力がありません: {p}')
 outputs[p.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
path.write_text(json.dumps({'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fingerprint':sys.argv[2],'stage':sys.argv[3],'outputs':outputs},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY
}

run_stage() {
  local name="$1" function="$2"; shift 2
  local index="$(stage_index "$name")"
  (( index < START_INDEX || index > END_INDEX )) && return
  if marker_valid "$name" 2>/dev/null; then echo "[SKIP] $name (verified SUCCESS marker)"; return; fi
  echo "[START] $name"; write_status running "$name" "stage started"
  "$function"
  write_marker "$name" "$@"
  echo "[DONE] $name"
}

preflight() {
  local free_kib
  free_kib="$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {print $4}')"
  (( free_kib >= 8*1024*1024 )) || { echo "ディスク空きが8GiB未満です。" >&2; exit 20; }
  if [[ "$DRY_RUN" != "1" ]] && ! command -v nvidia-smi >/dev/null; then
    echo "本学習に必要なnvidia-smiがありません。" >&2; exit 20
  fi
  echo "[preflight] free_disk_gib=$((free_kib/1024/1024)) dataset=$DATASET"
}

gpu_preflight() {
  local purpose="$1" devices="$2"
  [[ "$DRY_RUN" == "1" ]] && return
  local report
  report="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)" || return 1
  python3 - "$devices" "$TRAIN_GPU_MIN_FREE_MIB" "$purpose" "$report" <<'PY'
import sys
wanted={int(value) for value in sys.argv[1].split(',') if value.strip()}
minimum=int(sys.argv[2]); found={}
for line in sys.argv[4].splitlines():
 index,free=(part.strip() for part in line.split(',',1)); found[int(index)]=int(free)
missing=sorted(wanted-set(found)); low={idx:found[idx] for idx in wanted&set(found) if found[idx]<minimum}
if missing or low:
 raise SystemExit(f"{sys.argv[3]} GPU preflight失敗: missing={missing} low_free_mib={low} required={minimum}")
print(f"[gpu_preflight] purpose={sys.argv[3]} free_mib={{{', '.join(f'{idx}: {found[idx]}' for idx in sorted(wanted))}}}")
PY
}

prepare_data_stage() {
  python3 -m tools.gold_only_dpo --config "$CONFIG" prepare --dataset "$DATASET" \
    --output "$DATA_DIR/gold_only_train.jsonl" --manifest "$DATA_DIR/gold_only_manifest.json"
}

train_stage() {
  local args=(--dataset "$DATA_DIR/gold_only_train.jsonl" --model-id "$LOCAL_MODEL" --output-dir "$TRAIN_DIR" --num-train-epochs 1 --learning-rate 5e-6 --beta 0.1 --max-length 1024 --per-device-train-batch-size 1 --gradient-accumulation-steps 8 --eval-ratio 0 --seed "$SEED" --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --no-4bit --device-map "$TRAIN_DEVICE_MAP" --max-memory "$TRAIN_MAX_MEMORY" --save-steps 25 --save-total-limit "$TRAIN_SAVE_TOTAL_LIMIT" --warmup-ratio 0.03 --resume-from-checkpoint auto)
  python3 -m tools.train_qwen35_dpo_lora "${args[@]}" --dry-run
  if [[ "$DRY_RUN" == "1" ]]; then
    mkdir -p "$TRAIN_DIR"
    printf '{}\n' > "$TRAIN_DIR/adapter_config.json"
    printf 'dry-run\n' > "$TRAIN_DIR/adapter_model.safetensors"
    return
  fi
  gpu_preflight training "$TRAIN_CUDA_VISIBLE_DEVICES"
  local attempt status
  for attempt in 1 2; do
    set +e
    env CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" python3 -m tools.train_qwen35_dpo_lora "${args[@]}"
    status=$?; set -e
    (( status == 0 )) && break
    (( attempt == 1 )) && { echo "[train] 同一条件・最新checkpointで再試行します。"; sleep 15; }
  done
  (( status == 0 )) || exit 20
  [[ -f "$TRAIN_DIR/adapter_config.json" ]] || { echo "LoRA adapterが完成していません。" >&2; exit 20; }
  [[ -f "$TRAIN_DIR/adapter_model.safetensors" || -f "$TRAIN_DIR/adapter_model.bin" ]] || {
    echo "LoRA weightが完成していません。" >&2; exit 20;
  }
}

generate_stage() {
  local mock=(); [[ "$DRY_RUN" == "1" ]] && mock+=(--mock)
  gpu_preflight evaluation_generation "$EVAL_CUDA_VISIBLE_DEVICES"
  env CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" DPO_COMPARE_MAX_MEMORY="$TRAIN_MAX_MEMORY" \
    python3 -m tools.gold_only_dpo --config "$CONFIG" generate --dataset "$DATASET" \
      --lora-path "$TRAIN_DIR" --output "$EVAL_DIR/gold_only_responses.jsonl" \
      --manifest "$EVAL_DIR/generation_manifest.json" \
      --base-model "$LOCAL_MODEL" --seed "$SEED" "${mock[@]}"
  local ood=(); [[ "$DATASET" == "meditod" ]] && ood+=(--ood-output "$EVAL_DIR/oracle_input_gold_ood.jsonl")
  python3 -m tools.gold_only_dpo --config "$CONFIG" build-oracle-input --dataset "$DATASET" \
    --responses "$EVAL_DIR/gold_only_responses.jsonl" --output "$EVAL_DIR/oracle_input_gold.jsonl" "${ood[@]}"
}

run_oracle_category() {
  local module="$1" input="$2" output="$3" category="${4:-}" dry=()
  [[ "$DRY_RUN" == "1" ]] && dry+=(--dry-run)
  local args=(--input "$input" --output_dir "$output" --judge_model "$JUDGE_MODEL" --score-scale 10 --oracle-workers "$WORKERS" --resume)
  [[ -n "$category" ]] && args+=(--category "$category")
  python3 -m "$module" "${args[@]}" "${dry[@]}"
}

merge_category() {
  local existing="$1" gold="$2" combined="$3" expected="$4"
  python3 -m tools.gold_only_dpo --config "$CONFIG" merge-raw --existing "$existing" --gold "$gold" \
    --output "$combined" --manifest "${combined%.jsonl}.manifest.json" --expected-samples "$expected"
}

oracle_stage() {
  JUDGE_MODEL="$(python3 -m tools.gold_only_dpo --config "$CONFIG" resolve-judge-model --dataset "$DATASET")"
  [[ -n "$JUDGE_MODEL" ]] || { echo "judge modelが解決できません。" >&2; exit 20; }
  case "$DATASET" in
    esconv)
      run_oracle_category scripts.eval_oracle_tst "$EVAL_DIR/oracle_input_gold.jsonl" "$EVAL_DIR/oracle_gold/main/text_style_transfer"
      run_oracle_category scripts.eval_oracle_conversation_style_esconv_v2 "$EVAL_DIR/oracle_input_gold.jsonl" "$EVAL_DIR/oracle_gold/main/conversation_style"
      run_oracle_category scripts.eval_oracle_strategy_transition_esconv_v2 "$EVAL_DIR/oracle_input_gold.jsonl" "$EVAL_DIR/oracle_gold/main/strategy_transition"
      merge_category "artifacts/evaluations/oracle_eval_runs/esconv_topconf_three_model_gpt54_100_10pt_topconf_three_model_10pt/oracle_tst_10pt/raw.jsonl" "$EVAL_DIR/oracle_gold/main/text_style_transfer/raw.jsonl" "$EVAL_DIR/oracle_combined/main/text_style_transfer/raw.jsonl" 100
      merge_category "artifacts/evaluations/oracle_eval_runs/esconv_topconf_three_model_esconv_v2_100_gpt54_v1_topconf_three_model_esconv_v2_10pt/oracle_conversation_style_esconv_v2_10pt/raw.jsonl" "$EVAL_DIR/oracle_gold/main/conversation_style/raw.jsonl" "$EVAL_DIR/oracle_combined/main/conversation_style/raw.jsonl" 100
      merge_category "artifacts/evaluations/oracle_eval_runs/esconv_topconf_three_model_esconv_v2_100_gpt54_v1_topconf_three_model_esconv_v2_10pt/oracle_strategy_transition_esconv_v2_10pt/raw.jsonl" "$EVAL_DIR/oracle_gold/main/strategy_transition/raw.jsonl" "$EVAL_DIR/oracle_combined/main/strategy_transition/raw.jsonl" 100
      ;;
    mathdial)
      run_oracle_category scripts.eval_oracle_mathdial_v2 "$EVAL_DIR/oracle_input_gold.jsonl" "$EVAL_DIR/oracle_gold/main/pedagogical_v2" pedagogical
      run_oracle_category scripts.eval_oracle_mathdial_v2 "$EVAL_DIR/oracle_input_gold.jsonl" "$EVAL_DIR/oracle_gold/main/general" general
      merge_category "artifacts/mathdial_wildchat/evaluation_rechecks/mathdial_v6_instruction_outcome_selected_top100_v1/evaluation/oracle/pedagogical_v2/raw.jsonl" "$EVAL_DIR/oracle_gold/main/pedagogical_v2/raw.jsonl" "$EVAL_DIR/oracle_combined/main/pedagogical_v2/raw.jsonl" 100
      merge_category "artifacts/mathdial_wildchat/evaluation_rechecks/mathdial_v6_instruction_outcome_selected_top100_v1/evaluation/oracle/general/raw.jsonl" "$EVAL_DIR/oracle_gold/main/general/raw.jsonl" "$EVAL_DIR/oracle_combined/main/general/raw.jsonl" 100
      ;;
    meditod)
      local category
      for category in history general safety; do
        run_oracle_category scripts.eval_oracle_meditod "$EVAL_DIR/oracle_input_gold.jsonl" "$EVAL_DIR/oracle_gold/main/$category" "$category"
        run_oracle_category scripts.eval_oracle_meditod "$EVAL_DIR/oracle_input_gold_ood.jsonl" "$EVAL_DIR/oracle_gold/ood/$category" "$category"
        merge_category "artifacts/meditod_wildchat/runs/meditod_wildchat_gpt56_v2/evaluation/oracle/$category/raw.jsonl" "$EVAL_DIR/oracle_gold/main/$category/raw.jsonl" "$EVAL_DIR/oracle_combined/main/$category/raw.jsonl" 100
        merge_category "artifacts/meditod_wildchat/runs/meditod_wildchat_gpt56_v2/evaluation/oracle/ood_$category/raw.jsonl" "$EVAL_DIR/oracle_gold/ood/$category/raw.jsonl" "$EVAL_DIR/oracle_combined/ood/$category/raw.jsonl" 30
      done
      ;;
  esac
}

statistics_stage() {
  local permutations=10000 bootstrap=2000; [[ "$DRY_RUN" == "1" ]] && { permutations=100; bootstrap=100; }
  local common=(--permutations "$permutations" --bootstrap "$bootstrap" --seed "$SEED")
  case "$DATASET" in
    esconv) python3 -m scripts.run_gold_only_four_model_statistics --raw "text_style_transfer=$EVAL_DIR/oracle_combined/main/text_style_transfer/raw.jsonl" --raw "conversation_style=$EVAL_DIR/oracle_combined/main/conversation_style/raw.jsonl" --raw "strategy_transition=$EVAL_DIR/oracle_combined/main/strategy_transition/raw.jsonl" --output-dir "$OUTPUT_ROOT/statistics" "${common[@]}" ;;
    mathdial) python3 -m scripts.run_gold_only_four_model_statistics --raw "pedagogical_v2=$EVAL_DIR/oracle_combined/main/pedagogical_v2/raw.jsonl" --raw "general=$EVAL_DIR/oracle_combined/main/general/raw.jsonl" --output-dir "$OUTPUT_ROOT/statistics" --inference-status exploratory_outcome_selected_success_case_analysis "${common[@]}" ;;
    meditod)
      python3 -m scripts.run_gold_only_four_model_statistics --raw "history=$EVAL_DIR/oracle_combined/main/history/raw.jsonl" --raw "general=$EVAL_DIR/oracle_combined/main/general/raw.jsonl" --raw "safety=$EVAL_DIR/oracle_combined/main/safety/raw.jsonl" --output-dir "$OUTPUT_ROOT/statistics" --cluster-map "$EVAL_DIR/oracle_input_gold.jsonl" "${common[@]}"
      python3 -m scripts.run_gold_only_four_model_statistics --raw "history=$EVAL_DIR/oracle_combined/ood/history/raw.jsonl" --raw "general=$EVAL_DIR/oracle_combined/ood/general/raw.jsonl" --raw "safety=$EVAL_DIR/oracle_combined/ood/safety/raw.jsonl" --output-dir "$OUTPUT_ROOT/statistics_ood" --cluster-map "$EVAL_DIR/oracle_input_gold_ood.jsonl" --inference-status secondary_ood "${common[@]}"
      ;;
  esac
}

report_stage() { python3 -m tools.gold_only_dpo_report --dataset "$DATASET" --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/reports/final_report.md"; }

preflight
run_stage prepare_data prepare_data_stage "$DATA_DIR/gold_only_train.jsonl" "$DATA_DIR/gold_only_manifest.json"
run_stage train train_stage "$TRAIN_DIR/adapter_config.json" "$TRAIN_DIR/adapter_model.safetensors"
GENERATION_OUTPUTS=("$EVAL_DIR/gold_only_responses.jsonl" "$EVAL_DIR/generation_manifest.json" "$EVAL_DIR/oracle_input_gold.jsonl")
[[ "$DATASET" == "meditod" ]] && GENERATION_OUTPUTS+=("$EVAL_DIR/oracle_input_gold_ood.jsonl")
run_stage generate_responses generate_stage "${GENERATION_OUTPUTS[@]}"
case "$DATASET" in
  esconv) ORACLE_OUTPUTS=("$EVAL_DIR/oracle_combined/main/text_style_transfer/raw.jsonl" "$EVAL_DIR/oracle_combined/main/conversation_style/raw.jsonl" "$EVAL_DIR/oracle_combined/main/strategy_transition/raw.jsonl") ;;
  mathdial) ORACLE_OUTPUTS=("$EVAL_DIR/oracle_combined/main/pedagogical_v2/raw.jsonl" "$EVAL_DIR/oracle_combined/main/general/raw.jsonl") ;;
  meditod) ORACLE_OUTPUTS=("$EVAL_DIR/oracle_combined/main/history/raw.jsonl" "$EVAL_DIR/oracle_combined/main/general/raw.jsonl" "$EVAL_DIR/oracle_combined/main/safety/raw.jsonl" "$EVAL_DIR/oracle_combined/ood/history/raw.jsonl" "$EVAL_DIR/oracle_combined/ood/general/raw.jsonl" "$EVAL_DIR/oracle_combined/ood/safety/raw.jsonl") ;;
esac
run_stage oracle_eval oracle_stage "${ORACLE_OUTPUTS[@]}"
STATISTICS_OUTPUTS=("$OUTPUT_ROOT/statistics/model_summary.csv" "$OUTPUT_ROOT/statistics/omnibus_friedman.csv" "$OUTPUT_ROOT/statistics/posthoc_pairwise.csv")
if [[ "$DATASET" == "meditod" ]]; then
  STATISTICS_OUTPUTS+=("$OUTPUT_ROOT/statistics/cluster_model_summary.csv" "$OUTPUT_ROOT/statistics_ood/model_summary.csv" "$OUTPUT_ROOT/statistics_ood/cluster_model_summary.csv")
fi
run_stage statistics statistics_stage "${STATISTICS_OUTPUTS[@]}"
run_stage report report_stage "$OUTPUT_ROOT/reports/final_report.md"
write_status success completed "pipeline completed"
echo "Gold-only pipeline completed: $DATASET -> $OUTPUT_ROOT"
echo "Log: $LOG_FILE"
