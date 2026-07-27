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

RUN_TAG="${RUN_TAG:-meditod_wildchat_gpt56_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/meditod_wildchat/runs/${RUN_TAG}}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-4}"
START_STAGE="${START_STAGE:-preprocess}"
END_STAGE="${END_STAGE:-prepare_user_eval}"
STAGE="${STAGE:-}"
FORCE_STAGE="${FORCE_STAGE:-}"
MEDITOD_SOURCE_MODE="${MEDITOD_SOURCE_MODE:-public_raw}"
MEDITOD_CONFIG="${MEDITOD_CONFIG:-configs/datasets/meditod.yaml}"
MEDITOD_DIALOGS="${MEDITOD_DIALOGS:-}"
MEDITOD_ANNOTATIONS="${MEDITOD_ANNOTATIONS:-}"
MEDITOD_CANONICAL_DATA_DIR="${MEDITOD_CANONICAL_DATA_DIR:-}"
MEDITOD_DATA_TERMS_CONFIRMED="${MEDITOD_DATA_TERMS_CONFIRMED:-0}"
ANALYSIS_CONVERSATIONS="${MEDITOD_ANALYSIS_CONVERSATIONS:-24}"
ANALYSIS_MAX_INPUT_CHARS="${MEDITOD_ANALYSIS_MAX_INPUT_CHARS:-800000}"
ANALYSIS_MAX_OUTPUT_TOKENS="${MEDITOD_ANALYSIS_MAX_OUTPUT_TOKENS:-24000}"
ANALYSIS_MODEL="${MEDITOD_ANALYSIS_LLM_MODEL:-${AZURE_OPENAI_GPT56_SOL_DEPLOYMENT:-gpt-5.6-sol}}"
SCORING_MODEL="${MEDITOD_SCORING_LLM_MODEL:-${AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT:-gpt-5.6-terra}}"
GENERATION_MODEL="${MEDITOD_DPO_GENERATION_MODEL:-$SCORING_MODEL}"
JUDGE_MODEL="${MEDITOD_JUDGE_MODEL:-$SCORING_MODEL}"
LOCAL_MODEL="${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"
SCORING_PRESET="meditod_history_taking"
SCORING_PILOT_RECORDS="${SCORING_PILOT_RECORDS:-200}"
SCORING_BATCH_RECORDS="${SCORING_BATCH_RECORDS:-3000}"
SCORING_REQUESTS_PER_MINUTE="${SCORING_REQUESTS_PER_MINUTE:-120}"
SCORING_RATE_LIMIT_MAX_RETRIES="${SCORING_RATE_LIMIT_MAX_RETRIES:-6}"
SCORING_RATE_LIMIT_BACKOFF_SECONDS="${SCORING_RATE_LIMIT_BACKOFF_SECONDS:-15}"
DPO_INITIAL_SELECTION_POOL_COUNT="${DPO_INITIAL_SELECTION_POOL_COUNT:-3000}"
BASIS_SELECTED_COUNT="${MEDITOD_BASIS_SELECTED_COUNT:-3000}"
GOLD_DPO_COUNT="${MEDITOD_GOLD_COUNT:-500}"
RANDOM_DPO_COUNT="${MEDITOD_RANDOM_COUNT:-$((BASIS_SELECTED_COUNT + GOLD_DPO_COUNT))}"
DPO_RESCUE_MIN_CHOSEN="${MEDITOD_DPO_RESCUE_MIN_CHOSEN:-0.60}"
DPO_RESCUE_MAX_REJECTED="${MEDITOD_DPO_RESCUE_MAX_REJECTED:-0.65}"
DPO_RESCUE_MIN_GAP="${MEDITOD_DPO_RESCUE_MIN_GAP:-0.10}"
MEDITOD_RESUME_MIGRATION="${MEDITOD_RESUME_MIGRATION:-}"
DPO_MAX_SOURCE_CHARACTERS="${DPO_MAX_SOURCE_CHARACTERS:-16000}"
DPO_MAX_OUTPUT_TOKENS="${DPO_MAX_OUTPUT_TOKENS:-6144}"
WILDCHAT_FULL_SCAN="${WILDCHAT_FULL_SCAN:-1}"
WILDCHAT_CANDIDATE_TARGET_RECORDS="${WILDCHAT_CANDIDATE_TARGET_RECORDS:-25000}"
WILDCHAT_CHECKPOINT_EVERY="${WILDCHAT_CHECKPOINT_EVERY:-100000}"
EVAL_COUNT="${EVAL_COUNT:-100}"
OOD_EVAL_COUNT="${OOD_EVAL_COUNT:-30}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-$TRAIN_CUDA_VISIBLE_DEVICES}"
TRAIN_DEVICE_MAP="${TRAIN_DEVICE_MAP:-auto}"
TRAIN_MAX_MEMORY="${TRAIN_MAX_MEMORY:-0=46GiB,1=46GiB,cpu=0GiB}"
TRAIN_SAVE_TOTAL_LIMIT="${TRAIN_SAVE_TOTAL_LIMIT:-2}"
TRAIN_MIN_FREE_MEMORY_MIB="${TRAIN_MIN_FREE_MEMORY_MIB:-36000}"
PIPELINE_MIN_FREE_GB="${PIPELINE_MIN_FREE_GB:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAGES=(preprocess build_basis extract_wildchat score_wildchat select_data build_dpo train prepare_eval generate_responses oracle_eval statistics report prepare_user_eval)
if [[ -n "$STAGE" ]]; then START_STAGE="$STAGE"; END_STAGE="$STAGE"; fi

stage_index() {
  local target="$1" index
  for index in "${!STAGES[@]}"; do
    [[ "${STAGES[$index]}" == "$target" ]] && { echo "$index"; return; }
  done
  echo "未知のstageです: $target" >&2
  return 1
}

START_INDEX="$(stage_index "$START_STAGE")"
END_INDEX="$(stage_index "$END_STAGE")"
[[ "$START_INDEX" -le "$END_INDEX" ]] || { echo "START_STAGEはEND_STAGE以前にしてください。" >&2; exit 2; }

LOG_DIR="$OUTPUT_ROOT/logs"
STATE_DIR="$OUTPUT_ROOT/stage_state"
STATUS_FILE="$OUTPUT_ROOT/pipeline_status.json"
HEARTBEAT_FILE="${PIPELINE_HEARTBEAT_FILE:-$OUTPUT_ROOT/pipeline_heartbeat.json}"
mkdir -p "$LOG_DIR" "$STATE_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

FINGERPRINT="$(python3 - "$RUN_TAG" "$SEED" "$DRY_RUN" "$MEDITOD_SOURCE_MODE" "$MEDITOD_DATA_TERMS_CONFIRMED" "$ANALYSIS_MODEL" "$SCORING_MODEL" "$GENERATION_MODEL" "$JUDGE_MODEL" "$LOCAL_MODEL" "$ANALYSIS_CONVERSATIONS" "$ANALYSIS_MAX_INPUT_CHARS" "$ANALYSIS_MAX_OUTPUT_TOKENS" "$SCORING_PILOT_RECORDS" "$SCORING_BATCH_RECORDS" "$DPO_INITIAL_SELECTION_POOL_COUNT" "$BASIS_SELECTED_COUNT" "$GOLD_DPO_COUNT" "$RANDOM_DPO_COUNT" "$DPO_RESCUE_MIN_CHOSEN" "$DPO_RESCUE_MAX_REJECTED" "$DPO_RESCUE_MIN_GAP" "$DPO_MAX_SOURCE_CHARACTERS" "$DPO_MAX_OUTPUT_TOKENS" "$WILDCHAT_FULL_SCAN" "$WILDCHAT_CANDIDATE_TARGET_RECORDS" "$EVAL_COUNT" "$OOD_EVAL_COUNT" <<'PY'
import hashlib,json,pathlib,sys
paths=[
 "configs/datasets/meditod.yaml","configs/datasets/wildchat_health.yaml",
 "configs/evaluations/meditod_oracle_v1.yaml","configs/training/meditod_dpo.yaml",
 "configs/user_evaluations/meditod_likert_v1.yaml","core/dpo_prompting.py",
 "tools/meditod_dataset.py","tools/prepare_meditod.py","tools/prepare_meditod_for_analysis.py",
 "tools/analyze_meditod_corpus_transition_bayes.py","tools/wildchat_health.py",
 "tools/prioritize_health_candidates.py","tools/meditod_selection.py","tools/measure_basis_selection_pool.py",
 "tools/prepare_meditod_personal_pool.py","tools/prepare_meditod_broad_pool.py",
 "tools/promote_meditod_dpo_rescue.py","tools/meditod_available_data_decision.py",
 "core/transition_bayes_model.py","tools/score_dialogue_with_transition_bayes_model.py",
 "tools/translate_and_generate_dpo.py","tools/build_random_dailydialog_dpo.py",
 "tools/prepare_meditod_gold.py","tools/mix_meditod_dpo.py","tools/meditod_evaluation.py",
 "tools/meditod_annotation_metrics.py","tools/meditod_pipeline_support.py",
 "tools/prepare_three_model_likert_eval.py","core/three_model_likert_survey.py",
 "scripts/eval_oracle_meditod.py","scripts/run_meditod_statistics.py",
 "scripts/run_meditod_wildchat_pipeline.sh",
]
payload={"values":sys.argv[1:],"files":{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths}}
print(hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",", ":")).encode()).hexdigest())
PY
)"

if [[ "$MEDITOD_RESUME_MIGRATION" == "available1824_gold500_v4" \
  || "$MEDITOD_RESUME_MIGRATION" == "eval_fidelity_alias_reserve_v5" ]]; then
  [[ "$BASIS_SELECTED_COUNT" == "1824" ]] || {
    echo "今回のMediTOD互換移行ではMEDITOD_BASIS_SELECTED_COUNT=1824が必要です。" >&2
    exit 20
  }
  [[ "$GOLD_DPO_COUNT" == "500" ]] || {
    echo "今回のMediTOD互換移行ではMEDITOD_GOLD_COUNT=500が必要です。" >&2
    exit 20
  }
  [[ "$RANDOM_DPO_COUNT" == "2324" ]] || {
    echo "今回のMediTOD互換移行ではMEDITOD_RANDOM_COUNT=2324が必要です。" >&2
    exit 20
  }
  python3 -m tools.meditod_available_data_decision \
    --accepted "$OUTPUT_ROOT/dpo/basis_selected_ja.jsonl" \
    --candidates "$OUTPUT_ROOT/wildchat/general_health_consultation_candidates.jsonl" \
    --scored "$OUTPUT_ROOT/scoring/wildchat_scored_raw.jsonl" \
    --output "$OUTPUT_ROOT/dpo/available_data_training_decision.json" \
    --basis-count "$BASIS_SELECTED_COUNT" \
    --gold-count "$GOLD_DPO_COUNT" \
    --random-count "$RANDOM_DPO_COUNT"
fi

python3 - "$OUTPUT_ROOT/run_metadata.json" "$FINGERPRINT" "$RUN_TAG" "$SEED" "$DRY_RUN" "$ANALYSIS_MODEL" "$SCORING_MODEL" "$GENERATION_MODEL" "$JUDGE_MODEL" "$LOCAL_MODEL" "$MEDITOD_SOURCE_MODE" "$MEDITOD_RESUME_MIGRATION" <<'PY'
import datetime,json,pathlib,sys
path=pathlib.Path(sys.argv[1])
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
payload={"experiment_fingerprint":sys.argv[2],"run_tag":sys.argv[3],"seed":int(sys.argv[4]),"dry_run":sys.argv[5]=="1","models":{"analysis":sys.argv[6],"scoring":sys.argv[7],"generation":sys.argv[8],"judge":sys.argv[9],"local":sys.argv[10]},"dataset_mode":sys.argv[11],"created_at":now}
if path.exists():
 current=json.loads(path.read_text())
 if current.get("experiment_fingerprint") != sys.argv[2]:
  migration=sys.argv[12]
  allowed_migrations={
   "target3000_personal_health_fidelity_v2",
   "target3000_broad_health_fidelity_v3",
   "available1824_gold500_v4",
   "eval_fidelity_alias_reserve_v5",
  }
  if migration not in allowed_migrations:
   raise SystemExit("同じRUN_TAGの実験条件が変わっています。互換移行には対応するMEDITOD_RESUME_MIGRATIONを指定してください。")
  if current.get("run_tag")!=sys.argv[3] or current.get("seed")!=int(sys.argv[4]) or current.get("dataset_mode")!=sys.argv[11] or current.get("models")!=payload["models"]:
   raise SystemExit("再開移行で変更できないdataset/seed/model条件が一致しません。")
  history=list(current.get("migrations",[]))
  history.append({"name":migration,"migrated_at":now,"from_fingerprint":current.get("experiment_fingerprint"),"to_fingerprint":sys.argv[2]})
  payload["created_at"]=current.get("created_at",now)
  payload["migrations"]=history
  payload["previous_experiment_fingerprint"]=current.get("experiment_fingerprint")
  path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
  print(f"[migration] {migration}: run成果物を監査して互換再開します。")
path.parent.mkdir(parents=True,exist_ok=True)
if not path.exists(): path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
PY

if [[ "$MEDITOD_RESUME_MIGRATION" == "target3000_broad_health_fidelity_v3" ]]; then
  MIGRATION_MARKER="$STATE_DIR/broad_health_fidelity_v3_migration_applied"
  if [[ ! -f "$MIGRATION_MARKER" || "$(<"$MIGRATION_MARKER")" != "$FINGERPRINT" ]]; then
    rm -f \
      "$STATE_DIR/score_wildchat_SUCCESS.json" \
      "$STATE_DIR/select_data_SUCCESS.json" \
      "$STATE_DIR/build_dpo_SUCCESS.json" \
      "$STATE_DIR/train_SUCCESS.json" \
      "$STATE_DIR/prepare_eval_SUCCESS.json" \
      "$STATE_DIR/generate_responses_SUCCESS.json" \
      "$STATE_DIR/oracle_eval_SUCCESS.json" \
      "$STATE_DIR/statistics_SUCCESS.json" \
      "$STATE_DIR/report_SUCCESS.json" \
      "$STATE_DIR/prepare_user_eval_SUCCESS.json"
    printf '%s\n' "$FINGERPRINT" > "$MIGRATION_MARKER"
  fi
fi

if [[ "$MEDITOD_RESUME_MIGRATION" == "available1824_gold500_v4" ]]; then
  MIGRATION_MARKER="$STATE_DIR/available1824_gold500_v4_migration_applied"
  if [[ ! -f "$MIGRATION_MARKER" || "$(<"$MIGRATION_MARKER")" != "$FINGERPRINT" ]]; then
    rm -f \
      "$STATE_DIR/build_dpo_SUCCESS.json" \
      "$STATE_DIR/train_SUCCESS.json" \
      "$STATE_DIR/prepare_eval_SUCCESS.json" \
      "$STATE_DIR/generate_responses_SUCCESS.json" \
      "$STATE_DIR/oracle_eval_SUCCESS.json" \
      "$STATE_DIR/statistics_SUCCESS.json" \
      "$STATE_DIR/report_SUCCESS.json" \
      "$STATE_DIR/prepare_user_eval_SUCCESS.json"
    printf '%s\n' "$FINGERPRINT" > "$MIGRATION_MARKER"
  fi
fi

if [[ "$MEDITOD_RESUME_MIGRATION" == "eval_fidelity_alias_reserve_v5" ]]; then
  MIGRATION_MARKER="$STATE_DIR/eval_fidelity_alias_reserve_v5_migration_applied"
  if [[ ! -f "$MIGRATION_MARKER" || "$(<"$MIGRATION_MARKER")" != "$FINGERPRINT" ]]; then
    rm -f \
      "$STATE_DIR/prepare_eval_SUCCESS.json" \
      "$STATE_DIR/generate_responses_SUCCESS.json" \
      "$STATE_DIR/oracle_eval_SUCCESS.json" \
      "$STATE_DIR/statistics_SUCCESS.json" \
      "$STATE_DIR/report_SUCCESS.json" \
      "$STATE_DIR/prepare_user_eval_SUCCESS.json"
    printf '%s\n' "$FINGERPRINT" > "$MIGRATION_MARKER"
  fi
fi

write_status() {
  python3 - "$STATUS_FILE" "$HEARTBEAT_FILE" "$RUN_TAG" "$1" "$2" "$3" <<'PY'
import datetime,json,pathlib,sys
payload={"timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),"run_tag":sys.argv[3],"state":sys.argv[4],"stage":sys.argv[5],"message":sys.argv[6]}
for value in sys.argv[1:3]:
 path=pathlib.Path(value); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
PY
}

CURRENT_STAGE="startup"
PIPELINE_SUCCEEDED=0
on_exit() {
  local code=$?
  [[ "$PIPELINE_SUCCEEDED" == "1" ]] || write_status "$([[ "$code" -eq 20 ]] && echo fatal || echo incomplete)" "$CURRENT_STAGE" "pipeline exited status=$code" || true
}
trap on_exit EXIT

retry_command() {
  local attempt=1 delay=15
  while ! "$@"; do
    if [[ "$attempt" -ge 4 ]]; then
      echo "[RETRY] 4回失敗しました: $*" >&2
      return 1
    fi
    echo "[RETRY] attempt=$attempt; ${delay}s後に再試行: $*" >&2
    sleep "$delay"
    delay=$((delay * 2))
    attempt=$((attempt + 1))
  done
}

preflight_storage() {
  [[ "$DRY_RUN" == "1" ]] && return
  local available
  available="$(df -Pk "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')"
  (( available >= PIPELINE_MIN_FREE_GB * 1024 * 1024 )) || { echo "ディスク空きが${PIPELINE_MIN_FREE_GB}GiB未満です。" >&2; exit 20; }
}

gpu_preflight() {
  local devices="$1" label="$2"
  [[ "$DRY_RUN" == "1" ]] && return
  python3 - "$devices" "$TRAIN_MIN_FREE_MEMORY_MIB" "$label" <<'PY'
import subprocess,sys
required=int(sys.argv[2]); requested=[x.strip() for x in sys.argv[1].split(",") if x.strip()]
rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],text=True)
free={a.strip():int(b.strip()) for a,b in (line.split(",",1) for line in rows.splitlines() if line.strip())}
low={index:free.get(index,0) for index in requested if free.get(index,0)<required}
if low: raise SystemExit(f"{sys.argv[3]} GPU空きメモリ不足: {low}; required={required}MiB")
print(f"[preflight] {sys.argv[3]} GPU free={ {x:free[x] for x in requested} }")
PY
}

stage_outputs() {
  case "$1" in
    preprocess) printf '%s\n' "$MED_ROOT/data/meditod_conversations.jsonl" "$MED_ROOT/data/meditod_assistant_samples.jsonl" ;;
    build_basis) printf '%s\n' "$COMPAT_MODEL" "$OUTPUT_ROOT/basis_model/meditod_model_quality.json" ;;
    extract_wildchat) printf '%s\n' "$WILD_DIR/general_health_consultation_candidates.jsonl" "$WILD_DIR/manifest.json" ;;
    score_wildchat) printf '%s\n' "$SCORED" "$OUTPUT_ROOT/scoring/selection_pool_progress.json" "$WILD_DIR/reuse_manifest.json" ;;
    select_data) printf '%s\n' "$SELECT_DIR/basis_top.jsonl" "$SELECT_DIR/domain_random.jsonl" "$SELECT_DIR/topic_similarity_top.jsonl" "$SELECT_DIR/selection_report.json" ;;
    build_dpo)
      printf '%s\n' "$DPO_DIR/meditod_basis_train.jsonl" "$DPO_DIR/meditod_random_train.jsonl" "$DPO_DIR/broad_resume_manifest.json"
      [[ "$MEDITOD_RESUME_MIGRATION" == "available1824_gold500_v4" ]] && printf '%s\n' "$DPO_DIR/available_data_training_decision.json"
      ;;
    train) printf '%s\n' "$TRAIN_DIR/basis_lora/adapter_config.json" "$TRAIN_DIR/random_lora/adapter_config.json" ;;
    prepare_eval)
      printf '%s\n' "$EVAL_DIR/prompts_ja.jsonl" "$EVAL_DIR/prompts_all_ja.jsonl" "$EVAL_DIR/prompt_selection_manifest.json"
      (( OOD_EVAL_COUNT > 0 )) && printf '%s\n' "$EVAL_DIR/ood_prompts_ja.jsonl" "$EVAL_DIR/ood_prompt_selection_manifest.json"
      ;;
    generate_responses) printf '%s\n' "$EVAL_DIR/responses.jsonl" "$EVAL_DIR/oracle_input.jsonl" ;;
    oracle_eval)
      printf '%s\n' "$EVAL_DIR/oracle/history/raw.jsonl" "$EVAL_DIR/oracle/general/raw.jsonl" "$EVAL_DIR/oracle/safety/raw.jsonl"
      (( OOD_EVAL_COUNT > 0 )) && printf '%s\n' "$EVAL_DIR/oracle/ood_history/raw.jsonl" "$EVAL_DIR/oracle/ood_general/raw.jsonl" "$EVAL_DIR/oracle/ood_safety/raw.jsonl"
      ;;
    statistics)
      printf '%s\n' "$EVAL_DIR/statistics/model_summary.csv" "$EVAL_DIR/statistics/cluster_omnibus_friedman.csv"
      (( OOD_EVAL_COUNT > 0 )) && printf '%s\n' "$EVAL_DIR/statistics_ood/model_summary.csv" "$EVAL_DIR/statistics_ood/cluster_omnibus_friedman.csv"
      ;;
    report) printf '%s\n' "$OUTPUT_ROOT/reports/final_report.md" ;;
    prepare_user_eval) printf '%s\n' "$OUTPUT_ROOT/user_eval/manifest.json" ;;
  esac
}

write_stage_marker() {
  local name="$1" marker
  local -a outputs
  marker="$STATE_DIR/${name}_SUCCESS.json"
  mapfile -t outputs < <(stage_outputs "$name")
  python3 - "$marker" "$FINGERPRINT" "$name" "${outputs[@]}" <<'PY'
import datetime,hashlib,json,pathlib,sys

def file_hash(path):
 with path.open("rb") as source:
  digest=hashlib.sha256()
  for chunk in iter(lambda: source.read(1024 * 1024), b""):
   digest.update(chunk)
 return digest.hexdigest()

missing=[name for name in sys.argv[4:] if not pathlib.Path(name).is_file()]
if missing: raise SystemExit(f"stage完了成果物が不足しています: {missing}")
payload={"stage":sys.argv[3],"experiment_fingerprint":sys.argv[2],"completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"output_hashes":{name:file_hash(pathlib.Path(name)) for name in sys.argv[4:]}}
path=pathlib.Path(sys.argv[1]); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
PY
}

run_stage() {
  local name="$1" index marker
  shift
  index="$(stage_index "$name")"
  [[ "$index" -ge "$START_INDEX" && "$index" -le "$END_INDEX" ]] || return 0
  marker="$STATE_DIR/${name}_SUCCESS.json"
  if [[ -f "$marker" && "$FORCE_STAGE" != "$name" && "$FORCE_STAGE" != all ]]; then
    python3 - "$marker" "$FINGERPRINT" <<'PY'
import hashlib,json,pathlib,sys

def file_hash(path):
 with path.open("rb") as source:
  digest=hashlib.sha256()
  for chunk in iter(lambda: source.read(1024 * 1024), b""):
   digest.update(chunk)
 return digest.hexdigest()

payload=json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload.get("experiment_fingerprint") != sys.argv[2]: raise SystemExit("stage marker fingerprint不一致")
for name,digest in payload.get("output_hashes",{}).items():
 path=pathlib.Path(name)
 if not path.exists() or file_hash(path)!=digest: raise SystemExit(f"stage成果物が変更または欠落しています: {name}")
PY
    echo "[SKIP] $name completed"
    return 0
  fi
  CURRENT_STAGE="$name"
  write_status running "$name" "stage started"
  echo "[START] $name"
  preflight_storage
  "$@"
  write_stage_marker "$name"
  echo "[DONE] $name"
}

MED_ROOT="$OUTPUT_ROOT/meditod"
MED_CONV="$MED_ROOT/data/meditod_conversations.jsonl"
MED_SAMPLES="$MED_ROOT/data/meditod_assistant_samples.jsonl"
ANALYSIS_CORPUS="$OUTPUT_ROOT/basis_model/meditod_analysis_corpus.jsonl"
ANALYSIS_AGGREGATES="$OUTPUT_ROOT/basis_model/meditod_train_aggregates.json"
COMPAT_MODEL="$OUTPUT_ROOT/basis_model/meditod_transition_compat.json"
WILD_DIR="$OUTPUT_ROOT/wildchat"
HEALTH_CONVERSATIONS="$WILD_DIR/general_health_consultation_conversations.jsonl"
HEALTH_CANDIDATES="$WILD_DIR/general_health_consultation_candidates.jsonl"
PRIORITIZED_CANDIDATES="$OUTPUT_ROOT/scoring/prioritized_candidates.jsonl"
SCORED_RAW="$OUTPUT_ROOT/scoring/wildchat_scored_raw.jsonl"
SCORED="$OUTPUT_ROOT/scoring/wildchat_scored.jsonl"
SELECT_DIR="$OUTPUT_ROOT/selections"
DPO_DIR="$OUTPUT_ROOT/dpo"
TRAIN_DIR="$OUTPUT_ROOT/training"
EVAL_DIR="$OUTPUT_ROOT/evaluation"

preprocess_stage() {
  local args=(--config "$MEDITOD_CONFIG" --output-root "$MED_ROOT" --source-mode "$MEDITOD_SOURCE_MODE" --seed "$SEED")
  if [[ "$DRY_RUN" == "1" || "$MEDITOD_DATA_TERMS_CONFIRMED" == "1" ]]; then
    args+=(--data-terms-confirmed)
  else
    echo "MediTODの公式利用条件を確認し、MEDITOD_DATA_TERMS_CONFIRMED=1を指定してください。" >&2
    return 20
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    args=(--config tests/fixtures/meditod_public_raw_config.yaml --output-root "$MED_ROOT" --source-mode public_raw --seed "$SEED" --dialogs tests/fixtures/meditod_dialogs.json --annotations tests/fixtures/meditod_annotations.json --data-terms-confirmed)
  elif [[ "$MEDITOD_SOURCE_MODE" == canonical_full ]]; then
    args+=(--canonical-data-dir "$MEDITOD_CANONICAL_DATA_DIR")
  elif [[ -n "$MEDITOD_DIALOGS" || -n "$MEDITOD_ANNOTATIONS" ]]; then
    [[ -n "$MEDITOD_DIALOGS" && -n "$MEDITOD_ANNOTATIONS" ]] || { echo "MEDITOD_DIALOGS/ANNOTATIONSは両方必要です。" >&2; return 20; }
    args+=(--dialogs "$MEDITOD_DIALOGS" --annotations "$MEDITOD_ANNOTATIONS")
  fi
  python3 -m tools.prepare_meditod "${args[@]}"
}

build_basis_stage() {
  local count="$ANALYSIS_CONVERSATIONS" mock=()
  [[ "$DRY_RUN" == "1" ]] && { count=2; mock+=(--mock); }
  python3 -m tools.prepare_meditod_for_analysis --input "$MED_CONV" --output "$ANALYSIS_CORPUS" --aggregate-output "$ANALYSIS_AGGREGATES" --manifest "$OUTPUT_ROOT/basis_model/meditod_analysis_corpus.manifest.json" --count "$count" --seed "$SEED"
  retry_command python3 -m tools.analyze_meditod_corpus_transition_bayes --input "$ANALYSIS_CORPUS" --aggregates "$ANALYSIS_AGGREGATES" --output "$OUTPUT_ROOT/basis_model/meditod_transition_bayes_model.json" --compat-output "$COMPAT_MODEL" --manifest "$OUTPUT_ROOT/basis_model/meditod_transition_bayes_model.manifest.json" --prompt-output "$OUTPUT_ROOT/basis_model/meditod_analysis_prompt.txt" --input-text-output "$OUTPUT_ROOT/basis_model/meditod_analysis_input.txt" --quality-report-output "$OUTPUT_ROOT/basis_model/meditod_model_quality.json" --rejected-models-output "$OUTPUT_ROOT/basis_model/rejected_models.jsonl" --model "$ANALYSIS_MODEL" --max-input-chars "$ANALYSIS_MAX_INPUT_CHARS" --max-output-tokens "$ANALYSIS_MAX_OUTPUT_TOKENS" --emission-margin 0.10 --min-negative-observations 2 "${mock[@]}"
}

extract_wildchat_stage() {
  local args=(--config configs/datasets/wildchat_health.yaml --output-dir "$WILD_DIR" --seed "$SEED" --checkpoint-every "$WILDCHAT_CHECKPOINT_EVERY" --heartbeat-file "$HEARTBEAT_FILE")
  [[ "$WILDCHAT_FULL_SCAN" == "1" ]] || args+=(--target-candidate-records "$WILDCHAT_CANDIDATE_TARGET_RECORDS")
  if [[ "$DRY_RUN" == "1" ]]; then args+=(--fixture tests/fixtures/wildchat_health.jsonl); else [[ -n "$LIMIT" ]] && args+=(--limit "$LIMIT"); fi
  python3 -m tools.wildchat_health "${args[@]}"
}

reconcile_scoring() {
  if [[ ! -f "$SCORED" ]]; then
    python3 -m tools.meditod_pipeline_support enrich-score --input "$SCORED_RAW" --output "$SCORED"
    return
  fi
  local raw enriched
  raw="$(wc -l < "$SCORED_RAW")"; enriched="$(wc -l < "$SCORED")"
  if (( enriched < raw )); then
    python3 -m tools.meditod_pipeline_support enrich-score --input "$SCORED_RAW" --output "$SCORED" --skip-records "$enriched" --append
  elif (( enriched > raw )); then
    echo "enriched scoringがrawを超えています。" >&2; return 20
  fi
}

prepare_broad_pool() {
  local audit="${1:-0}"
  local args=(
    --config configs/datasets/wildchat_health.yaml
    --conversations "$HEALTH_CONVERSATIONS"
    --candidates "$HEALTH_CANDIDATES"
    --manifest "$WILD_DIR/manifest.json"
    --statistics "$WILD_DIR/statistics.json"
    --reuse-manifest "$WILD_DIR/reuse_manifest.json"
    --diagnostic-report "$WILD_DIR/broad_health_diagnostic_report.json"
    --seed "$SEED"
  )
  if [[ "$audit" == "1" ]]; then
    args=(
      --config configs/datasets/wildchat_health.yaml
      --conversations "$HEALTH_CONVERSATIONS"
      --candidates "$HEALTH_CANDIDATES"
      --manifest "$WILD_DIR/manifest.json"
      --statistics "$WILD_DIR/statistics.json"
      --reuse-manifest "$DPO_DIR/broad_resume_manifest.json"
      --diagnostic-report "$DPO_DIR/broad_resume_diagnostic.json"
      --seed "$SEED"
    )
    args+=(
      --accepted "$DPO_DIR/basis_selected_ja.jsonl"
      --skipped "$DPO_DIR/basis_selected_ja_skipped.jsonl"
      --quarantine-dir "$DPO_DIR/quarantine"
      --bayes-model "$COMPAT_MODEL"
      --generation-model "$GENERATION_MODEL"
      --scoring-model "$SCORING_MODEL"
      --rejected-candidates 4
      --min-score-gap 0.20
      --min-chosen-posterior 0.70
      --max-rejected-posterior 0.55
    )
  fi
  python3 -m tools.prepare_meditod_broad_pool "${args[@]}"
  if [[ "$audit" == "1" \
    && "$MEDITOD_RESUME_MIGRATION" == "available1824_gold500_v4" \
    && -s "$PRIORITIZED_CANDIDATES" ]]; then
    echo "[resume] 全候補処理済みのため既存prioritized candidatesを再利用します。"
    return
  fi
  python3 -m tools.prioritize_health_candidates \
    --input "$HEALTH_CANDIDATES" \
    --output "$PRIORITIZED_CANDIDATES" \
    --report "$OUTPUT_ROOT/scoring/candidate_priority_report.json" \
    --seed "$SEED"
}

measure_selection_pool() {
  local required="$1"
  python3 -m tools.measure_basis_selection_pool \
    --input "$SCORED" \
    --bayes-model "$COMPAT_MODEL" \
    --output "$OUTPUT_ROOT/scoring/selection_pool_progress.json" \
    --history "$OUTPUT_ROOT/scoring/selection_pool_history.jsonl" \
    --method state_specific_margin \
    --margin 0.05 \
    --required "$required" \
    --exclude-fallback-conversations \
    --exclude-explicit-unsafe-medical-advice \
    --max-source-characters "$DPO_MAX_SOURCE_CHARACTERS"
}

score_next_batch() {
  local before after
  before="$(wc -l < "$SCORED_RAW")"
  retry_command python3 -m tools.score_dialogue_with_transition_bayes_model \
    --input "$PRIORITIZED_CANDIDATES" \
    --bayes-model "$COMPAT_MODEL" \
    --output "$SCORED_RAW" \
    --model "$SCORING_MODEL" \
    --workers "$WORKERS" \
    --max-new-records "$SCORING_BATCH_RECORDS" \
    --scoring-preset "$SCORING_PRESET" \
    --invalid-observation-retries 2 \
    --requests-per-minute "$SCORING_REQUESTS_PER_MINUTE" \
    --rate-limit-max-retries "$SCORING_RATE_LIMIT_MAX_RETRIES" \
    --rate-limit-initial-backoff-seconds "$SCORING_RATE_LIMIT_BACKOFF_SECONDS" \
    --fallback-on-errors
  after="$(wc -l < "$SCORED_RAW")"
  (( after > before )) || return 21
  reconcile_scoring
}

score_wildchat_stage() {
  mkdir -p "$OUTPUT_ROOT/scoring"
  local pilot="$SCORING_PILOT_RECORDS"
  prepare_broad_pool 0
  if [[ "$DRY_RUN" == "1" ]]; then
    python3 -m tools.meditod_pipeline_support mock-score --input "$PRIORITIZED_CANDIDATES" --output "$SCORED_RAW" --bayes-model "$COMPAT_MODEL"
    pilot="$(wc -l < "$SCORED_RAW")"
  elif [[ ! -f "$SCORED_RAW" ]]; then
    retry_command python3 -m tools.score_dialogue_with_transition_bayes_model --input "$PRIORITIZED_CANDIDATES" --bayes-model "$COMPAT_MODEL" --output "$SCORED_RAW" --model "$SCORING_MODEL" --workers "$WORKERS" --max-records "$pilot" --include-crossing-conversation --scoring-preset "$SCORING_PRESET" --invalid-observation-retries 2 --requests-per-minute "$SCORING_REQUESTS_PER_MINUTE" --rate-limit-max-retries "$SCORING_RATE_LIMIT_MAX_RETRIES" --rate-limit-initial-backoff-seconds "$SCORING_RATE_LIMIT_BACKOFF_SECONDS" --fallback-on-errors
  fi
  python3 -m tools.validate_mathdial_scoring_pilot --input "$SCORED_RAW" --bayes-model "$COMPAT_MODEL" --output "$OUTPUT_ROOT/scoring/pilot_diagnostics.json" --required-records "$pilot" --max-fallback-rate 0.01 --max-invalid-rate 0.01 --min-observations 2
  reconcile_scoring
  [[ "$DRY_RUN" == "1" ]] && { measure_selection_pool 2 || true; return; }
  while true; do
    measure_selection_pool "$DPO_INITIAL_SELECTION_POOL_COUNT"
    local eligible
    eligible="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["eligible_records"])' "$OUTPUT_ROOT/scoring/selection_pool_progress.json")"
    (( eligible >= DPO_INITIAL_SELECTION_POOL_COUNT )) && break
    score_next_batch || {
      [[ "$?" == "21" ]] && break
      return 1
    }
  done
  python3 -m tools.validate_scoring_fallbacks --input "$SCORED" --output "$OUTPUT_ROOT/scoring/fallback_diagnostics.json" --warning-rate 0.01 --fatal-rate 0.05 --diagnostic-only
}

select_data_stage() {
  local requested_random_count="${1:-0}"
  local count random_count
  if [[ "$DRY_RUN" == "1" ]]; then count=2; random_count=2; else
    local eligible; eligible="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["eligible_records"])' "$OUTPUT_ROOT/scoring/selection_pool_progress.json")"
    count="$eligible"
    local broad_count
    broad_count="$(wc -l < "$HEALTH_CANDIDATES")"
    random_count=$(( RANDOM_DPO_COUNT * 2 ))
    (( random_count < count )) && random_count="$count"
    (( requested_random_count > random_count )) && random_count="$requested_random_count"
    (( random_count > broad_count )) && random_count="$broad_count"
    (( count > 0 )) || { echo "BASiS選別候補がありません。" >&2; return 20; }
    (( random_count >= RANDOM_DPO_COUNT )) || { echo "広域健康Random候補が不足しています: $random_count/$RANDOM_DPO_COUNT" >&2; return 20; }
  fi
  python3 -m tools.meditod_selection --scored "$SCORED" --domain-candidates "$HEALTH_CANDIDATES" --meditod-conversations "$MED_CONV" --wildchat-conversations "$HEALTH_CONVERSATIONS" --health-config configs/datasets/wildchat_health.yaml --bayes-model "$COMPAT_MODEL" --output-dir "$SELECT_DIR" --count "$count" --random-count "$random_count" --seed "$SEED" --selection-margin 0.05 --max-source-characters "$DPO_MAX_SOURCE_CHARACTERS"
}

extend_scoring_selection() {
  score_next_batch || return $?
  measure_selection_pool "$DPO_INITIAL_SELECTION_POOL_COUNT"
  # build_dpo中の追加scoringも正当なresume成果物として記録する。
  write_stage_marker score_wildchat
  select_data_stage
  write_stage_marker select_data
}

build_dpo_stage() {
  mkdir -p "$DPO_DIR"
  local basis="$BASIS_SELECTED_COUNT" gold="$GOLD_DPO_COUNT" random="$RANDOM_DPO_COUNT" gold_source=1200
  [[ "$DRY_RUN" == "1" ]] && { basis=1; gold=1; random=2; gold_source=2; }
  prepare_broad_pool 1
  reconcile_scoring
  measure_selection_pool "$DPO_INITIAL_SELECTION_POOL_COUNT"
  if [[ "$DRY_RUN" != "1" ]]; then
    while true; do
      local initial_eligible
      initial_eligible="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["eligible_records"])' "$OUTPUT_ROOT/scoring/selection_pool_progress.json")"
      (( initial_eligible >= DPO_INITIAL_SELECTION_POOL_COUNT )) && break
      extend_scoring_selection || {
        [[ "$?" == "21" ]] && break
        return 1
      }
    done
  fi
  if [[ ! -s "$SELECT_DIR/basis_top.jsonl" || ! -s "$SELECT_DIR/domain_random.jsonl" || ! -s "$SELECT_DIR/topic_similarity_top.jsonl" ]]; then
    select_data_stage
    write_stage_marker select_data
  fi
  python3 -m tools.prepare_meditod_gold \
    --samples "$MED_SAMPLES" \
    --output "$DPO_DIR/gold_candidates_en.jsonl" \
    --target "$gold_source" \
    --allow-target-shortfall \
    --minimum-records "$gold" \
    --seed "$SEED"
  if [[ "$DRY_RUN" == "1" ]]; then
    python3 -m tools.meditod_pipeline_support mock-dpo --input "$SELECT_DIR/basis_top.jsonl" --output "$DPO_DIR/basis_selected_ja.jsonl" --count "$basis" --source-dataset WildChat-BASiS
    python3 -m tools.meditod_pipeline_support mock-dpo --input "$DPO_DIR/gold_candidates_en.jsonl" --output "$DPO_DIR/meditod_gold_ja.jsonl" --count "$gold" --source-dataset MediTOD --gold
    python3 -m tools.meditod_pipeline_support mock-dpo --input "$SELECT_DIR/domain_random.jsonl" --output "$DPO_DIR/random_ja.jsonl" --count "$random" --source-dataset WildChat-Random
  else
    while true; do
      retry_command python3 -m tools.translate_and_generate_dpo --input "$SELECT_DIR/basis_top.jsonl" --bayes-model "$COMPAT_MODEL" --output "$DPO_DIR/basis_selected_ja.jsonl" --model "$GENERATION_MODEL" --score-model "$SCORING_MODEL" --style-preset meditod_history_taking --candidates 4 --max-output-tokens "$DPO_MAX_OUTPUT_TOKENS" --max-source-characters "$DPO_MAX_SOURCE_CHARACTERS" --min-score-gap 0.20 --min-chosen-posterior 0.70 --max-rejected-posterior 0.55 --target-records "$basis" --workers "$WORKERS" --skip-sample-errors --allow-target-shortfall --heartbeat-file "$HEARTBEAT_FILE" --heartbeat-stage-prefix basis_dpo --seed "$SEED"
      local accepted
      accepted="$(wc -l < "$DPO_DIR/basis_selected_ja.jsonl")"
      (( accepted >= basis )) && break
      local extend_status=0
      set +e
      extend_scoring_selection
      extend_status=$?
      set -e
      if (( extend_status != 0 )); then
        (( extend_status == 21 )) || return "$extend_status"
        echo "[adaptive scoring] 全広域健康候補を処理したため順位救済を実行します。"
        python3 -m tools.promote_meditod_dpo_rescue \
          --accepted "$DPO_DIR/basis_selected_ja.jsonl" \
          --skipped "$DPO_DIR/basis_selected_ja_skipped.jsonl" \
          --target-records "$basis" \
          --min-chosen "$DPO_RESCUE_MIN_CHOSEN" \
          --max-rejected "$DPO_RESCUE_MAX_REJECTED" \
          --min-gap "$DPO_RESCUE_MIN_GAP" \
          --report "$DPO_DIR/basis_ranked_rescue_report.json"
        break
      fi
    done
    retry_command python3 -m tools.translate_and_generate_dpo --input "$DPO_DIR/gold_candidates_en.jsonl" --bayes-model "$COMPAT_MODEL" --output "$DPO_DIR/meditod_gold_ja.jsonl" --model "$GENERATION_MODEL" --score-model "$SCORING_MODEL" --style-preset meditod_history_taking --candidates 4 --max-output-tokens "$DPO_MAX_OUTPUT_TOKENS" --max-source-characters "$DPO_MAX_SOURCE_CHARACTERS" --min-score-gap 0.20 --min-chosen-posterior 0.70 --max-rejected-posterior 0.55 --target-records "$gold" --workers "$WORKERS" --skip-sample-errors --heartbeat-file "$HEARTBEAT_FILE" --heartbeat-stage-prefix gold_dpo --seed "$SEED"
    local random_selection_count=$(( random * 2 ))
    while true; do
      retry_command python3 -m tools.build_random_dailydialog_dpo --input "$SELECT_DIR/domain_random.jsonl" --source-dataset WildChat --prompt-preset meditod_history_taking --output "$DPO_DIR/random_ja.jsonl" --daily-output "$DPO_DIR/random_ja.jsonl" --target-records "$random" --candidates 1 --max-output-tokens "$DPO_MAX_OUTPUT_TOKENS" --model "$GENERATION_MODEL" --workers "$WORKERS" --skip-sample-errors --allow-target-shortfall --heartbeat-file "$HEARTBEAT_FILE" --seed "$SEED"
      local random_accepted broad_count
      random_accepted="$(wc -l < "$DPO_DIR/random_ja.jsonl")"
      (( random_accepted >= random )) && break
      broad_count="$(wc -l < "$HEALTH_CANDIDATES")"
      (( random_selection_count < broad_count )) || {
        echo "全広域健康候補を使ってもRandom DPOが不足しました: $random_accepted/$random" >&2
        return 20
      }
      random_selection_count=$(( random_selection_count + random ))
      (( random_selection_count > broad_count )) && random_selection_count="$broad_count"
      echo "[adaptive random] 候補を${random_selection_count}件へ拡張します。"
      select_data_stage "$random_selection_count"
      write_stage_marker select_data
    done
  fi
  python3 -m tools.mix_meditod_dpo --basis "$DPO_DIR/basis_selected_ja.jsonl" --gold "$DPO_DIR/meditod_gold_ja.jsonl" --random "$DPO_DIR/random_ja.jsonl" --basis-output "$DPO_DIR/meditod_basis_train.jsonl" --random-output "$DPO_DIR/meditod_random_train.jsonl" --basis-count "$basis" --gold-count "$gold" --random-count "$random"
}

train_stage() {
  mkdir -p "$TRAIN_DIR"
  local common=(--model-id "$LOCAL_MODEL" --num-train-epochs 1 --learning-rate 5e-6 --beta 0.1 --per-device-train-batch-size 1 --gradient-accumulation-steps 8 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --save-steps 25 --save-total-limit "$TRAIN_SAVE_TOTAL_LIMIT" --warmup-ratio 0.03 --eval-ratio 0 --seed "$SEED" --no-4bit --device-map "$TRAIN_DEVICE_MAP" --max-memory "$TRAIN_MAX_MEMORY" --resume-from-checkpoint auto)
  train_one() {
    local dataset="$1" output_dir="$2" attempt status offset
    for attempt in 1 2; do
      offset="$(stat -c %s "$LOG_FILE")"
      set +e
      env CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" python3 -m tools.train_qwen35_dpo_lora --dataset "$dataset" --output-dir "$output_dir" "${common[@]}"
      status=$?
      set -e
      (( status == 0 )) && return 0
      if tail -c "+$((offset + 1))" "$LOG_FILE" | grep -qiE 'CUDA out of memory|OutOfMemoryError'; then
        if (( attempt == 1 )); then
          echo "[train] 同一条件・最新checkpointでOOMを1回だけ再試行します。" >&2
          sleep 10
          continue
        fi
        echo "[train] 同一条件でOOMが再発したため研究条件を変更せず停止します。" >&2
        return 20
      fi
      return "$status"
    done
  }
  python3 -m tools.train_qwen35_dpo_lora --dataset "$DPO_DIR/meditod_basis_train.jsonl" --output-dir "$TRAIN_DIR/basis_lora" "${common[@]}" --dry-run
  python3 -m tools.train_qwen35_dpo_lora --dataset "$DPO_DIR/meditod_random_train.jsonl" --output-dir "$TRAIN_DIR/random_lora" "${common[@]}" --dry-run
  if [[ "$DRY_RUN" == "1" ]]; then
    mkdir -p "$TRAIN_DIR/basis_lora" "$TRAIN_DIR/random_lora"
    printf '{}\n' > "$TRAIN_DIR/basis_lora/adapter_config.json"
    printf '{}\n' > "$TRAIN_DIR/random_lora/adapter_config.json"
    return
  fi
  gpu_preflight "$TRAIN_CUDA_VISIBLE_DEVICES" training
  train_one "$DPO_DIR/meditod_basis_train.jsonl" "$TRAIN_DIR/basis_lora"
  gpu_preflight "$TRAIN_CUDA_VISIBLE_DEVICES" random_training
  train_one "$DPO_DIR/meditod_random_train.jsonl" "$TRAIN_DIR/random_lora"
}

prepare_eval_stage() {
  mkdir -p "$EVAL_DIR"
  local main_count="$EVAL_COUNT" ood_count="$OOD_EVAL_COUNT" mock=()
  [[ "$DRY_RUN" == "1" ]] && { main_count=2; ood_count=1; mock+=(--mock); }
  python3 -m tools.meditod_evaluation prepare --samples "$MED_SAMPLES" --output "$EVAL_DIR/prompts_ja.jsonl" --candidate-output "$EVAL_DIR/prompt_candidates_ja.jsonl" --manifest "$EVAL_DIR/prompt_selection_manifest.json" --errors-output "$EVAL_DIR/translation_errors.jsonl" --count "$main_count" --seed "$SEED" --max-per-consultation 6 --candidate-reserve -1 --allow-exhausted-shortfall --model "$SCORING_MODEL" --workers "$WORKERS" --requests-per-minute "$SCORING_REQUESTS_PER_MINUTE" --resume "${mock[@]}"
  if (( ood_count > 0 )); then
    python3 -m tools.meditod_evaluation prepare --samples "$MED_SAMPLES" --output "$EVAL_DIR/ood_prompts_ja.jsonl" --candidate-output "$EVAL_DIR/ood_prompt_candidates_ja.jsonl" --manifest "$EVAL_DIR/ood_prompt_selection_manifest.json" --errors-output "$EVAL_DIR/ood_translation_errors.jsonl" --count "$ood_count" --seed "$SEED" --max-per-consultation 6 --candidate-reserve -1 --allow-exhausted-shortfall --model "$SCORING_MODEL" --workers "$WORKERS" --requests-per-minute "$SCORING_REQUESTS_PER_MINUTE" --resume --ood "${mock[@]}"
  fi
  python3 - "$EVAL_DIR/prompts_ja.jsonl" "$EVAL_DIR/ood_prompts_ja.jsonl" "$EVAL_DIR/prompts_all_ja.jsonl" <<'PY'
import pathlib,sys
target=pathlib.Path(sys.argv[3]); target.parent.mkdir(parents=True,exist_ok=True)
with target.open("w",encoding="utf-8") as out:
 for name in sys.argv[1:3]:
  path=pathlib.Path(name)
  if path.exists(): out.write(path.read_text(encoding="utf-8"))
PY
}

generate_responses_stage() {
  local mock=(); [[ "$DRY_RUN" == "1" ]] && mock+=(--mock)
  gpu_preflight "$EVAL_CUDA_VISIBLE_DEVICES" evaluation_generation
  env CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" DPO_COMPARE_MAX_MEMORY="$TRAIN_MAX_MEMORY" python3 -m tools.meditod_evaluation generate --input "$EVAL_DIR/prompts_all_ja.jsonl" --output "$EVAL_DIR/responses.jsonl" --oracle-output "$EVAL_DIR/oracle_input_all.jsonl" --errors-output "$EVAL_DIR/generation_errors.jsonl" --base-model "$LOCAL_MODEL" --basis-lora "$TRAIN_DIR/basis_lora" --random-lora "$TRAIN_DIR/random_lora" --seed "$SEED" "${mock[@]}"
  python3 - "$EVAL_DIR/oracle_input_all.jsonl" "$EVAL_DIR/oracle_input.jsonl" "$EVAL_DIR/oracle_input_ood.jsonl" <<'PY'
import json,pathlib,sys
outputs=[pathlib.Path(sys.argv[2]),pathlib.Path(sys.argv[3])]
files=[path.open("w",encoding="utf-8") for path in outputs]
try:
 for line in pathlib.Path(sys.argv[1]).open(encoding="utf-8"):
  if not line.strip(): continue
  row=json.loads(line); files[bool(row.get("metadata",{}).get("ood"))].write(json.dumps(row,ensure_ascii=False)+"\n")
finally:
 [f.close() for f in files]
PY
  cp "$EVAL_DIR/oracle_input_all.jsonl" "$EVAL_DIR/oracle_input.jsonl.all"
  python3 -m tools.meditod_annotation_metrics --input "$EVAL_DIR/responses.jsonl" --output "$EVAL_DIR/annotation_metrics.csv" --summary "$EVAL_DIR/annotation_metrics_summary.json"
}

run_oracle_set() {
  local input="$1" suffix="$2" dry=()
  [[ "$DRY_RUN" == "1" ]] && dry+=(--dry-run)
  local category
  for category in history general safety; do
    python3 -m scripts.eval_oracle_meditod --input "$input" --output_dir "$EVAL_DIR/oracle/${suffix}${category}" --category "$category" --judge_model "$JUDGE_MODEL" --score-scale 10 --oracle-workers "$WORKERS" --resume "${dry[@]}"
  done
}

oracle_eval_stage() {
  run_oracle_set "$EVAL_DIR/oracle_input.jsonl" ""
  [[ -s "$EVAL_DIR/oracle_input_ood.jsonl" ]] && run_oracle_set "$EVAL_DIR/oracle_input_ood.jsonl" "ood_"
}

statistics_stage() {
  local permutations=10000 bootstrap=2000
  [[ "$DRY_RUN" == "1" ]] && { permutations=100; bootstrap=100; }
  python3 -m scripts.run_meditod_statistics --raw "$EVAL_DIR/oracle/history/raw.jsonl" --raw "$EVAL_DIR/oracle/general/raw.jsonl" --raw "$EVAL_DIR/oracle/safety/raw.jsonl" --oracle-input "$EVAL_DIR/oracle_input.jsonl" --output-dir "$EVAL_DIR/statistics" --permutations "$permutations" --bootstrap "$bootstrap" --seed "$SEED"
  if [[ -s "$EVAL_DIR/oracle_input_ood.jsonl" ]]; then
    python3 -m scripts.run_meditod_statistics --raw "$EVAL_DIR/oracle/ood_history/raw.jsonl" --raw "$EVAL_DIR/oracle/ood_general/raw.jsonl" --raw "$EVAL_DIR/oracle/ood_safety/raw.jsonl" --oracle-input "$EVAL_DIR/oracle_input_ood.jsonl" --output-dir "$EVAL_DIR/statistics_ood" --permutations "$permutations" --bootstrap "$bootstrap" --seed "$SEED"
  fi
}

report_stage() {
  python3 -m tools.meditod_pipeline_support report --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/reports/final_report.md"
}

prepare_user_eval_stage() {
  local count=20
  [[ "$DRY_RUN" == "1" ]] && count=2
  python3 -m tools.prepare_three_model_likert_eval --dataset meditod --responses "$EVAL_DIR/responses.jsonl" --oracle-raw "$EVAL_DIR/oracle/history/raw.jsonl" --output-root "$OUTPUT_ROOT/user_eval" --count "$count" --seed "$SEED"
}

preflight_storage
run_stage preprocess preprocess_stage
run_stage build_basis build_basis_stage
run_stage extract_wildchat extract_wildchat_stage
run_stage score_wildchat score_wildchat_stage
run_stage select_data select_data_stage
run_stage build_dpo build_dpo_stage
run_stage train train_stage
run_stage prepare_eval prepare_eval_stage
run_stage generate_responses generate_responses_stage
run_stage oracle_eval oracle_eval_stage
run_stage statistics statistics_stage
run_stage report report_stage
run_stage prepare_user_eval prepare_user_eval_stage

PIPELINE_SUCCEEDED=1
write_status success completed "pipeline completed"
echo "MediTOD pipeline completed: $OUTPUT_ROOT"
echo "Log: $LOG_FILE"
