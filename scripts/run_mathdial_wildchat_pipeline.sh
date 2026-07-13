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

RUN_TAG="${RUN_TAG:-mathdial_wildchat_gpt56_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/mathdial_wildchat/runs/${RUN_TAG}}"
REUSE_DATA_RUN_TAG="${REUSE_DATA_RUN_TAG:-}"
REUSE_DATA_ROOT="${REUSE_DATA_ROOT:-${REUSE_DATA_RUN_TAG:+artifacts/mathdial_wildchat/runs/${REUSE_DATA_RUN_TAG}}}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-4}"
MATHDIAL_ANALYSIS_CONVERSATIONS="${MATHDIAL_ANALYSIS_CONVERSATIONS:-80}"
MATHDIAL_ANALYSIS_MAX_INPUT_CHARS="${MATHDIAL_ANALYSIS_MAX_INPUT_CHARS:-300000}"
MATHDIAL_ANALYSIS_MAX_OUTPUT_TOKENS="${MATHDIAL_ANALYSIS_MAX_OUTPUT_TOKENS:-24000}"
MAX_SCORING_FALLBACK_RATE="${MAX_SCORING_FALLBACK_RATE:-0.01}"
MAX_SCORING_INVALID_RATE="${MAX_SCORING_INVALID_RATE:-0.01}"
SCORING_PILOT_RECORDS="${SCORING_PILOT_RECORDS:-200}"
SCORING_PRESET="${SCORING_PRESET:-mathdial_tutoring}"
SCORING_PRESET_VERSION="${SCORING_PRESET_VERSION:-mathdial_v3}"
INVALID_OBSERVATION_RETRIES="${INVALID_OBSERVATION_RETRIES:-2}"
SELECTION_LABEL_METHOD="${SELECTION_LABEL_METHOD:-state_specific_margin}"
SELECTION_EMISSION_MARGIN="${SELECTION_EMISSION_MARGIN:-0.05}"
MODEL_EMISSION_QUALITY_MARGIN="${MODEL_EMISSION_QUALITY_MARGIN:-0.10}"
MODEL_MIN_NEGATIVE_OBSERVATIONS="${MODEL_MIN_NEGATIVE_OBSERVATIONS:-2}"
SELECTION_POOL_COUNT="${SELECTION_POOL_COUNT:-5000}"
WILDCHAT_SCORING_TARGET_RECORDS="${WILDCHAT_SCORING_TARGET_RECORDS:-$((SELECTION_POOL_COUNT * 4))}"
WILDCHAT_CANDIDATE_TARGET_RECORDS="${WILDCHAT_CANDIDATE_TARGET_RECORDS:-$((WILDCHAT_SCORING_TARGET_RECORDS + SELECTION_POOL_COUNT))}"
WILDCHAT_CHECKPOINT_EVERY="${WILDCHAT_CHECKPOINT_EVERY:-100000}"
ANALYSIS_MODEL="${MATHDIAL_ANALYSIS_LLM_MODEL:-${AZURE_OPENAI_GPT56_SOL_DEPLOYMENT:-gpt-5.6-sol}}"
SCORING_MODEL="${MATHDIAL_SCORING_LLM_MODEL:-${AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT:-gpt-5.6-terra}}"
GENERATION_MODEL="${MATHDIAL_DPO_GENERATION_MODEL:-${SCORING_MODEL}}"
JUDGE_MODEL="${MATHDIAL_JUDGE_MODEL:-${SCORING_MODEL}}"
LOCAL_MODEL="${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"
CUDA_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
START_STAGE="${START_STAGE:-preprocess}"
END_STAGE="${END_STAGE:-report}"
STAGE="${STAGE:-}"
FORCE_STAGE="${FORCE_STAGE:-}"

STAGES=(preprocess build_basis extract_wildchat score_wildchat select_data build_dpo train generate_responses oracle_eval statistics report)
if [[ -n "$STAGE" ]]; then START_STAGE="$STAGE"; END_STAGE="$STAGE"; fi

LOG_DIR="$OUTPUT_ROOT/logs"
STATE_DIR="$OUTPUT_ROOT/stage_state"
STATUS_FILE="$OUTPUT_ROOT/pipeline_status.json"
HEARTBEAT_FILE="${PIPELINE_HEARTBEAT_FILE:-$OUTPUT_ROOT/pipeline_heartbeat.json}"
mkdir -p "$LOG_DIR" "$STATE_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

EXPERIMENT_FINGERPRINT="$(python3 - "$RUN_TAG" "$SEED" "$DRY_RUN" "$ANALYSIS_MODEL" "$SCORING_MODEL" "$GENERATION_MODEL" "$JUDGE_MODEL" "$LOCAL_MODEL" "$SELECTION_POOL_COUNT" "$WILDCHAT_CANDIDATE_TARGET_RECORDS" "$WILDCHAT_SCORING_TARGET_RECORDS" "$MATHDIAL_ANALYSIS_CONVERSATIONS" "$MATHDIAL_ANALYSIS_MAX_INPUT_CHARS" "$MATHDIAL_ANALYSIS_MAX_OUTPUT_TOKENS" "$MAX_SCORING_FALLBACK_RATE" "$MAX_SCORING_INVALID_RATE" "$SCORING_PILOT_RECORDS" "$SCORING_PRESET" "$SCORING_PRESET_VERSION" "$INVALID_OBSERVATION_RETRIES" "$SELECTION_LABEL_METHOD" "$SELECTION_EMISSION_MARGIN" "$MODEL_EMISSION_QUALITY_MARGIN" "$MODEL_MIN_NEGATIVE_OBSERVATIONS" "$REUSE_DATA_RUN_TAG" "${REUSE_DATA_ROOT:-}" <<'PY'
import hashlib,json,pathlib,sys
configs=["configs/datasets/mathdial.yaml","configs/datasets/wildchat_tutoring.yaml","configs/evaluations/mathdial_oracle_v1.yaml","configs/training/mathdial_dpo.yaml","tools/mathdial_dataset.py","tools/prepare_mathdial.py","tools/prepare_mathdial_for_analysis.py","tools/analyze_mathdial_corpus_transition_bayes.py","tools/score_dialogue_with_transition_bayes_model.py","tools/translate_and_generate_dpo.py","tools/extract_high_posterior_dialogues.py","tools/mathdial_selection.py","tools/validate_mathdial_scoring_pilot.py","tools/reuse_mathdial_pipeline_data.py","scripts/run_mathdial_wildchat_pipeline.sh"]
payload={"values":sys.argv[1:],"files":{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in configs}}
reuse_root=pathlib.Path(sys.argv[-1]) if sys.argv[-1] else None
if reuse_root and (reuse_root/"run_metadata.json").exists():
 payload["reuse_run_metadata_sha256"]=hashlib.sha256((reuse_root/"run_metadata.json").read_bytes()).hexdigest()
print(hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest())
PY
)"

python3 - "$OUTPUT_ROOT/run_metadata.json" "$OUTPUT_ROOT/run_attempts.jsonl" "$EXPERIMENT_FINGERPRINT" "$RUN_TAG" "$SEED" "$DRY_RUN" "$ANALYSIS_MODEL" "$SCORING_MODEL" "$GENERATION_MODEL" "$JUDGE_MODEL" "$LOCAL_MODEL" "$WORKERS" "$SELECTION_POOL_COUNT" "$WILDCHAT_CANDIDATE_TARGET_RECORDS" "$WILDCHAT_SCORING_TARGET_RECORDS" "$MATHDIAL_ANALYSIS_CONVERSATIONS" "$MATHDIAL_ANALYSIS_MAX_INPUT_CHARS" "$MATHDIAL_ANALYSIS_MAX_OUTPUT_TOKENS" "$MAX_SCORING_FALLBACK_RATE" "$MAX_SCORING_INVALID_RATE" "$SCORING_PILOT_RECORDS" "$SCORING_PRESET" "$SCORING_PRESET_VERSION" "$INVALID_OBSERVATION_RETRIES" "$SELECTION_LABEL_METHOD" "$SELECTION_EMISSION_MARGIN" "$MODEL_EMISSION_QUALITY_MARGIN" "$MODEL_MIN_NEGATIVE_OBSERVATIONS" "$REUSE_DATA_RUN_TAG" <<'PY'
import datetime,hashlib,json,pathlib,sys
path=pathlib.Path(sys.argv[1]); attempts=pathlib.Path(sys.argv[2]); fingerprint=sys.argv[3]
configs=["configs/datasets/mathdial.yaml","configs/datasets/wildchat_tutoring.yaml","configs/evaluations/mathdial_oracle_v1.yaml","configs/training/mathdial_dpo.yaml","tools/mathdial_dataset.py","tools/prepare_mathdial.py","tools/prepare_mathdial_for_analysis.py","tools/analyze_mathdial_corpus_transition_bayes.py","tools/score_dialogue_with_transition_bayes_model.py","tools/translate_and_generate_dpo.py","tools/extract_high_posterior_dialogues.py","tools/mathdial_selection.py","tools/validate_mathdial_scoring_pilot.py","tools/reuse_mathdial_pipeline_data.py","scripts/run_mathdial_wildchat_pipeline.sh"]
payload={"experiment_fingerprint":fingerprint,"run_tag":sys.argv[4],"seed":int(sys.argv[5]),"dry_run":sys.argv[6]=="1","models":{"analysis":sys.argv[7],"scoring":sys.argv[8],"generation":sys.argv[9],"judge":sys.argv[10],"local":sys.argv[11]},"early_stop":{"selection_pool_records":int(sys.argv[13]),"wildchat_candidate_records":int(sys.argv[14]),"wildchat_scoring_records":int(sys.argv[15]),"scoring_pilot_records":int(sys.argv[21])},"basis_analysis":{"conversations":int(sys.argv[16]),"max_input_chars":int(sys.argv[17]),"max_output_tokens":int(sys.argv[18])},"scoring":{"preset":sys.argv[22],"preset_version":sys.argv[23],"invalid_observation_retries":int(sys.argv[24])},"selection":{"label_derivation_method":sys.argv[25],"emission_margin":float(sys.argv[26])},"quality_gates":{"max_scoring_fallback_rate":float(sys.argv[19]),"max_scoring_invalid_rate":float(sys.argv[20]),"model_emission_margin":float(sys.argv[27]),"minimum_negative_observations":int(sys.argv[28])},"reuse_data_run_tag":sys.argv[29],"configs":{name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in configs}}
if path.exists():
    current=json.loads(path.read_text())
    if current.get("experiment_fingerprint") != fingerprint:
        raise SystemExit("同じRUN_TAGの実験条件が変わっています。新しいRUN_TAGを使うか、既存runを明示的に退避してください。")
else:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
attempt={"timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),"experiment_fingerprint":fingerprint,"workers":int(sys.argv[12])}
with attempts.open("a",encoding="utf-8") as f: f.write(json.dumps(attempt,ensure_ascii=False)+"\n")
PY

stage_index() {
  local target="$1" i
  for i in "${!STAGES[@]}"; do [[ "${STAGES[$i]}" == "$target" ]] && { echo "$i"; return; }; done
  echo "Unknown stage: $target" >&2; return 1
}
START_INDEX="$(stage_index "$START_STAGE")"
END_INDEX="$(stage_index "$END_STAGE")"
[[ "$START_INDEX" -le "$END_INDEX" ]] || { echo "START_STAGE must precede END_STAGE" >&2; exit 2; }

run_stage() {
  local name="$1" index marker
  shift
  index="$(stage_index "$name")"
  [[ "$index" -ge "$START_INDEX" && "$index" -le "$END_INDEX" ]] || return 0
  marker="$STATE_DIR/${name}_SUCCESS.json"
  if [[ -f "$marker" && "$FORCE_STAGE" != "$name" && "$FORCE_STAGE" != "all" ]]; then
    python3 - "$marker" "$EXPERIMENT_FINGERPRINT" <<'PY'
import hashlib,json,pathlib,sys
payload=json.load(open(sys.argv[1]))
if payload.get("experiment_fingerprint") != sys.argv[2]:
    raise SystemExit(f"stage markerのfingerprintが一致しません: {sys.argv[1]}")
for value,expected in payload.get("input_hashes",{}).items():
 path=pathlib.Path(value)
 if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
  raise SystemExit(f"stage marker作成後に入力が変わっています。FORCE_STAGEで明示再実行してください: {value}")
PY
    echo "[SKIP] $name completed: $marker"; return 0
  fi
  echo "[START] $name"
  CURRENT_STAGE="$name"
  write_runtime_state "running" "$name" "stage started"
  "$@"
  python3 - "$marker" "$name" "$RUN_TAG" "$SEED" "$DRY_RUN" "$EXPERIMENT_FINGERPRINT" "$OUTPUT_ROOT" "$ANALYSIS_MODEL" "$SCORING_MODEL" "$GENERATION_MODEL" "$JUDGE_MODEL" "$LOCAL_MODEL" <<'PY'
import datetime,hashlib,json,pathlib,sys
path=pathlib.Path(sys.argv[1]); stage=sys.argv[2]; root=pathlib.Path(sys.argv[7])
inputs={
 "preprocess":[root/"mathdial/data/mathdial_conversations.jsonl",root/"mathdial/data/mathdial_assistant_samples.jsonl"],
 "build_basis":[root/"mathdial/data/mathdial_conversations.jsonl"],
 "extract_wildchat":[root/"wildchat/general_tutoring_candidates.jsonl",root/"wildchat/math_tutoring_candidates.jsonl",root/"wildchat/manifest.json"],
 "score_wildchat":[root/"wildchat/general_tutoring_candidates.jsonl",root/"basis_model/mathdial_transition_compat.json"],
 "select_data":[root/"scoring/wildchat_scored.jsonl",root/"basis_model/mathdial_transition_compat.json"],
 "build_dpo":[root/"selections/basis_top.jsonl",root/"selections/domain_random.jsonl",root/"basis_model/mathdial_transition_compat.json"],
 "train":[root/"dpo/mathdial_basis_train.jsonl",root/"dpo/mathdial_random_train.jsonl"],
 "generate_responses":[root/"mathdial/data/mathdial_assistant_samples.jsonl"],
 "oracle_eval":[root/"evaluation/oracle_input.jsonl"],
 "statistics":[root/"evaluation/oracle/pedagogical/raw.jsonl",root/"evaluation/oracle/general/raw.jsonl"],
}.get(stage,[])
def digest(value): return hashlib.sha256(value.read_bytes()).hexdigest()
def count(value):
 if value.suffix!=".jsonl": return None
 with value.open(encoding="utf-8",errors="replace") as f: return sum(bool(line.strip()) for line in f)
metadata_path=root/"run_metadata.json"
run_metadata=json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
payload={
 "stage":stage,"run_tag":sys.argv[3],"seed":int(sys.argv[4]),"dry_run":sys.argv[5]=="1",
 "experiment_fingerprint":sys.argv[6],"completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "input_hashes":{str(value):digest(value) for value in inputs if value.exists()},
 "input_counts":{str(value):count(value) for value in inputs if value.exists() and value.suffix==".jsonl"},
 "config_hashes":run_metadata.get("configs",{}),
 "models":{"analysis":sys.argv[8],"scoring":sys.argv[9],"generation":sys.argv[10],"judge":sys.argv[11],"local":sys.argv[12]},
}
path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); temporary.replace(path)
PY
  write_runtime_state "running" "$name" "stage completed"
  echo "[DONE] $name"
}

write_runtime_state() {
  local state="$1" stage="$2" message="$3"
  python3 - "$STATUS_FILE" "$HEARTBEAT_FILE" "$state" "$stage" "$message" "$RUN_TAG" "${WATCHDOG_ATTEMPT:-1}" "$OUTPUT_ROOT" <<'PY'
import datetime,json,pathlib,sys
root=pathlib.Path(sys.argv[8])
def count(relative):
 path=root/relative
 if not path.exists(): return 0
 with path.open(encoding="utf-8",errors="replace") as f: return sum(bool(line.strip()) for line in f)
counts={
 "analysis_conversations":count("basis_model/mathdial_analysis_corpus.jsonl"),
 "wildchat_candidates":count("wildchat/general_tutoring_candidates.jsonl"),
 "scored":count("scoring/wildchat_scored.jsonl"),
 "basis_selected":count("selections/basis_top.jsonl"),
 "basis_dpo":count("dpo/mathdial_basis_train.jsonl"),
 "random_dpo":count("dpo/mathdial_random_train.jsonl"),
 "evaluation_responses":count("evaluation/responses.jsonl"),
 "oracle_pedagogical":count("evaluation/oracle/pedagogical/raw.jsonl"),
 "oracle_general":count("evaluation/oracle/general/raw.jsonl"),
}
skip_counts={
 "basis_dpo":count("dpo/basis_selected_ja_skipped.jsonl"),
 "gold_dpo":count("dpo/mathdial_gold_ja_skipped.jsonl"),
 "random_dpo":count("dpo/random_ja.skipped.jsonl"),
 "evaluation_translation":count("evaluation/translation_errors.jsonl"),
 "evaluation_generation":count("evaluation/generation_errors.jsonl"),
 "oracle_pedagogical":count("evaluation/oracle/pedagogical/errors.jsonl"),
 "oracle_general":count("evaluation/oracle/general/errors.jsonl"),
}
fallback=0
scored=root/"scoring/wildchat_scored.jsonl"
if scored.exists():
 for line in scored.open(encoding="utf-8",errors="replace"):
  try: fallback += bool(json.loads(line).get("llm_error"))
  except Exception: pass
payload={"timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),"state":sys.argv[3],"stage":sys.argv[4],"message":sys.argv[5],"run_tag":sys.argv[6],"attempt":int(sys.argv[7]),"success_counts":counts,"skip_counts":skip_counts,"skip_count":sum(skip_counts.values()),"fallback_count":fallback,"fallback_rate":fallback/max(1,counts["scored"])}
for value in sys.argv[1:3]:
 p=pathlib.Path(value); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
PY
}

PIPELINE_SUCCEEDED=0
on_pipeline_exit() {
  local status=$?
  if [[ "$PIPELINE_SUCCEEDED" != "1" ]]; then
    local state="incomplete"
    [[ "$status" -eq 20 ]] && state="fatal"
    write_runtime_state "$state" "${CURRENT_STAGE:-startup}" "pipeline exited status=$status" || true
  fi
}
trap on_pipeline_exit EXIT

retry_command() {
  local attempt=1 max_attempts="${COMMAND_MAX_ATTEMPTS:-4}" delay="${COMMAND_RETRY_DELAY_SECONDS:-15}"
  while true; do
    if "$@"; then
      return 0
    fi
    if [[ "$attempt" -ge "$max_attempts" ]]; then
      echo "[RETRY] command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    echo "[RETRY] command failed attempt=${attempt}/${max_attempts}; retrying in ${delay}s: $*" >&2
    sleep "$delay"
    delay=$((delay * 2))
    attempt=$((attempt + 1))
  done
}

MATH_ROOT="$OUTPUT_ROOT/mathdial"
MATH_CONV="$MATH_ROOT/data/mathdial_conversations.jsonl"
MATH_SAMPLES="$MATH_ROOT/data/mathdial_assistant_samples.jsonl"
ANALYSIS_CORPUS="$OUTPUT_ROOT/basis_model/mathdial_analysis_corpus.jsonl"
ANALYSIS_CORPUS_MANIFEST="$OUTPUT_ROOT/basis_model/mathdial_analysis_corpus.manifest.json"
FINE_MODEL="$OUTPUT_ROOT/basis_model/mathdial_transition_bayes_model.json"
COMPAT_MODEL="$OUTPUT_ROOT/basis_model/mathdial_transition_compat.json"
WILD_DIR="$OUTPUT_ROOT/wildchat"
SCORED_RAW="$OUTPUT_ROOT/scoring/wildchat_scored_raw.jsonl"
SCORED="$OUTPUT_ROOT/scoring/wildchat_scored.jsonl"
SELECT_DIR="$OUTPUT_ROOT/selections"
DPO_DIR="$OUTPUT_ROOT/dpo"
TRAIN_DIR="$OUTPUT_ROOT/training"
EVAL_DIR="$OUTPUT_ROOT/evaluation"

preprocess_stage() {
  if [[ -n "$REUSE_DATA_RUN_TAG" && "$DRY_RUN" != "1" ]]; then
    python3 -m tools.reuse_mathdial_pipeline_data --source-root "$REUSE_DATA_ROOT" --target-root "$OUTPUT_ROOT" --mode preprocess --seed "$SEED" --project-root "$PROJECT_ROOT"
  else
    python3 -m tools.prepare_mathdial --config configs/datasets/mathdial.yaml --output-root "$MATH_ROOT"
  fi
}

build_basis_stage() {
  python3 -m tools.prepare_mathdial_for_analysis --input "$MATH_CONV" --output "$ANALYSIS_CORPUS" --manifest "$ANALYSIS_CORPUS_MANIFEST" --count "$MATHDIAL_ANALYSIS_CONVERSATIONS" --seed "$SEED"
  local args=(--input "$ANALYSIS_CORPUS" --output "$FINE_MODEL" --compat-output "$COMPAT_MODEL" --manifest "$OUTPUT_ROOT/basis_model/mathdial_transition_bayes_model.manifest.json" --prompt-output "$OUTPUT_ROOT/basis_model/mathdial_analysis_prompt.txt" --input-text-output "$OUTPUT_ROOT/basis_model/mathdial_analysis_input.txt" --quality-report-output "$OUTPUT_ROOT/basis_model/mathdial_model_quality.json" --rejected-models-output "$OUTPUT_ROOT/basis_model/rejected_models.jsonl" --model "$ANALYSIS_MODEL" --max-input-chars "$MATHDIAL_ANALYSIS_MAX_INPUT_CHARS" --max-output-tokens "$MATHDIAL_ANALYSIS_MAX_OUTPUT_TOKENS" --emission-margin "$MODEL_EMISSION_QUALITY_MARGIN" --min-negative-observations "$MODEL_MIN_NEGATIVE_OBSERVATIONS")
  [[ "$DRY_RUN" == "1" ]] && args+=(--mock)
  retry_command python3 -m tools.analyze_mathdial_corpus_transition_bayes "${args[@]}"
  python3 -c 'from core.transition_bayes_model import load_transition_bayes_model; import sys; load_transition_bayes_model(sys.argv[1])' "$COMPAT_MODEL"
}

extract_wildchat_stage() {
  if [[ -n "$REUSE_DATA_RUN_TAG" && "$DRY_RUN" != "1" ]]; then
    python3 -m tools.reuse_mathdial_pipeline_data --source-root "$REUSE_DATA_ROOT" --target-root "$OUTPUT_ROOT" --mode wildchat --seed "$SEED" --project-root "$PROJECT_ROOT"
  else
  local args=(--config configs/datasets/wildchat_tutoring.yaml --output-dir "$WILD_DIR" --seed "$SEED" --target-candidate-records "$WILDCHAT_CANDIDATE_TARGET_RECORDS" --checkpoint-every "$WILDCHAT_CHECKPOINT_EVERY" --heartbeat-file "$HEARTBEAT_FILE")
  if [[ "$DRY_RUN" == "1" ]]; then args+=(--fixture tests/fixtures/wildchat_tutoring.jsonl); else [[ -n "$LIMIT" ]] && args+=(--limit "$LIMIT"); fi
  python3 -m tools.wildchat_tutoring "${args[@]}"
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    if ! python3 - "$WILD_DIR/general_tutoring_candidates.jsonl" "$WILDCHAT_CANDIDATE_TARGET_RECORDS" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
actual = sum(1 for line in path.open(encoding="utf-8") if line.strip())
required = int(sys.argv[2])
print(f"[extract_wildchat] scoring pool coverage={actual}/{required}")
if actual < required:
    raise SystemExit(f"WildChat候補がスコアリング目標に不足しています: {actual}/{required}")
PY
    then
      return 20
    fi
  fi
}

score_wildchat_stage() {
  mkdir -p "$(dirname "$SCORED")"
  local pilot_records="$SCORING_PILOT_RECORDS"
  if [[ "$DRY_RUN" == "1" ]]; then
    python3 -m tools.mathdial_pipeline_support mock-score --input "$WILD_DIR/general_tutoring_candidates.jsonl" --output "$SCORED_RAW" --bayes-model "$COMPAT_MODEL"
    pilot_records=6
  else
    retry_command python3 -m tools.score_dialogue_with_transition_bayes_model --input "$WILD_DIR/general_tutoring_candidates.jsonl" --bayes-model "$COMPAT_MODEL" --output "$SCORED_RAW" --model "$SCORING_MODEL" --workers "$WORKERS" --max-records "$pilot_records" --scoring-preset "$SCORING_PRESET" --invalid-observation-retries "$INVALID_OBSERVATION_RETRIES" --fallback-on-errors
  fi
  python3 -m tools.validate_mathdial_scoring_pilot --input "$SCORED_RAW" --bayes-model "$COMPAT_MODEL" --output "$OUTPUT_ROOT/scoring/pilot_diagnostics.json" --required-records "$pilot_records" --max-fallback-rate "$MAX_SCORING_FALLBACK_RATE" --max-invalid-rate "$MAX_SCORING_INVALID_RATE" --min-observations 2 || return 20
  if [[ "$DRY_RUN" != "1" ]]; then
    retry_command python3 -m tools.score_dialogue_with_transition_bayes_model --input "$WILD_DIR/general_tutoring_candidates.jsonl" --bayes-model "$COMPAT_MODEL" --output "$SCORED_RAW" --model "$SCORING_MODEL" --workers "$WORKERS" --max-records "$WILDCHAT_SCORING_TARGET_RECORDS" --scoring-preset "$SCORING_PRESET" --invalid-observation-retries "$INVALID_OBSERVATION_RETRIES" --fallback-on-errors
  fi
  python3 -m tools.mathdial_pipeline_support enrich-score --input "$SCORED_RAW" --output "$SCORED"
  if [[ "$DRY_RUN" != "1" ]]; then
    if ! python3 - "$SCORED" "$SELECTION_POOL_COUNT" "$MAX_SCORING_FALLBACK_RATE" <<'PY'
import json,pathlib,sys
rows=[json.loads(line) for line in pathlib.Path(sys.argv[1]).open(encoding="utf-8") if line.strip()]
actual=len(rows)
required = int(sys.argv[2])
print(f"[score_wildchat] selection pool coverage={actual}/{required}")
if actual < required:
    raise SystemExit(f"WildChatスコア済み候補が選別目標に不足しています: {actual}/{required}")
fallback=sum(bool(row.get("llm_error")) for row in rows)
rate=fallback/max(1,actual)
print(f"[score_wildchat] fallback={fallback}/{actual} rate={rate:.4%}")
if rate > float(sys.argv[3]):
    raise SystemExit(f"scoring fallback率が上限を超えました: {rate:.4%} > {float(sys.argv[3]):.4%}")
PY
    then
      return 20
    fi
  fi
}

select_data_stage() {
  # DPO閾値落ちを見込み、最終採用数より大きい同数の比較候補プールを作る。
  local count="$SELECTION_POOL_COUNT" random_count="$SELECTION_POOL_COUNT"
  [[ "$DRY_RUN" == "1" ]] && { count=4; random_count=6; }
  python3 -m tools.mathdial_selection --scored "$SCORED" --mathdial-conversations "$MATH_CONV" --bayes-model "$COMPAT_MODEL" --output-dir "$SELECT_DIR" --count "$count" --random-count "$random_count" --seed "$SEED" --label-derivation-method "$SELECTION_LABEL_METHOD" --selection-margin "$SELECTION_EMISSION_MARGIN" || return 20
}

build_dpo_stage() {
  mkdir -p "$DPO_DIR"
  local basis_count=2000 gold_count=500 random_count=2500 gold_source=1000
  [[ "$DRY_RUN" == "1" ]] && { basis_count=4; gold_count=2; random_count=6; gold_source=3; }
  python3 -m tools.prepare_mathdial_gold --samples "$MATH_SAMPLES" --conversations "$MATH_CONV" --output "$DPO_DIR/gold_candidates_en.jsonl" --target "$gold_source" --seed "$SEED"
  if [[ "$DRY_RUN" == "1" ]]; then
    python3 -m tools.mathdial_pipeline_support mock-dpo --input "$SELECT_DIR/basis_top.jsonl" --output "$DPO_DIR/basis_selected_ja.jsonl" --count "$basis_count" --source-dataset WildChat-BASiS
    python3 -m tools.mathdial_pipeline_support mock-dpo --input "$DPO_DIR/gold_candidates_en.jsonl" --output "$DPO_DIR/mathdial_gold_ja.jsonl" --count "$gold_count" --source-dataset MathDial --gold
    python3 -m tools.mathdial_pipeline_support mock-dpo --input "$SELECT_DIR/domain_random.jsonl" --output "$DPO_DIR/random_ja.jsonl" --count "$random_count" --source-dataset WildChat-Random
  else
    retry_command python3 -m tools.translate_and_generate_dpo --input "$SELECT_DIR/basis_top.jsonl" --bayes-model "$COMPAT_MODEL" --output "$DPO_DIR/basis_selected_ja.jsonl" --model "$GENERATION_MODEL" --score-model "$SCORING_MODEL" --style-preset mathdial_tutoring --candidates 8 --min-score-gap 0.20 --min-chosen-posterior 0.70 --max-rejected-posterior 0.55 --target-records "$basis_count" --workers "$WORKERS" --skip-sample-errors --heartbeat-file "$HEARTBEAT_FILE" --heartbeat-stage-prefix basis_dpo --seed "$SEED"
    retry_command python3 -m tools.translate_and_generate_dpo --input "$DPO_DIR/gold_candidates_en.jsonl" --bayes-model "$COMPAT_MODEL" --output "$DPO_DIR/mathdial_gold_ja.jsonl" --model "$GENERATION_MODEL" --score-model "$SCORING_MODEL" --style-preset mathdial_tutoring --candidates 8 --min-score-gap 0.20 --min-chosen-posterior 0.70 --max-rejected-posterior 0.55 --target-records "$gold_count" --workers "$WORKERS" --skip-sample-errors --heartbeat-file "$HEARTBEAT_FILE" --heartbeat-stage-prefix gold_dpo --seed "$SEED"
    retry_command python3 -m tools.build_random_dailydialog_dpo --input "$SELECT_DIR/domain_random.jsonl" --source-dataset WildChat --prompt-preset mathdial_tutoring --output "$DPO_DIR/random_ja.jsonl" --daily-output "$DPO_DIR/random_ja.jsonl" --target-records "$random_count" --candidates 8 --model "$GENERATION_MODEL" --workers "$WORKERS" --skip-sample-errors --heartbeat-file "$HEARTBEAT_FILE" --seed "$SEED"
  fi
  python3 -m tools.mix_mathdial_dpo --basis "$DPO_DIR/basis_selected_ja.jsonl" --gold "$DPO_DIR/mathdial_gold_ja.jsonl" --random "$DPO_DIR/random_ja.jsonl" --basis-output "$DPO_DIR/mathdial_basis_train.jsonl" --random-output "$DPO_DIR/mathdial_random_train.jsonl" --basis-count "$basis_count" --gold-count "$gold_count" --random-count "$random_count" || return 20
}

train_stage() {
  mkdir -p "$TRAIN_DIR"
  local dry=()
  [[ "$DRY_RUN" == "1" ]] && dry+=(--dry-run)
  COMMAND_MAX_ATTEMPTS=2 retry_command env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python3 -m tools.train_qwen35_dpo_lora --dataset "$DPO_DIR/mathdial_basis_train.jsonl" --model-id "$LOCAL_MODEL" --output-dir "$TRAIN_DIR/basis_lora" --num-train-epochs 1 --learning-rate 5e-6 --beta 0.1 --per-device-train-batch-size 1 --gradient-accumulation-steps 8 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --save-steps 25 --warmup-ratio 0.03 --eval-ratio 0 --seed "$SEED" --no-4bit --resume-from-checkpoint auto "${dry[@]}" || return 20
  COMMAND_MAX_ATTEMPTS=2 retry_command env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python3 -m tools.train_qwen35_dpo_lora --dataset "$DPO_DIR/mathdial_random_train.jsonl" --model-id "$LOCAL_MODEL" --output-dir "$TRAIN_DIR/random_lora" --num-train-epochs 1 --learning-rate 5e-6 --beta 0.1 --per-device-train-batch-size 1 --gradient-accumulation-steps 8 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --save-steps 25 --warmup-ratio 0.03 --eval-ratio 0 --seed "$SEED" --no-4bit --resume-from-checkpoint auto "${dry[@]}" || return 20
}

generate_responses_stage() {
  local count=100 mock=()
  [[ "$DRY_RUN" == "1" ]] && { count=5; mock+=(--mock); }
  python3 -m tools.mathdial_evaluation prepare --samples "$MATH_SAMPLES" --conversations "$MATH_CONV" --output "$EVAL_DIR/prompts_ja.jsonl" --errors-output "$EVAL_DIR/translation_errors.jsonl" --skip-sample-errors --count "$count" --seed "$SEED" --model "$SCORING_MODEL" --resume "${mock[@]}"
  python3 -m tools.mathdial_evaluation generate --input "$EVAL_DIR/prompts_ja.jsonl" --output "$EVAL_DIR/responses.jsonl" --errors-output "$EVAL_DIR/generation_errors.jsonl" --skip-sample-errors --oracle-output "$EVAL_DIR/oracle_input.jsonl" --base-model "$LOCAL_MODEL" --basis-lora "$TRAIN_DIR/basis_lora" --random-lora "$TRAIN_DIR/random_lora" --seed "$SEED" "${mock[@]}"
  python3 - "$EVAL_DIR/prompts_ja.jsonl" "$EVAL_DIR/responses.jsonl" "$EVAL_DIR/oracle_input.jsonl" "$count" <<'PY' || return 1
import json,pathlib,sys
def rows(path): return [json.loads(line) for line in pathlib.Path(path).open(encoding="utf-8") if line.strip()]
prompts,responses,oracle=map(rows,sys.argv[1:4]); required=int(sys.argv[4])
if len(prompts)!=required or len(responses)!=required or len(oracle)!=required*3:
 raise SystemExit(f"評価応答が不足しています: prompts={len(prompts)} responses={len(responses)} oracle={len(oracle)} required={required}")
if any(not all(str(row.get(key, '')).strip() for key in ('base_response','basis_response','random_dpo_response')) for row in responses):
 raise SystemExit("3モデルのいずれかに空応答があります。")
PY
}

oracle_eval_stage() {
  local dry=()
  [[ "$DRY_RUN" == "1" ]] && dry+=(--dry-run)
  python3 scripts/eval_oracle_mathdial.py --input "$EVAL_DIR/oracle_input.jsonl" --output_dir "$EVAL_DIR/oracle/pedagogical" --category pedagogical --judge_model "$JUDGE_MODEL" --score-scale 10 --oracle-workers "$WORKERS" --resume "${dry[@]}"
  python3 scripts/eval_oracle_mathdial.py --input "$EVAL_DIR/oracle_input.jsonl" --output_dir "$EVAL_DIR/oracle/general" --category general --judge_model "$JUDGE_MODEL" --score-scale 10 --oracle-workers "$WORKERS" --resume "${dry[@]}"
  local category_dir
  for category_dir in "$EVAL_DIR/oracle/pedagogical" "$EVAL_DIR/oracle/general"; do
    cp "$category_dir/summary.csv" "$category_dir/model_summary.csv"
    cp "$category_dir/summary.csv" "$category_dir/axis_summary.csv"
    [[ -f "$category_dir/errors.jsonl" ]] || : > "$category_dir/errors.jsonl"
  done
  local required=300
  [[ "$DRY_RUN" == "1" ]] && required=15
  python3 - "$EVAL_DIR/oracle/pedagogical/raw.jsonl" "$EVAL_DIR/oracle/general/raw.jsonl" "$required" <<'PY' || return 1
import pathlib,sys
required=int(sys.argv[3])
for value in sys.argv[1:3]:
 path=pathlib.Path(value); actual=sum(bool(line.strip()) for line in path.open(encoding="utf-8"))
 if actual != required: raise SystemExit(f"Oracle評価件数が不足しています: {path} {actual}/{required}")
PY
}

statistics_stage() {
  local permutations=10000 bootstrap=2000
  [[ "$DRY_RUN" == "1" ]] && { permutations=100; bootstrap=100; }
  python3 scripts/run_mathdial_statistics.py --raw "$EVAL_DIR/oracle/pedagogical/raw.jsonl" --raw "$EVAL_DIR/oracle/general/raw.jsonl" --output-dir "$EVAL_DIR/statistics" --permutations "$permutations" --bootstrap "$bootstrap" --seed "$SEED"
}

report_stage() {
  python3 -m tools.mathdial_pipeline_support report --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/reports/final_report.md"
}

run_stage preprocess preprocess_stage
run_stage build_basis build_basis_stage
run_stage extract_wildchat extract_wildchat_stage
run_stage score_wildchat score_wildchat_stage
run_stage select_data select_data_stage
run_stage build_dpo build_dpo_stage
run_stage train train_stage
run_stage generate_responses generate_responses_stage
run_stage oracle_eval oracle_eval_stage
run_stage statistics statistics_stage
run_stage report report_stage

echo "MathDial pipeline completed: $OUTPUT_ROOT"
echo "Log: $LOG_FILE"
PIPELINE_SUCCEEDED=1
write_runtime_state "success" "completed" "pipeline completed"
