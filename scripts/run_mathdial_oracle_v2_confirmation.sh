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

SOURCE_RUN="${SOURCE_RUN:-artifacts/mathdial_wildchat/runs/mathdial_wildchat_gpt56_v6_candidates4_mixed}"
RUN_TAG="${RUN_TAG:-mathdial_oracle_v2_confirmation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/mathdial_wildchat/evaluation_rechecks/${RUN_TAG}}"
EVAL_COUNT="${EVAL_COUNT:-100}"
SEED="${SEED:-20260717}"
WORKERS="${WORKERS:-4}"
TRANSLATION_MODEL="${MATHDIAL_SCORING_MODEL:-${AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT:-gpt-5.6-terra}}"
JUDGE_MODEL="${MATHDIAL_JUDGE_MODEL:-${AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT:-gpt-5.6-terra}}"
LOCAL_MODEL="${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1}"
EVAL_MAX_MEMORY="${EVAL_MAX_MEMORY:-0=46GiB,1=46GiB,cpu=0GiB}"

SAMPLES="$SOURCE_RUN/mathdial/data/mathdial_assistant_samples.jsonl"
CONVERSATIONS="$SOURCE_RUN/mathdial/data/mathdial_conversations.jsonl"
PREVIOUS_PROMPTS="$SOURCE_RUN/evaluation/prompts_ja.jsonl"
BASIS_LORA="$SOURCE_RUN/training/basis_lora"
RANDOM_LORA="$SOURCE_RUN/training/random_lora"

PROMPTS="$OUTPUT_ROOT/prompts_ja.jsonl"
RESPONSES="$OUTPUT_ROOT/responses.jsonl"
ORACLE_INPUT="$OUTPUT_ROOT/oracle_input.jsonl"
PEDAGOGICAL_DIR="$OUTPUT_ROOT/oracle/pedagogical_v2"
GENERAL_DIR="$OUTPUT_ROOT/oracle/general"
STATISTICS_DIR="$OUTPUT_ROOT/statistics"

for required in "$SAMPLES" "$CONVERSATIONS" "$PREVIOUS_PROMPTS" \
  "$BASIS_LORA/adapter_model.safetensors" "$RANDOM_LORA/adapter_model.safetensors"; do
  if [[ ! -e "$required" ]]; then
    echo "必要な入力がありません: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT/logs"
LOG="$OUTPUT_ROOT/logs/oracle_v2_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG") 2>&1

echo "MathDial Oracle v2 confirmation started at $(date --iso-8601=seconds)"
echo "source_run: $SOURCE_RUN"
echo "output_root: $OUTPUT_ROOT"
echo "evaluation_count: $EVAL_COUNT"
echo "seed: $SEED"
echo "translation_model: $TRANSLATION_MODEL"
echo "judge_model: $JUDGE_MODEL"

echo "[START] prepare"
python3 -m tools.mathdial_evaluation prepare \
  --samples "$SAMPLES" \
  --conversations "$CONVERSATIONS" \
  --output "$PROMPTS" \
  --count "$EVAL_COUNT" \
  --seed "$SEED" \
  --model "$TRANSLATION_MODEL" \
  --exclude-prompts "$PREVIOUS_PROMPTS" \
  --stratify-teacher-moves \
  --prompt-id-prefix "mathdial_eval_v2_confirm" \
  --resume

python3 - "$PROMPTS" "$PREVIOUS_PROMPTS" "$EVAL_COUNT" <<'PY'
import json
import sys
from pathlib import Path

current_path, previous_path, required = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
current = [json.loads(line) for line in current_path.open(encoding="utf-8") if line.strip()]
previous = [json.loads(line) for line in previous_path.open(encoding="utf-8") if line.strip()]
if len(current) != required:
    raise SystemExit(f"v2確認promptが不足しています: {len(current)}/{required}")
current_samples = {str(row["sample_id"]) for row in current}
current_qids = {str(row["qid"]) for row in current}
previous_samples = {str(row["sample_id"]) for row in previous}
previous_qids = {str(row["qid"]) for row in previous}
if current_samples & previous_samples or current_qids & previous_qids:
    raise SystemExit("v1とv2の評価promptにsample idまたはqidの重複があります。")
if len(current_samples) != required or len(current_qids) != required:
    raise SystemExit("v2確認prompt内でsample idまたはqidが重複しています。")
PY
echo "[DONE] prepare"

echo "[START] generate"
env \
  CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" \
  DPO_COMPARE_MAX_MEMORY="$EVAL_MAX_MEMORY" \
  python3 -m tools.mathdial_evaluation generate \
    --input "$PROMPTS" \
    --output "$RESPONSES" \
    --oracle-output "$ORACLE_INPUT" \
    --base-model "$LOCAL_MODEL" \
    --basis-lora "$BASIS_LORA" \
    --random-lora "$RANDOM_LORA" \
    --seed "$SEED"

python3 - "$RESPONSES" "$ORACLE_INPUT" "$EVAL_COUNT" <<'PY'
import json
import sys
from pathlib import Path

responses_path, oracle_path, required = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
responses = [json.loads(line) for line in responses_path.open(encoding="utf-8") if line.strip()]
oracle = [json.loads(line) for line in oracle_path.open(encoding="utf-8") if line.strip()]
if len(responses) != required or len(oracle) != required * 3:
    raise SystemExit(
        f"評価応答が不足しています: responses={len(responses)} "
        f"oracle={len(oracle)} required={required}"
    )
PY
echo "[DONE] generate"

echo "[START] oracle"
python3 scripts/eval_oracle_mathdial_v2.py \
  --input "$ORACLE_INPUT" \
  --output_dir "$PEDAGOGICAL_DIR" \
  --category pedagogical \
  --judge_model "$JUDGE_MODEL" \
  --score-scale 10 \
  --oracle-workers "$WORKERS" \
  --resume

python3 scripts/eval_oracle_mathdial_v2.py \
  --input "$ORACLE_INPUT" \
  --output_dir "$GENERAL_DIR" \
  --category general \
  --judge_model "$JUDGE_MODEL" \
  --score-scale 10 \
  --oracle-workers "$WORKERS" \
  --resume
echo "[DONE] oracle"

echo "[START] statistics"
python3 scripts/run_mathdial_statistics.py \
  --raw "$PEDAGOGICAL_DIR/raw.jsonl" \
  --raw "$GENERAL_DIR/raw.jsonl" \
  --output-dir "$STATISTICS_DIR" \
  --permutations 10000 \
  --bootstrap 2000 \
  --seed "$SEED"
echo "[DONE] statistics"

echo "[START] report"
python3 - "$OUTPUT_ROOT" "$SOURCE_RUN" "$EVAL_COUNT" "$SEED" "$TRANSLATION_MODEL" "$JUDGE_MODEL" <<'PY'
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

root, source = Path(sys.argv[1]), Path(sys.argv[2])
count, seed = int(sys.argv[3]), int(sys.argv[4])
translation_model, judge_model = sys.argv[5], sys.argv[6]

def rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

prompts = [
    json.loads(line)
    for line in (root / "prompts_ja.jsonl").open(encoding="utf-8")
    if line.strip()
]
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "confirmatory_axes_frozen_before_scoring",
    "source_run": str(source),
    "evaluation_count": count,
    "seed": seed,
    "translation_model": translation_model,
    "judge_model": judge_model,
    "prompt_overlap_with_v1": 0,
    "source_teacher_move_counts": dict(
        Counter(
            move
            for row in prompts
            for move in row.get("source_teacher_moves", [])
        )
    ),
    "config_sha256": sha256(Path("configs/evaluations/mathdial_oracle_v2.yaml")),
    "evaluator_sha256": sha256(Path("scripts/eval_oracle_mathdial_v2.py")),
    "axes_document_sha256": sha256(Path("docs/MATHDIAL_EVALUATION_AXES_V2.md")),
}
(root / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

summary = rows(root / "statistics/model_summary.csv")
omnibus = rows(root / "statistics/omnibus_friedman.csv")
posthoc = rows(root / "statistics/posthoc_pairwise.csv")
lines = [
    "# MathDial Oracle v2 confirmation",
    "",
    "v1で未使用のtest qidだけを使った独立確認評価。軸は採点前に固定した。",
    "",
    "## Manifest",
    "",
    "```json",
    json.dumps(manifest, ensure_ascii=False, indent=2),
    "```",
    "",
    "## Model summary",
    "",
    "```csv",
]
if summary:
    lines.append(",".join(summary[0]))
    lines.extend(",".join(str(row[key]) for key in summary[0]) for row in summary)
lines.extend(["```", "", "## Friedman", "", "```csv"])
if omnibus:
    lines.append(",".join(omnibus[0]))
    lines.extend(",".join(str(row[key]) for key in omnibus[0]) for row in omnibus)
lines.extend(["```", "", "## Holm-adjusted post-hoc", "", "```csv"])
if posthoc:
    lines.append(",".join(posthoc[0]))
    lines.extend(",".join(str(row[key]) for key in posthoc[0]) for row in posthoc)
lines.extend(["```", ""])
(root / "report.md").write_text("\n".join(lines), encoding="utf-8")
PY
echo "[DONE] report"

echo "MathDial Oracle v2 confirmation completed at $(date --iso-8601=seconds)"
echo "Report: $OUTPUT_ROOT/report.md"
echo "Log: $LOG"
