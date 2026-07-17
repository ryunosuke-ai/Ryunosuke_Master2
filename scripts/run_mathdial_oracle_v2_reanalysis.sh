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
RUN_TAG="${RUN_TAG:-mathdial_oracle_v2_v1_prompts_reanalysis}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/mathdial_wildchat/evaluation_rechecks/${RUN_TAG}}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"
JUDGE_MODEL="${MATHDIAL_JUDGE_MODEL:-${AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT:-gpt-5.6-terra}}"

ORACLE_INPUT="$SOURCE_RUN/evaluation/oracle_input.jsonl"
V1_PEDAGOGICAL_RAW="$SOURCE_RUN/evaluation/oracle/pedagogical/raw.jsonl"
V1_GENERAL_RAW="$SOURCE_RUN/evaluation/oracle/general/raw.jsonl"
PEDAGOGICAL_DIR="$OUTPUT_ROOT/oracle/pedagogical_v2"
STATISTICS_DIR="$OUTPUT_ROOT/statistics"
CONDITIONS="$OUTPUT_ROOT/run_conditions.json"

for required in "$ORACLE_INPUT" "$V1_PEDAGOGICAL_RAW" "$V1_GENERAL_RAW"; do
  if [[ ! -f "$required" ]]; then
    echo "必要なv1評価成果物がありません: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT/logs"

python3 - "$CONDITIONS" "$SOURCE_RUN" "$ORACLE_INPUT" "$JUDGE_MODEL" "$SEED" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

conditions_path = Path(sys.argv[1])
source_run = Path(sys.argv[2])
oracle_input = Path(sys.argv[3])
judge_model = sys.argv[4]
seed = int(sys.argv[5])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

conditions = {
    "evaluation_status": "post_hoc_reanalysis_on_v1_prompts_and_responses",
    "source_run": str(source_run),
    "oracle_input": str(oracle_input),
    "oracle_input_sha256": sha256(oracle_input),
    "judge_model": judge_model,
    "seed": seed,
    "evaluation_config_sha256": sha256(
        Path("configs/evaluations/mathdial_oracle_v2.yaml")
    ),
    "evaluator_sha256": sha256(Path("scripts/eval_oracle_mathdial_v2.py")),
}
if conditions_path.exists():
    existing = json.loads(conditions_path.read_text(encoding="utf-8"))
    if existing != conditions:
        raise SystemExit(
            "同じOUTPUT_ROOTの評価条件が変わっています。新しいRUN_TAGを使用してください。"
        )
else:
    conditions_path.write_text(
        json.dumps(conditions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
PY

LOG="$OUTPUT_ROOT/logs/oracle_v2_reanalysis_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG") 2>&1

echo "MathDial Oracle v2 reanalysis started at $(date --iso-8601=seconds)"
echo "source_run: $SOURCE_RUN"
echo "output_root: $OUTPUT_ROOT"
echo "judge_model: $JUDGE_MODEL"
echo "注記: v1と同じprompt・3モデル応答を使う事後的再評価です。"

echo "[START] validate_v1_input"
python3 - "$ORACLE_INPUT" "$V1_GENERAL_RAW" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

oracle = [
    json.loads(line)
    for line in Path(sys.argv[1]).open(encoding="utf-8")
    if line.strip()
]
general = [
    json.loads(line)
    for line in Path(sys.argv[2]).open(encoding="utf-8")
    if line.strip()
]
if len(oracle) != 300:
    raise SystemExit(f"v1 Oracle入力が300件ではありません: {len(oracle)}")
if len(general) != 300:
    raise SystemExit(f"v1一般品質rawが300件ではありません: {len(general)}")
models = Counter(str(row.get("model_name")) for row in oracle)
if models != Counter({"base": 100, "basis": 100, "random_dpo": 100}):
    raise SystemExit(f"v1 Oracle入力のモデル件数が不正です: {dict(models)}")
samples = {str(row.get("sample_id")) for row in oracle}
if len(samples) != 100:
    raise SystemExit(f"v1 Oracle入力のprompt数が100件ではありません: {len(samples)}")
PY
echo "[DONE] validate_v1_input"

echo "[START] oracle_v2_pedagogical"
python3 scripts/eval_oracle_mathdial_v2.py \
  --input "$ORACLE_INPUT" \
  --output_dir "$PEDAGOGICAL_DIR" \
  --category pedagogical \
  --judge_model "$JUDGE_MODEL" \
  --score-scale 10 \
  --oracle-workers "$WORKERS" \
  --resume \
  --seed "$SEED"
echo "[DONE] oracle_v2_pedagogical"

echo "[START] statistics"
python3 scripts/run_mathdial_statistics.py \
  --raw "$PEDAGOGICAL_DIR/raw.jsonl" \
  --raw "$V1_GENERAL_RAW" \
  --output-dir "$STATISTICS_DIR" \
  --permutations 10000 \
  --bootstrap 2000 \
  --seed "$SEED"
echo "[DONE] statistics"

echo "[START] report"
python3 - "$OUTPUT_ROOT" "$SOURCE_RUN" "$V1_PEDAGOGICAL_RAW" "$V1_GENERAL_RAW" <<'PY'
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
source_run = Path(sys.argv[2])
v1_pedagogical = Path(sys.argv[3])
v1_general = Path(sys.argv[4])

def csv_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()

conditions = json.loads((root / "run_conditions.json").read_text(encoding="utf-8"))
manifest = {
    **conditions,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "v1_pedagogical_history": str(v1_pedagogical),
    "general_quality_reused_without_rejudging": str(v1_general),
    "v2_pedagogical_raw": str(root / "oracle/pedagogical_v2/raw.jsonl"),
    "interpretation": (
        "v1結果を確認後に定義したv2軸を同じ100 prompt・同じ応答へ適用した再分析。"
        "独立hold-out確認とは区別する。"
    ),
}
(root / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

sections = [
    "# MathDial Oracle v2: v1 prompt/response reanalysis",
    "",
    "v1と同じ100 prompt・300応答を、v2教育軸だけで再採点した結果。",
    "v1教育評価は履歴として保持し、一般品質rawは変更していない。",
    "",
    "## Manifest",
    "",
    "```json",
    json.dumps(manifest, ensure_ascii=False, indent=2),
    "```",
]
for title, relative in (
    ("Model summary", "statistics/model_summary.csv"),
    ("Friedman", "statistics/omnibus_friedman.csv"),
    ("Holm-adjusted post-hoc", "statistics/posthoc_pairwise.csv"),
):
    sections.extend(
        [
            "",
            f"## {title}",
            "",
            "```csv",
            csv_text(root / relative),
            "```",
        ]
    )
(root / "report.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
PY
echo "[DONE] report"

echo "MathDial Oracle v2 reanalysis completed at $(date --iso-8601=seconds)"
echo "Report: $OUTPUT_ROOT/report.md"
echo "Log: $LOG"
