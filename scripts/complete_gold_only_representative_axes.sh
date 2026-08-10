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

RUN_ROOT="${RUN_ROOT:-artifacts/gold_only_dpo/runs/gold_only_dpo500_v1}"
CONFIG="${GOLD_ONLY_CONFIG:-configs/experiments/gold_only_dpo500_v1.yaml}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"
JUDGE_MODEL="${ESCONV_JUDGE_MODEL:-gpt-5.4}"

ESCONV_ROOT="$RUN_ROOT/esconv"
EVAL_DIR="$ESCONV_ROOT/evaluation"
GOLD_INPUT="$EVAL_DIR/oracle_input_gold.jsonl"
GOLD_TST_DIR="$EVAL_DIR/oracle_gold/main/text_style_transfer"
GOLD_TST_RAW="$GOLD_TST_DIR/raw.jsonl"
EXISTING_TST_RAW="artifacts/evaluations/oracle_eval_runs/esconv_topconf_three_model_gpt54_100_10pt_topconf_three_model_10pt/oracle_tst_10pt/raw.jsonl"
COMBINED_TST_RAW="$EVAL_DIR/oracle_combined/main/text_style_transfer/raw.jsonl"
STYLE_RAW="$EVAL_DIR/oracle_combined/main/conversation_style/raw.jsonl"
TRANSITION_RAW="$EVAL_DIR/oracle_combined/main/strategy_transition/raw.jsonl"

for required in "$CONFIG" "$GOLD_INPUT" "$EXISTING_TST_RAW" "$STYLE_RAW" "$TRANSITION_RAW"; do
  [[ -f "$required" ]] || { echo "必須入力がありません: $required" >&2; exit 20; }
done

echo "[START] ESConv Gold-only TST Oracle（成功済みsampleはresumeでskip）"
python3 -m scripts.eval_oracle_tst \
  --input "$GOLD_INPUT" \
  --output_dir "$GOLD_TST_DIR" \
  --judge_model "$JUDGE_MODEL" \
  --score-scale 10 \
  --oracle-workers "$WORKERS" \
  --resume

echo "[START] ESConv TST 4モデルraw統合"
python3 -m tools.gold_only_dpo --config "$CONFIG" merge-raw \
  --existing "$EXISTING_TST_RAW" \
  --gold "$GOLD_TST_RAW" \
  --output "$COMBINED_TST_RAW" \
  --manifest "${COMBINED_TST_RAW%.jsonl}.manifest.json" \
  --expected-samples 100

echo "[START] ESConv 4モデル統計再計算"
python3 -m scripts.run_gold_only_four_model_statistics \
  --raw "text_style_transfer=$COMBINED_TST_RAW" \
  --raw "conversation_style=$STYLE_RAW" \
  --raw "strategy_transition=$TRANSITION_RAW" \
  --output-dir "$ESCONV_ROOT/statistics" \
  --permutations 10000 \
  --bootstrap 2000 \
  --seed "$SEED"

echo "[START] 3データセット代表7軸ファイル再生成"
python3 -m scripts.export_gold_only_axis_results --root "$RUN_ROOT"

python3 - "$RUN_ROOT/all_datasets_axis_scores.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8"))
counts = {row["dataset"]: len(row["axes"]) for row in rows}
if counts != {"esconv": 7, "mathdial": 7, "meditod": 7}:
    raise SystemExit(f"代表軸件数が不正です: {counts}")
print(f"[DONE] 代表7軸が揃いました: {counts}")
PY

echo "Output: $RUN_ROOT/all_datasets_axis_scores.txt"
