#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
[[ -f .env ]] && { set -a; source .env; set +a; }

DATASET="${DATASET:?DATASETにはmathdialまたはmeditodを指定してください}"
case "$DATASET" in mathdial|meditod) ;; *) echo "DATASETはmathdialまたはmeditodです。" >&2; exit 2;; esac
DEFINITION="${DEFINITION:-configs/user_evaluations/${DATASET}_likert_v1.yaml}"
FORM_ROOT="${FORM_ROOT:?FORM_ROOTにprepare済み人手評価ディレクトリを指定してください}"
DATABASE="${DATABASE:-artifacts/user_eval/web/${DATASET}_likert_responses.sqlite3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8504}"
PUBLIC_HOST="${PUBLIC_HOST:-}"
[[ -s "$FORM_ROOT/experiment_a/form_items_public.jsonl" && -s "$FORM_ROOT/experiment_b/form_items_public.jsonl" && -s "$FORM_ROOT/manifest.json" ]] || { echo "人手評価公開itemが不足しています: $FORM_ROOT" >&2; exit 2; }
mkdir -p "$(dirname "$DATABASE")"
if [[ -z "$PUBLIC_HOST" ]]; then PUBLIC_HOST="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"; fi
PUBLIC_HOST="${PUBLIC_HOST:-localhost}"
printf '[survey:%s] 実験A: http://%s:%s/?experiment=A\n' "$DATASET" "$PUBLIC_HOST" "$PORT"
printf '[survey:%s] 実験B: http://%s:%s/?experiment=B\n' "$DATASET" "$PUBLIC_HOST" "$PORT"
exec python3 -m streamlit run apps/three_model_likert_user_eval.py \
  --server.address "$HOST" --server.port "$PORT" --server.headless true \
  --browser.gatherUsageStats false -- \
  --definition "$DEFINITION" --form-root "$FORM_ROOT" --database "$DATABASE"
