#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

FORM_ROOT="${FORM_ROOT:-artifacts/user_eval/google_forms/esconv_human_reviewed_likert_two_forms_v7}"
DATABASE="${DATABASE:-artifacts/user_eval/web/esconv_likert_responses.sqlite3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8503}"
PUBLIC_HOST="${PUBLIC_HOST:-}"

if [[ ! -s "$FORM_ROOT/experiment_a/form_items_public.jsonl" ]]; then
  python3 -m scripts.prepare_esconv_google_form_likert_blocks \
    --output-dir "$FORM_ROOT"
fi

SINGLE_FORM=0
if [[ ! -s "$FORM_ROOT/experiment_b/form_items_public.jsonl" ]]; then
  SINGLE_FORM=1
fi

mkdir -p "$(dirname "$DATABASE")"

if [[ -z "$PUBLIC_HOST" ]]; then
  PUBLIC_HOST="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
fi
PUBLIC_HOST="${PUBLIC_HOST:-localhost}"
if [[ "$SINGLE_FORM" == "1" ]]; then
  printf '[survey] 全員共通: http://%s:%s/\n' "$PUBLIC_HOST" "$PORT"
else
  printf '[survey] 実験A: http://%s:%s/?experiment=A\n' "$PUBLIC_HOST" "$PORT"
  printf '[survey] 実験B: http://%s:%s/?experiment=B\n' "$PUBLIC_HOST" "$PORT"
fi

exec python3 -m streamlit run apps/esconv_likert_user_eval.py \
  --server.address "$HOST" \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false \
  -- \
  --form-root "$FORM_ROOT" \
  --database "$DATABASE"
