#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/user_eval/google_forms/esconv_human_reviewed_likert_single10_v8}"
SEED="${SEED:-42}"

python3 -m scripts.prepare_esconv_google_form_likert_blocks \
  --single-form \
  --count 10 \
  --seed "$SEED" \
  --output-dir "$OUTPUT_ROOT"
