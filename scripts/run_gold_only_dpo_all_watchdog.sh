#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
RUN_TAG="${RUN_TAG:-gold_only_dpo500_v1}"
START_DATASET="${START_DATASET:-esconv}"
END_DATASET="${END_DATASET:-meditod}"
DATASETS=(esconv mathdial meditod)

index_of() {
  local wanted="$1" i
  for i in "${!DATASETS[@]}"; do [[ "${DATASETS[$i]}" == "$wanted" ]] && { echo "$i"; return; }; done
  echo "未知のdataset: $wanted" >&2; exit 20
}
start="$(index_of "$START_DATASET")"; end="$(index_of "$END_DATASET")"
(( start <= end )) || { echo "START_DATASETはEND_DATASET以前にしてください。" >&2; exit 20; }

for i in "${!DATASETS[@]}"; do
  (( i < start || i > end )) && continue
  dataset="${DATASETS[$i]}"
  echo "===== Gold-only DPO: $dataset ====="
  DATASET="$dataset" RUN_TAG="$RUN_TAG" OUTPUT_ROOT="artifacts/gold_only_dpo/runs/$RUN_TAG/$dataset" \
    "$PROJECT_ROOT/scripts/run_gold_only_dpo_dataset_watchdog.sh"
done
echo "Gold-only DPO all datasets completed: $RUN_TAG"
