#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# 発表用の既存ESConv成果物は過去RUN_TAGを引き継いでいる。
# 名前にreminiscenceを含むが、実体はESConv支援対話スタイル学習実験。
RUN_TAG="${RUN_TAG:-reminiscence_5000_to_2000}"
DPO_WATCHDOG_SCRIPT="${DPO_WATCHDOG_SCRIPT:-${SCRIPT_DIR}/run_dpo_pipeline_esconv_2000_watchdog.sh}"
ORACLE_SCRIPT="${ORACLE_SCRIPT:-${SCRIPT_DIR}/run_oracle_evaluation_esconv.sh}"
AUDIT_LOG="${AUDIT_LOG:-audit_log.md}"
LOG_DIR="${END_TO_END_LOG_DIR:-logs/esconv_end_to_end}"
LOG_FILE="${LOG_DIR}/esconv_dpo_then_oracle_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

append_audit() {
  {
    echo
    echo "## $(date '+%Y-%m-%d %H:%M:%S %Z'): ESConv DPOからOracle評価までの一括実行"
    echo
    echo "- 対象ファイル:"
    echo "  - \`$DPO_WATCHDOG_SCRIPT\`"
    echo "  - \`$ORACLE_SCRIPT\`"
    echo "  - \`$LOG_FILE\`"
    echo "- 実行した操作:"
    echo "  - $1"
    echo "- なぜその操作が必要だったか:"
    echo "  - DPO学習後に同じRUN_TAGのLoRAを使ってOracle評価まで連続実行し、実験手順の抜けやパス間違いを防ぐため。"
    echo "- 代替案があったか:"
    echo "  - DPOとOracleを別々に手動実行する案があったが、長時間処理後の実行漏れや設定不一致が起きやすいため一括化した。"
    echo "- 実行したコマンド:"
    echo "  - \`$0\`"
    echo "- 変更前後の要約:"
    echo "  - RUN_TAG: $RUN_TAG"
    echo "  - DPO watchdog script: $DPO_WATCHDOG_SCRIPT"
    echo "  - Oracle script: $ORACLE_SCRIPT"
    echo "- リスクや注意点:"
    echo "  - DPOパイプラインが失敗した場合、Oracle評価には進まない。"
    echo "  - Oracle評価のLoRAパスはRUN_TAG由来の既定値または環境変数で指定された値を使う。"
  } >> "$AUDIT_LOG"
}

append_audit "ESConv DPOパイプラインとOracle評価の一括実行を開始した。"

echo "========================================"
echo "ESConv DPO then Oracle started at $(date)"
echo "run_tag: $RUN_TAG"
echo "dpo_watchdog_script: $DPO_WATCHDOG_SCRIPT"
echo "oracle_script: $ORACLE_SCRIPT"
echo "log_file: $LOG_FILE"
echo "========================================"

export RUN_TAG
"$DPO_WATCHDOG_SCRIPT"

echo "========================================"
echo "DPO pipeline completed; start Oracle evaluation at $(date)"
echo "========================================"

"$ORACLE_SCRIPT"

echo "========================================"
echo "ESConv DPO then Oracle completed at $(date)"
echo "========================================"
append_audit "ESConv DPOパイプラインとOracle評価の一括実行が正常終了した。"
