"""戦略分布・遷移分布のOracle評価CLI。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.oracle_eval_common import add_strategy_cli_args, run_strategy_cli  # noqa: E402


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(description="ESConv戦略分布・遷移分布をLLM Oracleで評価します。")
    add_strategy_cli_args(
        parser,
        default_output_dir="artifacts/evaluations/oracle_eval_runs/oracle_strategy_transition",
    )
    return parser.parse_args()


def main() -> int:
    """CLIエントリポイント。"""
    return run_strategy_cli(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
