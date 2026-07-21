"""ESConv Webアンケート回答を研究者用CSVへ出力する。"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.esconv_likert_survey import export_responses_csv


DEFAULT_DATABASE = Path("artifacts/user_eval/web/esconv_likert_responses.sqlite3")
DEFAULT_OUTPUT = Path("artifacts/user_eval/web/esconv_likert_responses_long.csv")


def parse_args() -> argparse.Namespace:
    """CLI引数を読む。"""
    parser = argparse.ArgumentParser(
        description="ESConv Webアンケート回答をlong形式CSVへ出力します。"
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """回答CSVを出力する。"""
    args = parse_args()
    written = export_responses_csv(args.database, args.output)
    print(f"回答CSVを書き出しました: {args.output} ({written} ratings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
