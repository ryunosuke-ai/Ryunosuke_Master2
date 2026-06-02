"""ベイズスコア済み対話から高posterior応答を抽出する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = "artifacts/scored_dialogues/dailydialog_transition_scored.jsonl"
DEFAULT_OUTPUT_PATH = "artifacts/datasets/dailydialog_selected_en.jsonl"
DEFAULT_MIN_POSTERIOR = 0.75


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="高posteriorの文脈付き応答を抽出します。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help=f"入力スコア済みJSONL（既定: {DEFAULT_INPUT_PATH}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"出力JSONL（既定: {DEFAULT_OUTPUT_PATH}）。")
    parser.add_argument("--min-posterior", type=float, default=DEFAULT_MIN_POSTERIOR, help="抽出するposteriorの下限。")
    parser.add_argument("--max-records", type=int, default=None, help="出力件数の上限。")
    parser.add_argument("--sort-by-posterior", action="store_true", help="posterior降順で抽出します。")
    parser.add_argument("--dry-run", action="store_true", help="書き出さず、抽出件数だけ表示します。")
    return parser.parse_args()


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """JSONLを読み込む。"""
    input_path = Path(path)
    records: list[dict[str, Any]] = []
    try:
        with input_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{line_number}行目をJSONとして読めません: {exc}") from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"入力JSONLが見つかりません: {input_path}") from exc
    return records


def _posterior(record: dict[str, Any]) -> float:
    """posteriorを数値として読む。"""
    value = record.get("posterior")
    if not isinstance(value, (int, float)):
        raise ValueError("`posterior` が数値でないレコードがあります。")
    return float(value)


def select_high_posterior_records(
    records: list[dict[str, Any]],
    *,
    min_posterior: float,
    max_records: int | None,
    sort_by_posterior: bool,
) -> list[dict[str, Any]]:
    """高posteriorレコードを抽出する。"""
    selected = [record for record in records if _posterior(record) >= min_posterior]
    if sort_by_posterior:
        selected = sorted(selected, key=_posterior, reverse=True)
    if max_records is not None:
        selected = selected[:max_records]
    return selected


def write_jsonl(records: list[dict[str, Any]], path: Path | str) -> None:
    """JSONLを書き出す。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    records = read_jsonl(args.input)
    selected = select_high_posterior_records(
        records,
        min_posterior=args.min_posterior,
        max_records=args.max_records,
        sort_by_posterior=args.sort_by_posterior,
    )
    if args.dry_run:
        print("高posterior抽出 dry-run")
        print(f"  input_records: {len(records)}")
        print(f"  selected_records: {len(selected)}")
        print(f"  min_posterior: {args.min_posterior}")
        return 0
    write_jsonl(selected, args.output)
    print(f"高posterior応答を書き出しました: {args.output} ({len(selected)} 件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
