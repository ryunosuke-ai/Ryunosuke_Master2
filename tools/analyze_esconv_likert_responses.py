"""ESConv 3モデルLikert人手評価を復号し、参加者単位で集計する。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from core.esconv_likert_survey import export_responses_csv
from tools.analyze_three_model_likert_responses import (
    analyze,
    load_ratings,
    write_csv,
)


def response_position(value: str) -> str:
    """`応答A`形式を匿名表示位置へ変換する。"""

    normalized = str(value).strip()
    if normalized not in {"応答A", "応答B", "応答C"}:
        raise ValueError(f"不正な応答位置です: {value!r}")
    return normalized[-1]


def load_private_answer_key(path: Path) -> dict[str, dict[str, str]]:
    """ESConvの非公開CSVを共通解析用mappingへ変換する。"""

    result: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for line_number, row in enumerate(csv.DictReader(file), start=2):
            item_id = str(row.get("item_id") or "").strip()
            if not item_id or item_id in result:
                raise ValueError(
                    f"{path}:{line_number}: item_idが空または重複しています。"
                )
            mapping = {
                response_position(row.get("basis_response_position") or ""): "basis",
                response_position(row.get("base_response_position") or ""): "base",
                response_position(row.get("random_response_position") or ""): "random_dpo",
            }
            if set(mapping) != {"A", "B", "C"} or len(mapping) != 3:
                raise ValueError(
                    f"{path}:{line_number}: A/B/Cのモデル配置が不正です。"
                )
            result[item_id] = mapping
    if not result:
        raise ValueError(f"非公開正解表が空です: {path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="ESConv 3モデル人手評価の統計")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--private-answer-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_responses_csv(
        args.database,
        args.output_dir / "responses_long_private.csv",
    )
    values, choices = load_ratings(
        args.database,
        load_private_answer_key(args.private_answer_key),
    )
    summary, omnibus, posthoc = analyze(
        values,
        permutations=args.permutations,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    write_csv(args.output_dir / "axis_model_summary.csv", summary)
    write_csv(args.output_dir / "friedman.csv", omnibus)
    write_csv(args.output_dir / "holm_posthoc.csv", posthoc)
    choice_total = sum(choices.values())
    write_csv(
        args.output_dir / "final_choice_counts.csv",
        [
            {
                "choice": key,
                "count": value,
                "win_rate": value / choice_total if choice_total else 0.0,
            }
            for key, value in sorted(choices.items())
        ],
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "esconv",
                "analysis_unit": "participant",
                "permutations": args.permutations,
                "bootstrap": args.bootstrap,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"ESConv人手評価の集計を書き出しました: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
