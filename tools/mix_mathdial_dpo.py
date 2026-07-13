"""MathDial BASiS/Randomの学習件数とsource構成を厳密に確定する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def validate_pair(row: dict[str, Any]) -> None:
    for key in ("prompt", "chosen", "rejected", "metadata"):
        if key not in row:
            raise ValueError(f"DPO recordに`{key}`がありません。")
    if not str(row["prompt"]).strip() or not str(row["chosen"]).strip() or not str(row["rejected"]).strip():
        raise ValueError("DPO prompt/chosen/rejectedが空です。")
    if row["chosen"] == row["rejected"]:
        raise ValueError("chosenとrejectedが同一です。")
    metadata = row["metadata"]
    if metadata.get("translated_prompt_hash") and metadata.get("rejected_prompt_hash") not in (None, metadata["translated_prompt_hash"]):
        raise ValueError("chosen/rejectedの日本語context hashが一致しません。")


def build_training_arms(basis: list[dict[str, Any]], gold: list[dict[str, Any]], random_rows: list[dict[str, Any]], *, basis_count: int = 2000, gold_count: int = 500, random_count: int = 2500) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for row in basis + gold + random_rows:
        validate_pair(row)
    if len(basis) < basis_count or len(gold) < gold_count or len(random_rows) < random_count:
        raise ValueError(f"DPO source不足: basis={len(basis)}, gold={len(gold)}, random={len(random_rows)}")
    basis_arm = basis[:basis_count] + gold[:gold_count]
    random_arm = random_rows[:random_count]
    if len(basis_arm) != len(random_arm) or len(basis_arm) != basis_count + gold_count or len(random_arm) != random_count:
        raise ValueError("BASiS/Randomの総学習件数が一致しません。")
    if any(row.get("metadata", {}).get("gold") or row.get("source_dataset") == "MathDial" for row in random_arm):
        raise ValueError("Random-DPOへMathDial goldが混入しています。")
    return basis_arm, random_arm


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="MathDial DPO学習armを構築")
    parser.add_argument("--basis", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--random", required=True)
    parser.add_argument("--basis-output", required=True)
    parser.add_argument("--random-output", required=True)
    parser.add_argument("--basis-count", type=int, default=2000)
    parser.add_argument("--gold-count", type=int, default=500)
    parser.add_argument("--random-count", type=int, default=2500)
    args = parser.parse_args()
    basis, random_rows = build_training_arms(read_jsonl(args.basis), read_jsonl(args.gold), read_jsonl(args.random), basis_count=args.basis_count, gold_count=args.gold_count, random_count=args.random_count)
    write_jsonl(basis, Path(args.basis_output))
    write_jsonl(random_rows, Path(args.random_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
