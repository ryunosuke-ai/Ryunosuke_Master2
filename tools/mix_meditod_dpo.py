"""MediTOD BASiS/Random学習armの件数とgold条件を監査する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def validate_pair(row: dict[str, Any]) -> None:
    if any(key not in row for key in ("prompt", "chosen", "rejected", "metadata")):
        raise ValueError("MediTOD DPO recordの必須キーが不足しています。")
    if any(not str(row[key]).strip() for key in ("prompt", "chosen", "rejected")):
        raise ValueError("MediTOD DPO prompt/chosen/rejectedが空です。")
    if row["chosen"] == row["rejected"]:
        raise ValueError("MediTOD DPO chosen/rejectedが同一です。")
    metadata = row["metadata"]
    translated = metadata.get("translated_prompt_hash")
    rejected = metadata.get("rejected_prompt_hash")
    if not translated or not rejected:
        raise ValueError("MediTOD DPOのcontext hashが不足しています。")
    if rejected != translated:
        raise ValueError("MediTOD DPO chosen/rejectedのcontext hashが一致しません。")
    normalized_hash = hashlib.sha256(str(row["prompt"]).encode()).hexdigest()
    metadata.setdefault("normalized_prompt_hash", normalized_hash)


def build_training_arms(
    basis: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    *,
    basis_count: int,
    gold_count: int,
    random_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for row in basis + gold + random_rows:
        validate_pair(row)
    if len(basis) < basis_count or len(gold) < gold_count or len(random_rows) < random_count:
        raise ValueError(f"MediTOD DPO source不足: basis={len(basis)}, gold={len(gold)}, random={len(random_rows)}")
    basis_arm = basis[:basis_count] + gold[:gold_count]
    random_arm = random_rows[:random_count]
    if len(basis_arm) != len(random_arm) or len(basis_arm) != basis_count + gold_count:
        raise ValueError("MediTOD BASiS/Randomの学習件数が一致しません。")
    if sum(bool(row.get("metadata", {}).get("gold")) for row in basis_arm) != gold_count:
        raise ValueError("MediTOD BASiS armのgold件数が不正です。")
    if any(row.get("metadata", {}).get("gold") or row.get("source_dataset") == "MediTOD" for row in random_arm):
        raise ValueError("MediTOD Random-DPOへgoldが混入しています。")
    return basis_arm, random_arm


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="MediTOD DPO学習armを構築")
    parser.add_argument("--basis", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--random", required=True)
    parser.add_argument("--basis-output", required=True)
    parser.add_argument("--random-output", required=True)
    parser.add_argument("--basis-count", type=int, default=2000)
    parser.add_argument("--gold-count", type=int, default=500)
    parser.add_argument("--random-count", type=int, default=2500)
    args = parser.parse_args()
    basis, random_rows = build_training_arms(
        read_jsonl(args.basis),
        read_jsonl(args.gold),
        read_jsonl(args.random),
        basis_count=args.basis_count,
        gold_count=args.gold_count,
        random_count=args.random_count,
    )
    write_jsonl(basis, Path(args.basis_output))
    write_jsonl(random_rows, Path(args.random_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
