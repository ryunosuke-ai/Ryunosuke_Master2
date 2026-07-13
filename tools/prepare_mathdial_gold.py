"""MathDial train応答を既存DPO生成器へ渡すgold候補に変換する。"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def history_text(history: list[dict[str, str]], question: str) -> str:
    lines = [f"Problem: {question}"] if question else []
    lines.extend(f"{'User' if turn['role'] == 'user' else 'AI'}: {turn['text']}" for turn in history)
    return "\n".join(lines)


def collect_gold_candidates(samples: list[dict[str, Any]], conversations: list[dict[str, Any]], *, target: int, seed: int) -> list[dict[str, Any]]:
    conversations_by_id = {row["conversation_id"]: row for row in conversations}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        metadata = sample.get("metadata", {})
        if metadata.get("split") != "train" or not metadata.get("dpo_eligible"):
            continue
        moves = metadata.get("teacher_moves", []) or ["unlabeled"]
        groups[str(moves[0])].append(sample)
    rng = random.Random(seed)
    chosen: dict[str, dict[str, Any]] = {}
    per_group = max(1, target // max(1, len(groups)))
    for move in sorted(groups):
        values = list(groups[move])
        rng.shuffle(values)
        for sample in values[:per_group]:
            chosen[sample["sample_id"]] = sample
    remaining = [sample for values in groups.values() for sample in values if sample["sample_id"] not in chosen]
    rng.shuffle(remaining)
    for sample in remaining:
        if len(chosen) >= target:
            break
        chosen[sample["sample_id"]] = sample
    rows = []
    for sample in list(chosen.values())[:target]:
        conversation = conversations_by_id[sample["conversation_id"]]
        question = str(conversation.get("metadata", {}).get("question", ""))
        rows.append({
            "sample_id": sample["sample_id"],
            "conversation_id": sample["conversation_id"],
            "turn_index": sample["metadata"]["assistant_turn_index"],
            "prompt": history_text(sample["history"], question),
            "response": sample["response"],
            "history": sample["history"],
            "next_user_turn": sample.get("next_user_turn"),
            "metadata": {
                "source_dataset": "MathDial",
                "source_split": "train",
                "context_turns": len(sample["history"]),
                "teacher_moves": sample["metadata"].get("teacher_moves", []),
                "gold": True,
            },
        })
    if len(rows) != target:
        raise ValueError(f"MathDial gold候補が不足しています: target={target}, actual={len(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="MathDial gold候補を作成")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--conversations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = collect_gold_candidates(read_jsonl(args.samples), read_jsonl(args.conversations), target=args.target, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
