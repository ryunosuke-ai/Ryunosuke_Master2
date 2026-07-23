"""MediTOD train医療者応答をgold DPO候補へ変換する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.wildchat_health import protected_medical_terms


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def history_text(history: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{'User' if turn['role'] == 'user' else 'AI'}: {turn['text']}" for turn in history
    )


def collect_gold_candidates(
    samples: list[dict[str, Any]],
    *,
    target: int,
    seed: int,
    allow_target_shortfall: bool = False,
    minimum_records: int | None = None,
) -> list[dict[str, Any]]:
    if target <= 0:
        raise ValueError("targetは1以上にしてください。")
    if minimum_records is not None and not 0 < minimum_records <= target:
        raise ValueError("minimum_recordsは1以上target以下にしてください。")
    eligible = [
        sample for sample in samples
        if sample.get("metadata", {}).get("split") == "train"
        and sample.get("metadata", {}).get("dpo_eligible")
        and not sample.get("metadata", {}).get("ood")
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in eligible:
        slots = sample["metadata"].get("response_slots") or ["unlabeled"]
        groups[str(slots[0])].append(sample)
    frequency = Counter(key for key, values in groups.items() for _ in values)
    ordered = sorted(
        eligible,
        key=lambda sample: (
            frequency[(sample["metadata"].get("response_slots") or ["unlabeled"])[0]],
            _rank(seed, sample["sample_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    per_conversation: Counter[str] = Counter()
    for sample in ordered:
        if per_conversation[sample["conversation_id"]] >= 6:
            continue
        selected.append(sample)
        per_conversation[sample["conversation_id"]] += 1
        if len(selected) >= target:
            break
    required = minimum_records if minimum_records is not None else target
    if len(selected) < target and (not allow_target_shortfall or len(selected) < required):
        raise ValueError(f"MediTOD gold候補が不足しています: {len(selected)}/{target}")
    rows = []
    for sample in selected:
        metadata = sample["metadata"]
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "conversation_id": sample["conversation_id"],
                "turn_index": metadata["assistant_turn_index"],
                "prompt": history_text(sample["history"]),
                "response": sample["response"],
                "history": sample["history"],
                "next_user_turn": sample.get("next_user_turn"),
                "metadata": {
                    "source_dataset": "MediTOD",
                    "source_split": "train",
                    "context_turns": len(sample["history"]),
                    "response_intents": metadata.get("response_intents", []),
                    "response_slots": metadata.get("response_slots", []),
                    "response_attributes": metadata.get("response_attributes", []),
                    "protected_medical_terms": protected_medical_terms(sample),
                    "gold": True,
                },
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="MediTOD gold候補を作成")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument(
        "--allow-target-shortfall",
        action="store_true",
        help="target未満でもminimum-records以上の候補があれば保存する",
    )
    parser.add_argument(
        "--minimum-records",
        type=int,
        help="allow-target-shortfall時に必要な最低件数",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = collect_gold_candidates(
        read_jsonl(args.samples),
        target=args.target,
        seed=args.seed,
        allow_target_shortfall=args.allow_target_shortfall,
        minimum_records=args.minimum_records,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"MediTOD gold候補を書き出しました: {output} ({len(rows)}/{args.target}件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
