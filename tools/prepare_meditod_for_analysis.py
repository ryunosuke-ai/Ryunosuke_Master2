"""MediTOD trainから完全診療と全train annotation集計を分析用に作る。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.meditod_dataset import ATTRIBUTE_KEYS, file_sha256, write_jsonl


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def _annotations(turn: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for variant in turn.get("metadata", {}).get("annotation_variants", [])
        for item in variant.get("annotations", [])
        if isinstance(item, dict)
    ]


def turn_slots(turn: dict[str, Any]) -> set[str]:
    return {
        str(item.get("slot", "")).strip().lower()
        for item in _annotations(turn)
        if item.get("slot")
    }


def conversation_slots(record: dict[str, Any]) -> set[str]:
    return {slot for turn in record["turns"] for slot in turn_slots(turn)}


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_analysis_conversations(
    conversations: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    """slot coverageと会話長を層化し、train完全診療を選ぶ。"""
    train = [
        row
        for row in conversations
        if row.get("split") == "train"
        and not row.get("metadata", {}).get("ood")
        and row.get("metadata", {}).get("eligible_for_training") is True
    ]
    if len(train) < count:
        raise ValueError(f"MediTOD train診療が不足しています: {len(train)}/{count}")
    lengths = sorted(row["num_messages"] for row in train)
    lower = lengths[len(lengths) // 3]
    upper = lengths[(2 * len(lengths)) // 3]

    def length_bucket(row: dict[str, Any]) -> str:
        value = row["num_messages"]
        return "short" if value <= lower else "medium" if value <= upper else "long"

    slot_frequency = Counter(slot for row in train for slot in conversation_slots(row))
    buckets = {name: [] for name in ("short", "medium", "long")}
    for row in train:
        rarity = sum(1.0 / slot_frequency[slot] for slot in conversation_slots(row))
        buckets[length_bucket(row)].append((-rarity, _rank(seed, row["conversation_id"]), row))
    for values in buckets.values():
        values.sort(key=lambda item: (item[0], item[1]))
    quota = {"short": count // 3, "medium": count // 3, "long": count // 3}
    for name in list(quota)[: count % 3]:
        quota[name] += 1
    selected = [item[2] for name, values in buckets.items() for item in values[: quota[name]]]
    if len(selected) < count:
        selected_ids = {row["conversation_id"] for row in selected}
        remaining = sorted(
            (row for row in train if row["conversation_id"] not in selected_ids),
            key=lambda row: _rank(seed, row["conversation_id"]),
        )
        selected.extend(remaining[: count - len(selected)])
    return sorted(selected, key=lambda row: row["conversation_id"])


def build_corpus_aggregates(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    """train全体のannotation頻度・段階・遷移を決定論的に集計する。"""
    train = [
        row for row in conversations
        if row.get("split") == "train" and not row.get("metadata", {}).get("ood")
    ]
    intents: Counter[str] = Counter()
    slots: Counter[str] = Counter()
    attributes: Counter[str] = Counter()
    slot_by_decile: dict[int, Counter[str]] = defaultdict(Counter)
    doctor_slot_transitions: Counter[str] = Counter()
    doctor_to_patient: Counter[str] = Counter()
    disagreement_turns = 0
    for record in train:
        previous_doctor_slots: set[str] | None = None
        turns = record["turns"]
        for index, turn in enumerate(turns):
            annotation_variants = turn.get("metadata", {}).get("annotation_variants", [])
            signatures = {
                json.dumps(variant.get("annotations", []), sort_keys=True, ensure_ascii=False)
                for variant in annotation_variants
            }
            disagreement_turns += len(signatures) > 1
            annotations = _annotations(turn)
            current_slots = turn_slots(turn)
            for item in annotations:
                if item.get("intent"):
                    intents[str(item["intent"]).strip().lower()] += 1
                if item.get("slot"):
                    slots[str(item["slot"]).strip().lower()] += 1
                for key in item:
                    if key in ATTRIBUTE_KEYS:
                        attributes[key] += 1
            decile = min(9, int(index * 10 / max(1, len(turns))))
            for slot in current_slots:
                slot_by_decile[decile][slot] += 1
            if turn["role"] == "assistant":
                if previous_doctor_slots:
                    for left in previous_doctor_slots or {"none"}:
                        for right in current_slots or {"none"}:
                            doctor_slot_transitions[f"{left}->{right}"] += 1
                previous_doctor_slots = current_slots
                if index + 1 < len(turns) and turns[index + 1]["role"] == "user":
                    next_slots = turn_slots(turns[index + 1])
                    for left in current_slots or {"none"}:
                        for right in next_slots or {"none"}:
                            doctor_to_patient[f"{left}->{right}"] += 1
    return {
        "train_conversations": len(train),
        "train_turns": sum(row["num_messages"] for row in train),
        "intent_frequency": dict(intents.most_common()),
        "slot_frequency": dict(slots.most_common()),
        "attribute_frequency": dict(attributes.most_common()),
        "slot_by_conversation_decile": {
            str(index): dict(slot_by_decile[index].most_common()) for index in range(10)
        },
        "doctor_slot_transitions": dict(doctor_slot_transitions.most_common(100)),
        "doctor_action_to_next_patient_information": dict(doctor_to_patient.most_common(100)),
        "annotation_disagreement_turns": disagreement_turns,
    }


def to_analysis_record(record: dict[str, Any]) -> dict[str, Any]:
    dialog = []
    for index, turn in enumerate(record["turns"]):
        dialog.append(
            {
                "turn_index": index,
                "speaker": turn["role"],
                "text": turn["text"],
                "annotation_variants": turn.get("metadata", {}).get("annotation_variants", []),
            }
        )
    return {
        "conversation_id": record["conversation_id"],
        "source_dataset": "MediTOD",
        "source_split": "train",
        "source_dialogue_ids": record["metadata"].get("source_dialogue_ids", []),
        "dialog": dialog,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MediTOD分析用完全診療を選択")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aggregate-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    conversations = read_jsonl(args.input)
    selected = select_analysis_conversations(conversations, count=args.count, seed=args.seed)
    analysis = [to_analysis_record(row) for row in selected]
    aggregates = build_corpus_aggregates(conversations)
    write_jsonl(analysis, args.output)
    aggregate_path = Path(args.aggregate_output)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    covered_slots = sorted({slot for row in selected for slot in conversation_slots(row)})
    manifest = {
        "input": args.input,
        "input_sha256": file_sha256(args.input),
        "output": args.output,
        "output_sha256": file_sha256(args.output),
        "aggregate_output": args.aggregate_output,
        "aggregate_sha256": file_sha256(args.aggregate_output),
        "seed": args.seed,
        "requested_conversations": args.count,
        "selected_conversations": len(selected),
        "selected_turns": sum(row["num_messages"] for row in selected),
        "covered_slots": covered_slots,
        "train_only": all(row["split"] == "train" for row in selected),
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
