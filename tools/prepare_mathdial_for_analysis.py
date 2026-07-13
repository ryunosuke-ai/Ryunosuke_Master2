"""MathDial trainからESConv互換の小コーパス分析標本を作る。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


TEACHER_MOVES = ("probing", "focus", "telling", "generic")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """JSONLをobject配列として読む。"""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}をJSONとして読めません: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}はJSON objectではありません。")
            rows.append(row)
    return rows


def file_sha256(path: Path | str) -> str:
    """ファイルのSHA-256を返す。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def teacher_moves(record: dict[str, Any]) -> set[str]:
    """1会話に含まれる既知Teacher moveを返す。"""
    return {
        str(move)
        for turn in record.get("turns", [])
        if turn.get("role") == "assistant"
        for move in turn.get("metadata", {}).get("teacher_moves", [])
        if str(move) in TEACHER_MOVES
    }


def _seeded_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_analysis_conversations(
    records: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """trainのみからqid一意でTeacher move coverageを優先して選ぶ。"""
    train = [row for row in records if row.get("split") == "train"]
    if not train:
        raise ValueError("MathDial train会話がありません。")

    # 同一qidからはTeacher move種類が多い会話を代表として残す。
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for row in train:
        qid = str(row.get("metadata", {}).get("qid", "")).strip()
        if not qid:
            raise ValueError(f"qidが空のtrain会話があります: {row.get('conversation_id')}")
        by_qid.setdefault(qid, []).append(row)
    representatives = []
    for qid, candidates in sorted(by_qid.items()):
        representatives.append(
            min(
                candidates,
                key=lambda row: (
                    -len(teacher_moves(row)),
                    _seeded_rank(seed, str(row["conversation_id"])),
                ),
            )
        )

    if count <= 0 or count >= len(representatives):
        return sorted(representatives, key=lambda row: str(row["conversation_id"]))

    rng = random.Random(seed)
    by_move = {
        move: [row for row in representatives if move in teacher_moves(row)]
        for move in TEACHER_MOVES
    }
    for candidates in by_move.values():
        rng.shuffle(candidates)

    selected: dict[str, dict[str, Any]] = {}
    offsets = {move: 0 for move in TEACHER_MOVES}
    while len(selected) < count:
        changed = False
        for move in TEACHER_MOVES:
            candidates = by_move[move]
            while offsets[move] < len(candidates):
                row = candidates[offsets[move]]
                offsets[move] += 1
                key = str(row["conversation_id"])
                if key in selected:
                    continue
                selected[key] = row
                changed = True
                break
            if len(selected) >= count:
                break
        if not changed:
            break

    if len(selected) < count:
        remaining = [
            row for row in representatives
            if str(row["conversation_id"]) not in selected
        ]
        rng.shuffle(remaining)
        for row in remaining[: count - len(selected)]:
            selected[str(row["conversation_id"])] = row
    if len(selected) < count:
        raise ValueError(f"qid一意なMathDial train会話が不足しています: {len(selected)}/{count}")
    return sorted(selected.values(), key=lambda row: str(row["conversation_id"]))


def to_analysis_record(record: dict[str, Any]) -> dict[str, Any]:
    """共通会話schemaをTeacher move付き分析schemaへ変換する。"""
    dialog = []
    for turn_index, turn in enumerate(record["turns"]):
        item = {
            "turn_index": turn_index,
            "speaker": turn["role"],
            "text": turn["text"],
        }
        if turn["role"] == "assistant":
            item["annotated_teacher_moves"] = list(
                turn.get("metadata", {}).get("teacher_moves", [])
            )
        dialog.append(item)
    metadata = record.get("metadata", {})
    return {
        "conversation_id": record["conversation_id"],
        "source_dataset": "MathDial",
        "source_split": "train",
        "qid": str(metadata.get("qid", "")),
        "question": metadata.get("question"),
        "ground_truth": metadata.get("ground_truth"),
        "dialog": dialog,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """分析標本の件数とTeacher move分布を返す。"""
    moves = Counter(
        move
        for row in records
        for turn in row["dialog"]
        if turn["speaker"] == "assistant"
        for move in turn.get("annotated_teacher_moves", [])
    )
    return {
        "conversations": len(records),
        "unique_qids": len({row["qid"] for row in records}),
        "turns": sum(len(row["dialog"]) for row in records),
        "assistant_turns": sum(
            turn["speaker"] == "assistant"
            for row in records for turn in row["dialog"]
        ),
        "teacher_moves": {move: moves.get(move, 0) for move in TEACHER_MOVES},
    }


def write_jsonl(records: list[dict[str, Any]], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="MathDial分析用代表会話を作成")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected = select_analysis_conversations(
        read_jsonl(args.input), count=args.count, seed=args.seed
    )
    analysis_records = [to_analysis_record(row) for row in selected]
    summary = summarize(analysis_records)
    missing_moves = [move for move, value in summary["teacher_moves"].items() if value == 0]
    if missing_moves:
        raise ValueError(f"分析標本に含まれないTeacher moveがあります: {missing_moves}")
    write_jsonl(analysis_records, args.output)
    manifest = {
        "input": args.input,
        "input_sha256": file_sha256(args.input),
        "output": args.output,
        "output_sha256": file_sha256(args.output),
        "seed": args.seed,
        "requested_conversations": args.count,
        "filters": {"split": "train", "qid_unique": True, "quarantine": False},
        "summary": summary,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"MathDial分析標本を書き出しました: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
