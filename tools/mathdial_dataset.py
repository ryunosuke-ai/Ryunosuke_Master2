"""MathDialの取得、正規化、分割、統計生成。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from core.dialogue_schema import (
    build_assistant_samples,
    canonical_json_hash,
    validate_conversation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets" / "mathdial.yaml"
TEACHER_PATTERN = re.compile(r"^\s*Teacher\s*:\s*(.*)$", re.IGNORECASE | re.DOTALL)
PERSONA_PATTERN = re.compile(r"^\s*([^:\n]+?)\s*:\s*(.*)$", re.DOTALL)


@dataclass
class NormalizationCounts:
    """正規化で発生した操作数。"""

    source_segments: int = 0
    empty_segments_removed: int = 0
    consecutive_assistant_boundaries: int = 0
    consecutive_assistant_boundaries_before_empty_removal: int = 0
    identical_assistant_segments_removed: int = 0
    assistant_merge_groups: int = 0
    unknown_teacher_move_prefixes: int = 0
    source_user_segments: int = 0
    source_assistant_segments: int = 0
    source_user_turns_retained: int = 0
    source_assistant_turns_retained: int = 0
    source_user_characters: int = 0
    source_assistant_characters: int = 0
    empty_assistant_between_assistant_and_user: int = 0
    empty_assistant_between_assistants: int = 0
    empty_assistant_between_users: int = 0
    empty_assistant_other_context: int = 0

    def add(self, other: "NormalizationCounts") -> None:
        """別会話の集計を加算する。"""
        for key in self.__dataclass_fields__:
            setattr(self, key, getattr(self, key) + getattr(other, key))

    def as_dict(self) -> dict[str, int]:
        """JSON用dictへ変換する。"""
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass
class ParsedTurn:
    """MathDialの元発話。"""

    role: str
    text: str
    source_turn_index: int
    source_speaker: str
    teacher_move: str | None = None


@dataclass
class PreparedMathDial:
    """前処理結果一式。"""

    conversations: list[dict[str, Any]]
    samples: list[dict[str, Any]]
    quarantine: list[dict[str, Any]]
    summary: dict[str, Any]
    normalization_counts: NormalizationCounts = field(default_factory=NormalizationCounts)


def load_yaml_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """MathDial YAML設定を読み込む。"""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML設定の読み込みには `PyYAML` が必要です。") from exc
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"設定ファイルはobjectである必要があります: {config_path}")
    return payload


def sha256_file(path: Path | str) -> str:
    """ファイルのSHA-256を返す。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """JSONLを厳密に読み込む。"""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}の{line_number}行目をJSONとして読めません: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}の{line_number}行目はobjectである必要があります。")
            records.append(payload)
    return records


def download_mathdial_files(dataset_name: str, revision: str) -> tuple[dict[str, Path], dict[str, Any]]:
    """公式Hugging Faceからrevision固定でtrain/testを取得する。"""
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "hf_cache"))
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("MathDial取得には `huggingface_hub` が必要です。") from exc
    paths = {
        split: Path(
            hf_hub_download(
                repo_id=dataset_name,
                filename=f"{split}.jsonl",
                repo_type="dataset",
                revision=revision,
            )
        )
        for split in ("train", "test")
    }
    info = HfApi().dataset_info(repo_id=dataset_name, revision=revision)
    card_data = getattr(info, "card_data", None)
    license_value = getattr(card_data, "license", None) if card_data is not None else None
    return paths, {
        "dataset_name": dataset_name,
        "source_url": f"https://huggingface.co/datasets/{dataset_name}",
        "official_github_url": "https://github.com/eth-nlped/mathdial",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_revision": revision,
        "resolved_revision": getattr(info, "sha", revision),
        "huggingface_license": license_value,
        "source_files": {
            split: {"filename": path.name, "sha256": sha256_file(path)}
            for split, path in paths.items()
        },
    }


def _teacher_move_pattern(teacher_moves: Iterable[str]) -> re.Pattern[str]:
    choices = "|".join(re.escape(move) for move in teacher_moves)
    return re.compile(rf"^\s*\(({choices})\)\s*", re.IGNORECASE)


def parse_mathdial_conversation(
    conversation: str,
    *,
    teacher_moves: Iterable[str],
) -> tuple[list[ParsedTurn], NormalizationCounts]:
    """MathDialのconversation文字列を元発話列へ変換する。"""
    move_pattern = _teacher_move_pattern(teacher_moves)
    turns: list[ParsedTurn] = []
    audit_turns: list[tuple[str, str]] = []
    counts = NormalizationCounts()
    for source_turn_index, raw_segment in enumerate(conversation.split("|EOM|")):
        counts.source_segments += 1
        segment = raw_segment.strip()
        if not segment:
            counts.empty_segments_removed += 1
            continue
        teacher_match = TEACHER_PATTERN.match(segment)
        if teacher_match:
            text = teacher_match.group(1)
            move_match = move_pattern.match(text)
            teacher_move = None
            if move_match:
                teacher_move = move_match.group(1).lower()
                text = text[move_match.end() :]
            elif re.match(r"^\s*\([^)]*\)", text):
                counts.unknown_teacher_move_prefixes += 1
            text = text.strip()
            counts.source_assistant_segments += 1
            audit_turns.append(("assistant", text))
            if not text:
                counts.empty_segments_removed += 1
                continue
            counts.source_assistant_turns_retained += 1
            counts.source_assistant_characters += len(text)
            turns.append(
                ParsedTurn(
                    role="assistant",
                    text=text,
                    source_turn_index=source_turn_index,
                    source_speaker="Teacher",
                    teacher_move=teacher_move,
                )
            )
            continue
        persona_match = PERSONA_PATTERN.match(segment)
        if not persona_match:
            raise ValueError(f"話者と本文を分離できない発話があります: {segment[:120]!r}")
        persona = persona_match.group(1).strip()
        text = persona_match.group(2).strip()
        counts.source_user_segments += 1
        audit_turns.append(("user", text))
        if not text:
            counts.empty_segments_removed += 1
            continue
        counts.source_user_turns_retained += 1
        counts.source_user_characters += len(text)
        turns.append(
            ParsedTurn(
                role="user",
                text=text,
                source_turn_index=source_turn_index,
                source_speaker=persona,
            )
        )
    counts.consecutive_assistant_boundaries_before_empty_removal = sum(
        left[0] == right[0] == "assistant"
        for left, right in zip(audit_turns, audit_turns[1:])
    )
    for index, (role, text) in enumerate(audit_turns):
        if role != "assistant" or text:
            continue
        previous_role = audit_turns[index - 1][0] if index else "start"
        next_role = audit_turns[index + 1][0] if index + 1 < len(audit_turns) else "end"
        if (previous_role, next_role) == ("assistant", "user"):
            counts.empty_assistant_between_assistant_and_user += 1
        elif (previous_role, next_role) == ("assistant", "assistant"):
            counts.empty_assistant_between_assistants += 1
        elif (previous_role, next_role) == ("user", "user"):
            counts.empty_assistant_between_users += 1
        else:
            counts.empty_assistant_other_context += 1
    return turns, counts


def normalize_turns(
    parsed_turns: list[ParsedTurn],
    *,
    merge_consecutive_assistant_turns: bool,
    deduplicate_identical_adjacent_assistant_text: bool,
) -> tuple[list[dict[str, Any]], NormalizationCounts]:
    """連続Teacher発話を監査可能な形で結合する。"""
    counts = NormalizationCounts()
    normalized: list[dict[str, Any]] = []
    for turn in parsed_turns:
        metadata = {
            "source_turn_indices": [turn.source_turn_index],
            "source_speakers": [turn.source_speaker],
            "teacher_moves": [turn.teacher_move] if turn.teacher_move else [],
            "source_teacher_moves": [turn.teacher_move],
            "source_segments": [turn.text],
        }
        if (
            merge_consecutive_assistant_turns
            and turn.role == "assistant"
            and normalized
            and normalized[-1]["role"] == "assistant"
        ):
            counts.consecutive_assistant_boundaries += 1
            previous = normalized[-1]
            previous_metadata = previous["metadata"]
            if previous_metadata["merged_source_segment_count"] == 1:
                counts.assistant_merge_groups += 1
            previous_source_text = previous_metadata["source_segments"][-1]
            previous_metadata["source_turn_indices"].append(turn.source_turn_index)
            previous_metadata["source_speakers"].append(turn.source_speaker)
            previous_metadata["source_segments"].append(turn.text)
            previous_metadata["source_teacher_moves"].append(turn.teacher_move)
            if turn.teacher_move:
                previous_metadata["teacher_moves"].append(turn.teacher_move)
            if (
                deduplicate_identical_adjacent_assistant_text
                and previous_source_text.strip() == turn.text.strip()
            ):
                counts.identical_assistant_segments_removed += 1
            else:
                previous["text"] = f"{previous['text']}\n{turn.text}"
            previous_metadata["merged_source_segment_count"] = len(
                previous_metadata["source_turn_indices"]
            )
            continue
        metadata["merged_source_segment_count"] = 1
        normalized.append({"role": turn.role, "text": turn.text, "metadata": metadata})
    return normalized, counts


def mathdial_conversation_id(source_split: str, row_index: int, row: dict[str, Any]) -> str:
    """revision内で安定した会話IDを作る。"""
    row_hash = canonical_json_hash(row)[:12]
    return f"mathdial_{source_split}_{row_index:06d}_{row_hash}"


def convert_mathdial_row(
    row: dict[str, Any],
    *,
    source_split: str,
    output_split: str,
    row_index: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], NormalizationCounts]:
    """公式MathDial 1行を共通会話schemaへ変換する。"""
    raw_conversation = row.get("conversation")
    if not isinstance(raw_conversation, str) or not raw_conversation.strip():
        raise ValueError(f"{source_split}[{row_index}] のconversationが空です。")
    parsed, parse_counts = parse_mathdial_conversation(
        raw_conversation,
        teacher_moves=config["teacher_moves"],
    )
    turns, merge_counts = normalize_turns(
        parsed,
        merge_consecutive_assistant_turns=bool(config["merge_consecutive_assistant_turns"]),
        deduplicate_identical_adjacent_assistant_text=bool(
            config["deduplicate_identical_adjacent_assistant_text"]
        ),
    )
    parse_counts.add(merge_counts)
    conversation_id = mathdial_conversation_id(source_split, row_index, row)
    record = {
        "conversation_id": conversation_id,
        "source_dataset": "mathdial",
        "split": output_split,
        "turns": turns,
        "num_messages": len(turns),
        "num_user_turns": sum(turn["role"] == "user" for turn in turns),
        "num_assistant_turns": sum(turn["role"] == "assistant" for turn in turns),
        "language": config.get("language", "English"),
        "metadata": {
            "source_split": source_split,
            "source_row_index": row_index,
            "qid": str(row.get("qid", "")),
            "scenario": row.get("scenario"),
            "question": row.get("question"),
            "ground_truth": row.get("ground_truth"),
            "student_incorrect_solution": row.get("student_incorrect_solution"),
            "student_profile": row.get("student_profile"),
            "teacher_described_confusion": row.get("teacher_described_confusion"),
            "self_correctness": row.get("self-correctness"),
            "self_typical_confusion": row.get("self-typical-confusion"),
            "self_typical_interactions": row.get("self-typical-interactions"),
            "source_conversation_sha256": hashlib.sha256(
                raw_conversation.encode("utf-8")
            ).hexdigest(),
            "normalization": parse_counts.as_dict(),
        },
    }
    return validate_conversation(record), parse_counts


def _validation_qids(qids: set[str], *, ratio: float, seed: int) -> set[str]:
    """qid単位の決定論的validation集合を返す。"""
    if not 0.0 <= ratio < 1.0:
        raise ValueError("validation_ratioは0以上1未満である必要があります。")
    ordered = sorted(
        qids,
        key=lambda qid: hashlib.sha256(f"{seed}:{qid}".encode("utf-8")).hexdigest(),
    )
    count = round(len(ordered) * ratio)
    return set(ordered[:count])


def assert_no_split_leakage(records: list[dict[str, Any]]) -> None:
    """会話ID、qid、本文hashのsplit間リークを検出する。"""
    split_by_conversation: dict[str, str] = {}
    split_by_qid: dict[str, str] = {}
    split_by_content: dict[str, str] = {}
    for record in records:
        split = record["split"]
        keys = (
            (split_by_conversation, record["conversation_id"], "conversation_id"),
            (split_by_qid, str(record["metadata"]["qid"]), "qid"),
            (
                split_by_content,
                canonical_json_hash(
                    [{"role": turn["role"], "text": turn["text"]} for turn in record["turns"]]
                ),
                "normalized content",
            ),
        )
        for mapping, key, label in keys:
            previous = mapping.get(key)
            if previous is not None and previous != split:
                raise ValueError(f"{label}がsplit間で重複しています: {key} ({previous}/{split})")
            mapping[key] = split


def _mean(values: list[int]) -> float:
    return round(statistics.mean(values), 3) if values else 0.0


def summarize_conversations(records: list[dict[str, Any]]) -> dict[str, Any]:
    """共通会話とサンプルの統計を返す。"""
    turns = [turn for record in records for turn in record["turns"]]
    by_role = {
        role: [turn for turn in turns if turn["role"] == role]
        for role in ("user", "assistant")
    }
    teacher_moves = Counter(
        move
        for turn in by_role["assistant"]
        for move in turn.get("metadata", {}).get("teacher_moves", [])
    )
    return {
        "conversations": len(records),
        "messages": len(turns),
        "average_messages_per_conversation": _mean(
            [record["num_messages"] for record in records]
        ),
        "user_turns": len(by_role["user"]),
        "assistant_turns": len(by_role["assistant"]),
        "average_user_characters": _mean([len(turn["text"]) for turn in by_role["user"]]),
        "average_assistant_characters": _mean(
            [len(turn["text"]) for turn in by_role["assistant"]]
        ),
        "teacher_moves": dict(sorted(teacher_moves.items())),
    }


def audit_exact_assistant_duplicates(records: list[dict[str, Any]]) -> dict[str, Any]:
    """連続assistant内の完全一致segmentを監査する。"""
    duplicates: list[dict[str, Any]] = []
    merge_group_sizes: Counter[int] = Counter()
    for record in records:
        for turn_index, turn in enumerate(record["turns"]):
            if turn["role"] != "assistant":
                continue
            metadata = turn.get("metadata", {})
            merged_count = int(metadata.get("merged_source_segment_count", 1))
            if merged_count > 1:
                merge_group_sizes[merged_count] += 1
            segments = metadata.get("source_segments", [])
            source_indices = metadata.get("source_turn_indices", [])
            source_moves = metadata.get("source_teacher_moves", [])
            for segment_index, (left, right) in enumerate(zip(segments, segments[1:])):
                if left.strip() != right.strip():
                    continue
                previous_user = None
                if turn_index and record["turns"][turn_index - 1]["role"] == "user":
                    previous_user = record["turns"][turn_index - 1]["text"]
                next_user = None
                if (
                    turn_index + 1 < len(record["turns"])
                    and record["turns"][turn_index + 1]["role"] == "user"
                ):
                    next_user = record["turns"][turn_index + 1]["text"]
                duplicates.append(
                    {
                        "conversation_id": record["conversation_id"],
                        "source_split": record["metadata"]["source_split"],
                        "qid": record["metadata"]["qid"],
                        "source_turn_indices": source_indices[
                            segment_index : segment_index + 2
                        ],
                        "teacher_moves": source_moves[segment_index : segment_index + 2],
                        "previous_user": previous_user,
                        "duplicate_assistant_text": left,
                        "next_user": next_user,
                    }
                )
    lengths = [len(item["duplicate_assistant_text"]) for item in duplicates]
    unique_texts = {item["duplicate_assistant_text"] for item in duplicates}
    same_move = sum(
        len(item["teacher_moves"]) == 2
        and item["teacher_moves"][0] == item["teacher_moves"][1]
        for item in duplicates
    )
    return {
        "exact_duplicate_boundaries": len(duplicates),
        "unique_duplicate_texts": len(unique_texts),
        "same_teacher_move": same_move,
        "different_teacher_move": len(duplicates) - same_move,
        "minimum_characters": min(lengths) if lengths else 0,
        "median_characters": statistics.median(lengths) if lengths else 0,
        "mean_characters": round(statistics.mean(lengths), 3) if lengths else 0.0,
        "maximum_characters": max(lengths) if lengths else 0,
        "assistant_merge_groups": sum(merge_group_sizes.values()),
        "assistant_merge_group_size_distribution": {
            str(size): count for size, count in sorted(merge_group_sizes.items())
        },
        "examples": duplicates[:5],
    }


def prepare_mathdial(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    config: dict[str, Any],
) -> PreparedMathDial:
    """公式train/testを厳密qid分離して正規化する。"""
    train_rows = rows_by_split["train"]
    test_rows = rows_by_split["test"]
    test_qids = {str(row.get("qid", "")) for row in test_rows}
    train_qids = {str(row.get("qid", "")) for row in train_rows}
    overlap_qids = train_qids & test_qids
    eligible_train_rows = [row for row in train_rows if str(row.get("qid", "")) not in overlap_qids]
    validation_qids = _validation_qids(
        {str(row.get("qid", "")) for row in eligible_train_rows},
        ratio=float(config["validation_ratio"]),
        seed=int(config["seed"]),
    )

    conversations: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    counts = NormalizationCounts()
    counts_by_source_split = {
        "train": NormalizationCounts(),
        "test": NormalizationCounts(),
    }
    for row_index, row in enumerate(train_rows):
        qid = str(row.get("qid", ""))
        if qid in overlap_qids:
            record, row_counts = convert_mathdial_row(
                row,
                source_split="train",
                output_split="train",
                row_index=row_index,
                config=config,
            )
            record["metadata"].update(
                {
                    "eligible_for_training": False,
                    "exclusion_reason": "qid_overlaps_official_test",
                }
            )
            quarantine.append(record)
        else:
            output_split = "validation" if qid in validation_qids else "train"
            record, row_counts = convert_mathdial_row(
                row,
                source_split="train",
                output_split=output_split,
                row_index=row_index,
                config=config,
            )
            record["metadata"]["eligible_for_training"] = True
            conversations.append(record)
        counts.add(row_counts)
        counts_by_source_split["train"].add(row_counts)
    for row_index, row in enumerate(test_rows):
        record, row_counts = convert_mathdial_row(
            row,
            source_split="test",
            output_split="test",
            row_index=row_index,
            config=config,
        )
        record["metadata"]["eligible_for_training"] = False
        conversations.append(record)
        counts.add(row_counts)
        counts_by_source_split["test"].add(row_counts)

    assert_no_split_leakage(conversations)
    samples = [sample for record in conversations for sample in build_assistant_samples(record)]
    all_normalized_records = [*conversations, *quarantine]
    split_summary = {
        split: summarize_conversations([record for record in conversations if record["split"] == split])
        for split in ("train", "validation", "test")
    }
    summary = {
        "raw": {
            "train_conversations": len(train_rows),
            "test_conversations": len(test_rows),
            "total_conversations": len(train_rows) + len(test_rows),
            "train_qids": len(train_qids),
            "test_qids": len(test_qids),
            "overlap_qids": len(overlap_qids),
            "train_conversations_quarantined": len(quarantine),
            "messages": counts.source_user_turns_retained
            + counts.source_assistant_turns_retained,
            "average_messages_per_conversation": round(
                (counts.source_user_turns_retained + counts.source_assistant_turns_retained)
                / (len(train_rows) + len(test_rows)),
                3,
            ),
            "user_turns": counts.source_user_turns_retained,
            "assistant_turns": counts.source_assistant_turns_retained,
            "average_user_characters": round(
                counts.source_user_characters / max(1, counts.source_user_turns_retained), 3
            ),
            "average_assistant_characters": round(
                counts.source_assistant_characters
                / max(1, counts.source_assistant_turns_retained),
                3,
            ),
        },
        "normalization": counts.as_dict(),
        "normalization_by_source_split": {
            split: split_counts.as_dict()
            for split, split_counts in counts_by_source_split.items()
        },
        "exact_duplicate_audit": audit_exact_assistant_duplicates(all_normalized_records),
        "normalized_all": summarize_conversations(all_normalized_records),
        "normalized_main": summarize_conversations(conversations),
        "by_split": split_summary,
        "samples": {
            "total": len(samples),
            "after_state_observed": sum(
                sample["metadata"]["after_state_observed"] for sample in samples
            ),
            "after_state_unobserved": sum(
                not sample["metadata"]["after_state_observed"] for sample in samples
            ),
            "dpo_eligible": sum(sample["metadata"]["dpo_eligible"] for sample in samples),
        },
        "leakage_check": {
            "status": "passed",
            "dimensions": ["conversation_id", "qid", "normalized_content_sha256"],
        },
    }
    return PreparedMathDial(
        conversations=conversations,
        samples=samples,
        quarantine=quarantine,
        summary=summary,
        normalization_counts=counts,
    )
