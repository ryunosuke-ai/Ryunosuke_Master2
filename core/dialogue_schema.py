"""BASiS追加データセットで共有する会話schema。"""

from __future__ import annotations

import hashlib
import json
from typing import Any


VALID_ROLES = {"user", "assistant"}
VALID_SPLITS = {"train", "validation", "test", "candidate"}


def canonical_json_hash(value: Any) -> str:
    """JSON互換値の安定したSHA-256を返す。"""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_conversation(record: dict[str, Any]) -> dict[str, Any]:
    """共通会話レコードの必須項目と集計値を検証する。"""
    conversation_id = record.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("`conversation_id` は空でない文字列である必要があります。")
    source_dataset = record.get("source_dataset")
    if not isinstance(source_dataset, str) or not source_dataset.strip():
        raise ValueError("`source_dataset` は空でない文字列である必要があります。")
    split = record.get("split")
    if split not in VALID_SPLITS:
        raise ValueError(f"未対応のsplitです: {split!r}")
    turns = record.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("`turns` は1発話以上の配列である必要があります。")

    user_turns = 0
    assistant_turns = 0
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"turns[{turn_index}] はobjectである必要があります。")
        role = turn.get("role")
        if role not in VALID_ROLES:
            raise ValueError(f"turns[{turn_index}] のroleが不正です: {role!r}")
        text = turn.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"turns[{turn_index}] のtextが空です。")
        if role == "user":
            user_turns += 1
        else:
            assistant_turns += 1

    expected = {
        "num_messages": len(turns),
        "num_user_turns": user_turns,
        "num_assistant_turns": assistant_turns,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"`{key}`={record.get(key)!r} と実際の値{value}が一致しません。")
    if not isinstance(record.get("metadata"), dict):
        raise ValueError("`metadata` はobjectである必要があります。")
    return record


def build_assistant_samples(record: dict[str, Any]) -> list[dict[str, Any]]:
    """会話を完全履歴付きassistant応答サンプルへ変換する。"""
    validate_conversation(record)
    turns = record["turns"]
    samples: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue
        history = [
            {"role": previous["role"], "text": previous["text"]}
            for previous in turns[:turn_index]
        ]
        next_user_turn = None
        if turn_index + 1 < len(turns) and turns[turn_index + 1]["role"] == "user":
            next_user_turn = turns[turn_index + 1]["text"]
        source_turn_indices = turn.get("metadata", {}).get("source_turn_indices", [turn_index])
        sample_id = f"{record['conversation_id']}#assistant-{turn_index:04d}"
        structurally_dpo_eligible = bool(history and history[-1]["role"] == "user")
        split_allows_training = record["split"] in {"train", "candidate"}
        source_allows_training = record["metadata"].get("eligible_for_training", True) is True
        samples.append(
            {
                "sample_id": sample_id,
                "conversation_id": record["conversation_id"],
                "history": history,
                "response": turn["text"],
                "next_user_turn": next_user_turn,
                "metadata": {
                    "source_dataset": record["source_dataset"],
                    "split": record["split"],
                    "assistant_turn_index": turn_index,
                    "source_turn_indices": source_turn_indices,
                    "teacher_moves": turn.get("metadata", {}).get("teacher_moves", []),
                    "after_state": "observed" if next_user_turn is not None else "unobserved",
                    "after_state_observed": next_user_turn is not None,
                    "history_ends_with_user": structurally_dpo_eligible,
                    "structurally_dpo_eligible": structurally_dpo_eligible,
                    "split_allows_training": split_allows_training,
                    "source_allows_training": source_allows_training,
                    "dpo_eligible": (
                        structurally_dpo_eligible
                        and split_allows_training
                        and source_allows_training
                    ),
                },
            }
        )
    return samples
