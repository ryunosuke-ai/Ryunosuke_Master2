"""MathDial共通形式変換パイプラインのテスト。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from core.dialogue_schema import build_assistant_samples, validate_conversation
from tools.mathdial_dataset import (
    assert_no_split_leakage,
    convert_mathdial_row,
    normalize_turns,
    parse_mathdial_conversation,
    prepare_mathdial,
)


CONFIG = {
    "language": "English",
    "validation_ratio": 0.5,
    "seed": 42,
    "teacher_moves": ["probing", "focus", "telling", "generic"],
    "merge_consecutive_assistant_turns": True,
    "deduplicate_identical_adjacent_assistant_text": True,
}


def make_row(qid: str, marker: str, *, scenario: str = "1") -> dict:
    """テスト用MathDial行を作る。"""
    return {
        "qid": qid,
        "scenario": scenario,
        "question": f"Question {marker}",
        "ground_truth": f"Answer {marker}",
        "student_incorrect_solution": f"Wrong {marker}",
        "student_profile": "A student profile.",
        "teacher_described_confusion": "A confusion.",
        "self-correctness": "Yes",
        "self-typical-confusion": "4",
        "self-typical-interactions": "5",
        "conversation": (
            f"Teacher: (generic)Explain your attempt {marker}.|EOM|"
            f"Steven: I tried method {marker}.|EOM|"
            f"Teacher: (probing)What should happen next {marker}?|EOM|"
            f"Steven: I can correct it {marker}."
        ),
    }


def test_parser_maps_persona_and_removes_only_known_teacher_moves():
    turns, counts = parse_mathdial_conversation(
        "Teacher: (focus)Check the units.|EOM|Luca: I used 12.|EOM|"
        "Teacher: (2/3) should stay in the text.|EOM|Teacher:   ",
        teacher_moves=CONFIG["teacher_moves"],
    )

    assert [turn.role for turn in turns] == ["assistant", "user", "assistant"]
    assert turns[0].text == "Check the units."
    assert turns[0].teacher_move == "focus"
    assert turns[1].source_speaker == "Luca"
    assert turns[2].text == "(2/3) should stay in the text."
    assert turns[2].teacher_move is None
    assert counts.empty_segments_removed == 1
    assert counts.unknown_teacher_move_prefixes == 1


def test_consecutive_teacher_turns_are_merged_with_all_source_metadata():
    parsed, _ = parse_mathdial_conversation(
        "Teacher: (generic)Start.|EOM|Steven: Attempt.|EOM|"
        "Teacher: (focus)Use one unit.|EOM|Teacher: (probing)Use one unit.|EOM|"
        "Teacher: (telling)Then divide by two.|EOM|Steven: I see.",
        teacher_moves=CONFIG["teacher_moves"],
    )

    turns, counts = normalize_turns(
        parsed,
        merge_consecutive_assistant_turns=True,
        deduplicate_identical_adjacent_assistant_text=True,
    )

    assert [turn["role"] for turn in turns] == ["assistant", "user", "assistant", "user"]
    assert turns[2]["text"] == "Use one unit.\nThen divide by two."
    assert turns[2]["metadata"]["teacher_moves"] == ["focus", "probing", "telling"]
    assert turns[2]["metadata"]["source_turn_indices"] == [2, 3, 4]
    assert turns[2]["metadata"]["merged_source_segment_count"] == 3
    assert counts.consecutive_assistant_boundaries == 2
    assert counts.assistant_merge_groups == 1
    assert counts.identical_assistant_segments_removed == 1


def test_multiline_exact_duplicate_is_detected_from_whole_source_segment():
    parsed, _ = parse_mathdial_conversation(
        "Teacher: (generic)Start.|EOM|Student: Attempt.|EOM|"
        "Teacher: (focus)First line.\nSecond line.|EOM|"
        "Teacher: (focus)First line.\nSecond line.|EOM|Student: Done.",
        teacher_moves=CONFIG["teacher_moves"],
    )

    turns, counts = normalize_turns(
        parsed,
        merge_consecutive_assistant_turns=True,
        deduplicate_identical_adjacent_assistant_text=True,
    )

    assert turns[2]["text"] == "First line.\nSecond line."
    assert counts.identical_assistant_segments_removed == 1
    assert turns[2]["metadata"]["source_teacher_moves"] == ["focus", "focus"]


def test_convert_mathdial_row_uses_common_schema_and_preserves_annotations():
    row = make_row("q1", "alpha")

    record, _ = convert_mathdial_row(
        row,
        source_split="train",
        output_split="validation",
        row_index=7,
        config=CONFIG,
    )

    assert validate_conversation(record) is record
    assert record["conversation_id"].startswith("mathdial_train_000007_")
    assert record["source_dataset"] == "mathdial"
    assert record["split"] == "validation"
    assert record["turns"][0]["role"] == "assistant"
    assert "(generic)" not in record["turns"][0]["text"]
    assert record["turns"][0]["metadata"]["teacher_moves"] == ["generic"]
    assert record["metadata"]["qid"] == "q1"
    assert record["metadata"]["ground_truth"] == "Answer alpha"


def test_assistant_samples_keep_full_history_and_next_user_turn():
    row = make_row("q1", "alpha")
    record, _ = convert_mathdial_row(
        row,
        source_split="train",
        output_split="train",
        row_index=0,
        config=CONFIG,
    )

    samples = build_assistant_samples(record)

    assert len(samples) == 2
    assert samples[0]["history"] == []
    assert samples[0]["next_user_turn"] == "I tried method alpha."
    assert samples[0]["metadata"]["dpo_eligible"] is False
    assert samples[1]["history"] == [
        {"role": "assistant", "text": "Explain your attempt alpha."},
        {"role": "user", "text": "I tried method alpha."},
    ]
    assert samples[1]["next_user_turn"] == "I can correct it alpha."
    assert samples[1]["metadata"]["after_state"] == "observed"
    assert samples[1]["metadata"]["dpo_eligible"] is True


def test_terminal_assistant_sample_marks_after_state_unobserved():
    row = make_row("q1", "alpha")
    row["conversation"] += "|EOM|Teacher: (generic)Good work."
    record, _ = convert_mathdial_row(
        row,
        source_split="train",
        output_split="train",
        row_index=0,
        config=CONFIG,
    )

    sample = build_assistant_samples(record)[-1]

    assert sample["next_user_turn"] is None
    assert sample["metadata"]["after_state"] == "unobserved"
    assert sample["metadata"]["after_state_observed"] is False


def test_test_split_samples_are_never_dpo_eligible():
    record, _ = convert_mathdial_row(
        make_row("q-test", "held-out"),
        source_split="test",
        output_split="test",
        row_index=0,
        config=CONFIG,
    )
    record["metadata"]["eligible_for_training"] = False

    samples = build_assistant_samples(record)

    assert any(sample["metadata"]["structurally_dpo_eligible"] for sample in samples)
    assert not any(sample["metadata"]["dpo_eligible"] for sample in samples)


def test_prepare_mathdial_quarantines_official_test_qids_and_is_deterministic():
    rows = {
        "train": [
            make_row("overlap", "train-overlap"),
            make_row("train-a", "train-a"),
            make_row("train-b", "train-b"),
        ],
        "test": [make_row("overlap", "test-overlap")],
    }

    first = prepare_mathdial(deepcopy(rows), config=CONFIG)
    second = prepare_mathdial(deepcopy(rows), config=CONFIG)

    assert len(first.quarantine) == 1
    assert first.quarantine[0]["metadata"]["exclusion_reason"] == "qid_overlaps_official_test"
    assert {record["metadata"]["qid"] for record in first.conversations if record["split"] == "test"} == {
        "overlap"
    }
    assert not any(
        record["metadata"]["qid"] == "overlap" and record["split"] == "train"
        for record in first.conversations
    )
    assert [record["split"] for record in first.conversations] == [
        record["split"] for record in second.conversations
    ]
    assert first.summary["leakage_check"]["status"] == "passed"
    assert first.summary["raw"]["train_conversations_quarantined"] == 1


def test_split_leakage_detector_rejects_same_qid_across_splits():
    train, _ = convert_mathdial_row(
        make_row("same-qid", "train"),
        source_split="train",
        output_split="train",
        row_index=0,
        config=CONFIG,
    )
    test, _ = convert_mathdial_row(
        make_row("same-qid", "test"),
        source_split="test",
        output_split="test",
        row_index=0,
        config=CONFIG,
    )

    with pytest.raises(ValueError, match="qidがsplit間で重複"):
        assert_no_split_leakage([train, test])
