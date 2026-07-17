"""ユーザ評価item作成・集計の軽量テスト。"""

import csv
import json
from collections import Counter
from pathlib import Path

from scripts.analyze_user_eval_results import (
    AXIS_KEYS,
    AXIS_WEIGHTS,
    load_and_normalize_responses,
    normalize_response,
    summarize_rows,
    write_analysis_outputs,
)
from scripts.prepare_user_eval_items import (
    SOURCE_BASIS,
    SOURCE_RANDOM,
    STRATUM_BASIS_WIN,
    STRATUM_CLOSE,
    STRATUM_RANDOM_WIN,
    SelectionConfig,
    build_candidate_records,
    build_eval_item,
    select_user_eval_records,
)
from apps.user_eval_app import load_answer_records_by_item_id, upsert_answer


def make_response(prompt_id: str, category: str) -> dict:
    """テスト用responsesレコードを作る。"""
    return {
        "prompt_id": prompt_id,
        "category": category,
        "prompt": f"{prompt_id} のprompt",
        "history": [{"speaker": "User", "text": "つらいです。"}],
        "axis_focus": ["emotional_reflection_validation"],
        "base_response": f"{prompt_id} basis response",
        "dpo_response": f"{prompt_id} random response",
    }


def make_judgment(prompt_id: str, category: str, stratum: str) -> dict:
    """テスト用judgmentsレコードを作る。"""
    if stratum == STRATUM_BASIS_WIN:
        basis_score = 90.0
        random_score = 80.0
        winner = "base"
    elif stratum == STRATUM_RANDOM_WIN:
        basis_score = 80.0
        random_score = 90.0
        winner = "dpo"
    else:
        basis_score = 86.0
        random_score = 84.0
        winner = "tie"
    return {
        "prompt_id": prompt_id,
        "category": category,
        "winner": winner,
        "weighted_esconv_overall_score_base": basis_score,
        "weighted_esconv_overall_score_dpo": random_score,
    }


def make_balanced_fixture() -> tuple[list[dict], list[dict]]:
    """20/5/5かつ10カテゴリ各3件になるfixtureを作る。"""
    responses = []
    judgments = []
    categories = [f"cat_{index:02d}" for index in range(10)]
    for category_index, category in enumerate(categories):
        strata = (
            [STRATUM_BASIS_WIN, STRATUM_BASIS_WIN, STRATUM_RANDOM_WIN]
            if category_index < 5
            else [STRATUM_BASIS_WIN, STRATUM_BASIS_WIN, STRATUM_CLOSE]
        )
        for item_index, stratum in enumerate(strata):
            prompt_id = f"{category}_{item_index}"
            responses.append(make_response(prompt_id, category))
            judgments.append(make_judgment(prompt_id, category, stratum))
    return responses, judgments


def write_jsonl(path: Path, records: list[dict]) -> None:
    """テスト用JSONLを書き出す。"""
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def axis_ratings(**overrides: int) -> dict[str, int]:
    """5軸評価のテスト用ratingを作る。"""
    ratings = {axis_key: 3 for axis_key in AXIS_KEYS}
    ratings.update(overrides)
    return ratings


def test_select_user_eval_records_matches_target_counts():
    responses, judgments = make_balanced_fixture()
    candidates = build_candidate_records(responses, judgments, close_threshold=3.0)
    selected, info = select_user_eval_records(candidates, config=SelectionConfig(seed=1234))

    assert info["constraints_relaxed"] is False
    assert len(selected) == 30
    assert Counter(record["stratum"] for record in selected) == {
        STRATUM_BASIS_WIN: 20,
        STRATUM_RANDOM_WIN: 5,
        STRATUM_CLOSE: 5,
    }
    assert set(Counter(record["category"] for record in selected).values()) == {3}
    assert Counter(record["model_a_source"] for record in selected) == {
        SOURCE_BASIS: 15,
        SOURCE_RANDOM: 15,
    }


def test_build_eval_item_keeps_hidden_source_for_analysis():
    responses, judgments = make_balanced_fixture()
    candidates = build_candidate_records(responses, judgments, close_threshold=3.0)
    selected, _ = select_user_eval_records(candidates, config=SelectionConfig(seed=1234))

    item = build_eval_item(
        selected[0],
        source_paths={
            "responses_path": "responses.jsonl",
            "judgments_path": "judgments.jsonl",
            "oracle_manifest_path": "manifest.json",
            "prompts_path": "prompts.jsonl",
        },
    )

    assert item["model_a_source"] in {SOURCE_BASIS, SOURCE_RANDOM}
    assert item["model_b_source"] in {SOURCE_BASIS, SOURCE_RANDOM}
    assert item["model_a_source"] != item["model_b_source"]
    assert "oracle_winner" in item
    assert "score_gap" in item


def test_normalize_response_converts_rating_to_basis_score():
    item_lookup = {
        "i1": {"item_id": "i1", "model_a_source": SOURCE_BASIS, "model_b_source": SOURCE_RANDOM},
        "i2": {"item_id": "i2", "model_a_source": SOURCE_RANDOM, "model_b_source": SOURCE_BASIS},
        "i3": {"item_id": "i3", "model_a_source": SOURCE_BASIS, "model_b_source": SOURCE_RANDOM},
    }

    basis_a = normalize_response({"item_id": "i1", "rating": 1}, item_lookup)
    random_a = normalize_response({"item_id": "i2", "rating": 1}, item_lookup)
    tie = normalize_response({"item_id": "i3", "rating": 3}, item_lookup)

    assert basis_a["basis_score"] == 2
    assert basis_a["winner"] == SOURCE_BASIS
    assert random_a["basis_score"] == -2
    assert random_a["winner"] == SOURCE_RANDOM
    assert tie["basis_score"] == 0
    assert tie["winner"] == "tie"


def test_normalize_response_uses_weighted_axis_scores():
    item_lookup = {
        "i1": {"item_id": "i1", "model_a_source": SOURCE_BASIS, "model_b_source": SOURCE_RANDOM},
        "i2": {"item_id": "i2", "model_a_source": SOURCE_RANDOM, "model_b_source": SOURCE_BASIS},
    }
    ratings = axis_ratings(
        emotion_reception=1,
        advice_timing=2,
        contextual_response=4,
        warmth=3,
        conversation_progress=5,
    )
    expected = (
        2 * AXIS_WEIGHTS["emotion_reception"]
        + 1 * AXIS_WEIGHTS["advice_timing"]
        - 1 * AXIS_WEIGHTS["contextual_response"]
        + 0 * AXIS_WEIGHTS["warmth"]
        - 2 * AXIS_WEIGHTS["conversation_progress"]
    )

    basis_a = normalize_response({"item_id": "i1", "axis_ratings": ratings}, item_lookup)
    random_a = normalize_response({"item_id": "i2", "axis_ratings": ratings}, item_lookup)

    assert abs(basis_a["basis_score"] - expected) < 1e-9
    assert basis_a["winner"] == SOURCE_BASIS
    assert abs(random_a["basis_score"] + expected) < 1e-9
    assert random_a["winner"] == SOURCE_RANDOM


def test_write_analysis_outputs_merges_multiple_jsonl_without_names(tmp_path: Path):
    items = [
        {
            "item_id": "i1",
            "category": "cat",
            "stratum": STRATUM_BASIS_WIN,
            "prompt": "prompt 1",
            "model_a_source": SOURCE_BASIS,
            "model_b_source": SOURCE_RANDOM,
            "displayed_order": "basis_a_random_b",
            "score_gap": 10.0,
        },
        {
            "item_id": "i2",
            "category": "cat",
            "stratum": STRATUM_RANDOM_WIN,
            "prompt": "prompt 2",
            "model_a_source": SOURCE_RANDOM,
            "model_b_source": SOURCE_BASIS,
            "displayed_order": "random_a_basis_b",
            "score_gap": -10.0,
        },
    ]
    item_path = tmp_path / "items.jsonl"
    write_jsonl(item_path, items)
    response_1 = tmp_path / "p1_s1.jsonl"
    response_2 = tmp_path / "p2_s2.jsonl"
    write_jsonl(
        response_1,
        [
            {
                "participant_name": "研究 太郎",
                "participant_id": "p1",
                "session_id": "s1",
                "item_id": "i1",
                "axis_ratings": axis_ratings(**{axis_key: 1 for axis_key in AXIS_KEYS}),
                "comment": "良い",
            }
        ],
    )
    write_jsonl(
        response_2,
        [
            {
                "participant_name": "研究 花子",
                "participant_id": "p2",
                "session_id": "s2",
                "item_id": "i2",
                "axis_ratings": axis_ratings(**{axis_key: 1 for axis_key in AXIS_KEYS}),
                "comment": "",
            }
        ],
    )
    item_lookup = {item["item_id"]: item for item in items}
    rows = load_and_normalize_responses([response_1, response_2], item_lookup)
    summary = summarize_rows(rows)

    assert summary["basis_win_count"] == 1
    assert summary["random_win_count"] == 1
    assert summary["tie_count"] == 0

    output_dir = tmp_path / "results"
    write_analysis_outputs(
        rows=rows,
        item_lookup=item_lookup,
        response_files=[response_1, response_2],
        items_path=item_path,
        output_dir=output_dir,
    )

    normalized_path = output_dir / "normalized_responses.csv"
    with normalized_path.open(encoding="utf-8") as file:
        header = next(csv.reader(file))
    assert "participant_name" not in header
    assert "emotion_reception_rating_raw" in header
    assert "overall_basis_score" in header
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "figures" / "win_rate_bar.png").exists()
    assert (output_dir / "figures" / "win_rate_bar.svg").exists()
    assert (output_dir / "figures" / "axis_mean_scores.png").exists()
    assert (output_dir / "figures" / "axis_mean_scores.svg").exists()


def test_upsert_answer_replaces_existing_item_answer(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    first = {
        "participant_id": "p1",
        "session_id": "s1",
        "item_id": "i1",
        "axis_ratings": axis_ratings(**{axis_key: 1 for axis_key in AXIS_KEYS}),
        "comment": "first",
        "timestamp": "2026-06-19T00:00:00+00:00",
    }
    second = {
        "participant_id": "p1",
        "session_id": "s1",
        "item_id": "i1",
        "axis_ratings": axis_ratings(**{axis_key: 5 for axis_key in AXIS_KEYS}),
        "comment": "updated",
        "timestamp": "2026-06-19T00:01:00+00:00",
    }

    first_result = upsert_answer(first, path)
    second_result = upsert_answer(second, path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_item = load_answer_records_by_item_id(path)

    assert first_result == {"updated": False, "revision": 1}
    assert second_result == {"updated": True, "revision": 2}
    assert len(records) == 1
    assert records[0]["axis_ratings"] == axis_ratings(**{axis_key: 5 for axis_key in AXIS_KEYS})
    assert records[0]["comment"] == "updated"
    assert records[0]["revision"] == 2
    assert records[0]["created_at"] == "2026-06-19T00:00:00+00:00"
    assert records[0]["updated_at"] == "2026-06-19T00:01:00+00:00"
    assert by_item["i1"]["axis_ratings"] == axis_ratings(**{axis_key: 5 for axis_key in AXIS_KEYS})


def test_load_and_normalize_responses_uses_latest_duplicate_answer(tmp_path: Path):
    item_lookup = {
        "i1": {
            "item_id": "i1",
            "model_a_source": SOURCE_BASIS,
            "model_b_source": SOURCE_RANDOM,
            "displayed_order": "basis_a_random_b",
        }
    }
    response_path = tmp_path / "answers.jsonl"
    write_jsonl(
        response_path,
        [
            {
                "participant_id": "p1",
                "session_id": "s1",
                "item_id": "i1",
                "axis_ratings": axis_ratings(**{axis_key: 1 for axis_key in AXIS_KEYS}),
                "timestamp": "2026-06-19T00:00:00+00:00",
            },
            {
                "participant_id": "p1",
                "session_id": "s1",
                "item_id": "i1",
                "axis_ratings": axis_ratings(**{axis_key: 5 for axis_key in AXIS_KEYS}),
                "timestamp": "2026-06-19T00:01:00+00:00",
            },
        ],
    )

    rows = load_and_normalize_responses([response_path], item_lookup)

    assert len(rows) == 1
    assert rows[0]["rating_raw"] == ""
    assert rows[0]["emotion_reception_rating_raw"] == 5
    assert rows[0]["basis_score"] == -2
    assert rows[0]["winner"] == SOURCE_RANDOM
