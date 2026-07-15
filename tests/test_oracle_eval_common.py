"""4カテゴリOracle評価共通処理の軽量テスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.oracle_eval_common import (
    EvaluationSpec,
    RubricAxis,
    StrategyEvaluationSpec,
    assert_resume_compatible,
    load_eval_samples,
    pairwise_winrate_rows,
    parse_score_payload,
    parse_strategy_payload,
    strategy_pairwise_rows,
    summarize_axis_records,
    summarize_score_records,
    summarize_strategy_records,
    write_metadata,
)


def make_spec() -> EvaluationSpec:
    """テスト用の単独採点仕様を返す。"""
    return EvaluationSpec(
        category_key="test_category",
        category_title="テスト評価",
        output_subdir="test",
        prompt_version="test.v1",
        reference_note="test",
        axes=(
            RubricAxis(
                key="axis_a",
                title="Axis A",
                description="Aを評価する。",
                high="よい。",
                low="悪い。",
            ),
            RubricAxis(
                key="axis_b",
                title="Axis B",
                description="Bを評価する。",
                high="よい。",
                low="悪い。",
            ),
        ),
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """JSONLを書き出す。"""
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_eval_samples_expands_existing_response_fields(tmp_path: Path):
    path = tmp_path / "responses.jsonl"
    write_jsonl(
        path,
        [
            {
                "prompt_id": "p1",
                "category": "emotion",
                "history": [{"speaker": "User", "text": "不安です。"}],
                "prompt": "明日が怖いです。",
                "base_response": "大丈夫ですよ。",
                "dpo_response": "明日が怖いほど不安が強いのですね。",
                "base_field_label": "bayes_dpo",
                "dpo_field_label": "random_dpo",
            }
        ],
    )

    samples = load_eval_samples(path)

    assert [sample.model_name for sample in samples] == ["bayes_dpo", "random_dpo"]
    assert samples[0].sample_id == "p1"
    assert samples[0].history == ({"speaker": "User", "text": "不安です。"},)
    assert samples[1].response == "明日が怖いほど不安が強いのですね。"


def test_load_eval_samples_expands_three_model_wide_fields(tmp_path: Path):
    path = tmp_path / "three_model_responses.jsonl"
    write_jsonl(
        path,
        [
            {
                "prompt_id": "p1",
                "category": "emotion",
                "history": [{"speaker": "User", "text": "不安です。"}],
                "prompt": "明日が怖いです。",
                "base_response": "落ち着いてください。",
                "bayes_dpo_response": "明日が怖いほど不安が強いのですね。",
                "random_dpo_response": "その不安について詳しく教えてください。",
            }
        ],
    )

    samples = load_eval_samples(path)

    assert [sample.model_name for sample in samples] == ["base", "bayes_dpo", "random_dpo"]
    assert [sample.response for sample in samples] == [
        "落ち着いてください。",
        "明日が怖いほど不安が強いのですね。",
        "その不安について詳しく教えてください。",
    ]


def test_pairwise_winrate_rows_outputs_three_comparisons():
    records = [
        {"sample_id": "p1", "model_name": "base", "overall_score": 3.0},
        {"sample_id": "p1", "model_name": "bayes_dpo", "overall_score": 5.0},
        {"sample_id": "p1", "model_name": "random_dpo", "overall_score": 2.0},
    ]

    rows = pairwise_winrate_rows(records)

    assert [row["comparison"] for row in rows] == [
        "BASiS_vs_Base",
        "BASiS_vs_Random",
        "Base_vs_Random",
    ]
    assert all(row["count"] == 1 for row in rows)
    assert rows[0]["wins"] == 1
    assert rows[1]["wins"] == 1
    assert rows[2]["wins"] == 1


def test_pairwise_winrate_rows_uses_custom_tie_threshold():
    records = [
        {"sample_id": "p1", "model_name": "base", "overall_score": 7.0},
        {"sample_id": "p1", "model_name": "bayes_dpo", "overall_score": 7.2},
        {"sample_id": "p1", "model_name": "random_dpo", "overall_score": 6.5},
    ]

    rows = pairwise_winrate_rows(records, threshold=0.25)

    assert rows[0]["comparison"] == "BASiS_vs_Base"
    assert rows[0]["ties"] == 1
    assert rows[0]["threshold"] == 0.25


def test_parse_score_payload_and_summary():
    spec = make_spec()
    sample = load_eval_samples(
        Path("configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl"),
        limit=1,
        allow_dry_placeholder=True,
    )[0]
    record = parse_score_payload(
        {
            "scores": {"axis_a": 4, "axis_b": 5},
            "overall_score": 4.5,
            "reason": "良い。",
        },
        spec,
        sample,
    )

    rows = summarize_score_records([record], spec, seed=1)

    assert record["scores"] == {"axis_a": 4, "axis_b": 5}
    assert rows[0]["model_name"] == "dry_run_model"
    assert rows[0]["overall_score_mean"] == 4.5
    assert rows[0]["axis_a_mean"] == 4.0

    axis_rows = summarize_axis_records([record], spec, seed=1)
    assert axis_rows[0] == {
        "model_name": "dry_run_model",
        "axis": "axis_a",
        "count": 1,
        "mean": 4.0,
        "std": 0.0,
        "ci95_low": 4.0,
        "ci95_high": 4.0,
    }


def test_parse_score_payload_accepts_10_point_integer_scores():
    spec = make_spec()
    sample = load_eval_samples(
        Path("configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl"),
        limit=1,
        allow_dry_placeholder=True,
    )[0]

    record = parse_score_payload(
        {
            "scores": {"axis_a": 8, "axis_b": 9},
            "overall_score": 10,
            "reason": "10段階評価として妥当。",
        },
        spec,
        sample,
        score_scale=10,
    )

    assert record["scores"] == {"axis_a": 8, "axis_b": 9}
    assert record["overall_score"] == 8.5
    assert record["score_scale"] == 10


def test_parse_score_payload_rejects_invalid_10_point_scores():
    spec = make_spec()
    sample = load_eval_samples(
        Path("configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl"),
        limit=1,
        allow_dry_placeholder=True,
    )[0]

    with pytest.raises(ValueError):
        parse_score_payload(
            {
                "scores": {"axis_a": 8.5, "axis_b": 11},
                "overall_score": 8,
                "reason": "不正。",
            },
            spec,
            sample,
            score_scale=10,
        )


def test_strategy_payload_and_distribution_summary():
    sample = load_eval_samples(
        Path("configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl"),
        limit=1,
        allow_dry_placeholder=True,
    )[0]
    record = parse_strategy_payload(
        {
            "labels": {
                "predicted_user_state_before_response": "emotional_disclosure",
                "response_strategy": "empathy_validation",
                "predicted_user_state_after_response": "feeling_organized",
                "transition_type": "emotional_disclosure -> empathy_validation -> feeling_organized",
                "ideal_strategy_for_context": "empathy_validation",
            },
            "scores": {
                "strategy_appropriateness_score": 5,
                "transition_smoothness_score": 4,
            },
            "reason": "文脈に合う。",
        },
        sample,
    )

    rows, reference_source = summarize_strategy_records([record])

    assert reference_source == "oracle_derived_ideal_strategy_pseudo_reference"
    assert rows[0]["strategy_accuracy"] == 1.0
    assert rows[0]["strategy_macro_f1"] > 0.0
    assert rows[0]["most_frequent_strategy"] == "empathy_validation"


def test_strategy_payload_accepts_10_point_scores():
    sample = load_eval_samples(
        Path("configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl"),
        limit=1,
        allow_dry_placeholder=True,
    )[0]

    record = parse_strategy_payload(
        {
            "labels": {
                "predicted_user_state_before_response": "emotional_disclosure",
                "response_strategy": "empathy_validation",
                "predicted_user_state_after_response": "feeling_organized",
                "transition_type": "emotional_disclosure -> empathy_validation -> feeling_organized",
                "ideal_strategy_for_context": "empathy_validation",
            },
            "scores": {
                "strategy_appropriateness_score": 9,
                "transition_smoothness_score": 8,
            },
            "reason": "10段階評価として文脈に合う。",
        },
        sample,
        score_scale=10,
    )

    assert record["scores"]["strategy_appropriateness_score"] == 9
    assert record["scores"]["transition_smoothness_score"] == 8
    assert record["score_scale"] == 10


def test_custom_strategy_spec_accepts_v2_axes_and_summary():
    sample = load_eval_samples(
        Path("configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl"),
        limit=1,
        allow_dry_placeholder=True,
    )[0]
    spec = StrategyEvaluationSpec(
        category_key="strategy_transition_esconv_v2",
        category_title="ESConv支援過程評価v2",
        output_subdir="oracle_strategy_transition_esconv_v2",
        prompt_version="oracle_strategy_transition_esconv_v2.v1",
        reference_note="test",
        score_axes=(
            RubricAxis(
                key="strategy_stage_alignment",
                title="stage",
                description="stage",
                high="high",
                low="low",
            ),
            RubricAxis(
                key="premature_advice_avoidance",
                title="advice",
                description="advice",
                high="high",
                low="low",
            ),
            RubricAxis(
                key="esconv_transition_plausibility",
                title="transition",
                description="transition",
                high="high",
                low="low",
            ),
        ),
    )
    record = parse_strategy_payload(
        {
            "labels": {
                "predicted_user_state_before_response": "emotional_disclosure",
                "response_strategy": "empathy_validation",
                "predicted_user_state_after_response": "feeling_organized",
                "transition_type": "emotional_disclosure -> empathy_validation -> feeling_organized",
                "ideal_strategy_for_context": "empathy_validation",
            },
            "scores": {
                "strategy_stage_alignment": 8,
                "premature_advice_avoidance": 9,
                "esconv_transition_plausibility": 7,
            },
            "reason": "ESConv支援過程に合う。",
        },
        sample,
        score_scale=10,
        spec=spec,
    )

    rows, _ = summarize_strategy_records([record], spec=spec)
    pairwise = strategy_pairwise_rows(
        [
            {**record, "model_name": "base", "overall_score": 7.0},
            {**record, "model_name": "bayes_dpo", "overall_score": 8.0},
            {**record, "model_name": "random_dpo", "overall_score": 6.0},
        ],
        threshold=0.25,
        spec=spec,
    )

    assert record["overall_score"] == 8.0
    assert record["oracle_prompt_version"] == "oracle_strategy_transition_esconv_v2.v1"
    assert rows[0]["strategy_stage_alignment_mean"] == 8.0
    assert rows[0]["premature_advice_avoidance_mean"] == 9.0
    assert pairwise[0]["comparison"] == "BASiS_vs_Base"
    assert pairwise[0]["wins"] == 1


def test_resume_compatibility_rejects_score_scale_mismatch(tmp_path: Path):
    output_dir = tmp_path / "oracle_tst_10pt"
    write_metadata(
        path=output_dir / "metadata.json",
        spec=make_spec(),
        category_key="test_category",
        judge_model="gpt-5.4",
        input_path="input.jsonl",
        temperature=0.0,
        max_retries=5,
        dry_run=False,
        score_scale=5,
        pairwise_tie_threshold=0.1,
    )

    with pytest.raises(ValueError):
        assert_resume_compatible(
            output_dir,
            resume=True,
            judge_model="gpt-5.4",
            score_scale=10,
            score_min=1,
            score_max=10,
            dry_run=False,
        )
