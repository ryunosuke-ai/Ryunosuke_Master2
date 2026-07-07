"""4カテゴリOracle評価共通処理の軽量テスト。"""

from __future__ import annotations

import json
from pathlib import Path

from core.oracle_eval_common import (
    EvaluationSpec,
    RubricAxis,
    load_eval_samples,
    parse_score_payload,
    parse_strategy_payload,
    summarize_score_records,
    summarize_strategy_records,
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
