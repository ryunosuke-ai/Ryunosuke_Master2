"""Oracle評価パイプラインの軽量テスト。"""

import json
from pathlib import Path

import pytest

from core.transition_bayes_model import parse_transition_bayes_model
from tools.run_oracle_evaluation import (
    build_judge_instructions,
    build_reference_instructions,
    model_order_for_prompt,
    parse_judge_payload,
    parse_reference_payload,
    read_evaluation_prompts,
    summarize_judgments,
)


def make_transition_payload():
    """テスト用の状態遷移ベイズモデルJSONを返す。"""
    return {
        "name": "transition_dialogue_model",
        "model_type": "transition_bayes_network",
        "states": ["opening", "deepening", "off_style"],
        "positive_states": ["deepening"],
        "negative_states": ["off_style"],
        "observations": ["followup", "reflection", "generic_shift"],
        "initial_state_prior": {
            "opening": 0.60,
            "deepening": 0.30,
            "off_style": 0.10,
        },
        "transition_likelihoods": {
            "opening": {"opening": 0.20, "deepening": 0.65, "off_style": 0.15},
            "deepening": {"opening": 0.10, "deepening": 0.75, "off_style": 0.15},
            "off_style": {"opening": 0.20, "deepening": 0.20, "off_style": 0.60},
        },
        "emission_likelihoods": {
            "opening": {"followup": 0.50, "reflection": 0.25, "generic_shift": 0.25},
            "deepening": {"followup": 0.75, "reflection": 0.20, "generic_shift": 0.05},
            "off_style": {"followup": 0.10, "reflection": 0.20, "generic_shift": 0.70},
        },
        "state_descriptions": {
            "opening": "会話の導入状態。",
            "deepening": "文脈を踏まえて深める状態。",
            "off_style": "望ましい進行から外れる状態。",
        },
        "observation_descriptions": {
            "followup": "具体的な追加質問で深めている。",
            "reflection": "温かく受け止めている。",
            "generic_shift": "一般論や別方向へ移っている。",
        },
        "dataset_hypothesis": "相手の話を受け止め、文脈に沿って深める会話を重視している。",
    }


def test_oracle_instructions_include_bayes_model_and_small_corpus():
    model = parse_transition_bayes_model(make_transition_payload())
    small_corpus_text = "# conversation_id=c1\nuser: 昔、京都に行きました。\nassistant: その時の景色で覚えているものはありますか。"

    reference = build_reference_instructions(model, small_corpus_text=small_corpus_text)
    judge = build_judge_instructions(model, small_corpus_text=small_corpus_text)

    assert "100点満点の正解応答" in reference
    assert "小コーパス本文抜粋" in reference
    assert "昔、京都に行きました" in reference
    assert "oracle_response" in reference
    assert "oracle_responseを100点満点の正解応答" in judge
    assert "観測ラベル・応答戦略" in judge
    assert "昔、京都に行きました" in judge


def test_read_evaluation_prompts_validates_unique_ids(tmp_path: Path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "p1", "category": "memory", "prompt": "昔の旅行を思い出しました。"}, ensure_ascii=False),
                json.dumps({"id": "p1", "category": "memory", "prompt": "昔の食事を思い出しました。"}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="重複"):
        read_evaluation_prompts(path)


def test_parse_reference_payload_requires_response():
    with pytest.raises(ValueError, match="oracle_response"):
        parse_reference_payload({"oracle_response": ""})


def test_parse_judge_payload_maps_scores_and_winner():
    payload = {
        "score_a": 110,
        "score_b": 72,
        "winner": "response_a",
        "rubric_scores": {
            "context_understanding": 90,
            "concrete_pickup": 88,
            "experiential_deepening": 95,
            "emotion_and_scene": 80,
            "conversation_continuity": 91,
            "avoids_generic_advice": 86,
            "japanese_naturalness": 94,
        },
        "reason": "response_aの方が具体的に深めている。",
    }

    result = parse_judge_payload(payload)

    assert result["score_a"] == 100.0
    assert result["score_b"] == 72.0
    assert result["winner"] == "response_a"
    assert result["rubric_scores"]["experiential_deepening"] == 95.0


def test_model_order_for_prompt_is_deterministic_and_varies_by_seed():
    first = model_order_for_prompt("eval_001", seed=42)
    second = model_order_for_prompt("eval_001", seed=42)
    other_seed = model_order_for_prompt("eval_001", seed=43)

    assert first == second
    assert set(first) == {"base", "dpo"}
    assert set(other_seed) == {"base", "dpo"}


def test_summarize_judgments_reports_dpo_gap_and_category():
    judgments = [
        {"category": "memory", "score_base": 60, "score_dpo": 85, "score_gap": 25, "winner": "dpo"},
        {"category": "memory", "score_base": 70, "score_dpo": 70, "score_gap": 0, "winner": "tie"},
        {"category": "control", "score_base": 80, "score_dpo": 75, "score_gap": -5, "winner": "base"},
    ]

    summary = summarize_judgments(judgments)

    assert summary["records"] == 3
    assert summary["mean_score_gap"] == pytest.approx(20 / 3)
    assert summary["dpo_win_rate"] == pytest.approx(1 / 3)
    assert summary["by_category"]["memory"]["count"] == 2
