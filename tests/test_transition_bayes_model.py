"""状態遷移ベイズモデルの生成・検証・スコアリングのテスト。"""

import json

import pytest

from core.transition_bayes_model import (
    TransitionObservationScore,
    parse_transition_bayes_model,
    score_transition_observation,
    update_state_distribution,
)
from tools.analyze_small_corpus_transition_bayes import (
    build_transition_analysis_instructions,
    generate_transition_bayes_model,
)
from tools.score_dialogue_with_transition_bayes_model import (
    build_transition_scoring_instructions,
    parse_transition_observation_score,
    score_records,
)


class StubGenerator:
    """LLM呼び出しを置き換えるテスト用生成器。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


def make_transition_payload():
    """テスト用の状態遷移ベイズモデルJSONを返す。"""
    return {
        "name": "transition_dialogue_model",
        "model_type": "transition_bayes_network",
        "states": ["opening", "deepening", "closing", "off_style"],
        "positive_states": ["deepening", "closing"],
        "negative_states": ["off_style"],
        "observations": ["followup", "reflection", "generic_shift"],
        "initial_state_prior": {
            "opening": 0.55,
            "deepening": 0.25,
            "closing": 0.10,
            "off_style": 0.10,
        },
        "transition_likelihoods": {
            "opening": {
                "opening": 0.10,
                "deepening": 0.65,
                "closing": 0.15,
                "off_style": 0.10,
            },
            "deepening": {
                "opening": 0.05,
                "deepening": 0.60,
                "closing": 0.25,
                "off_style": 0.10,
            },
            "closing": {
                "opening": 0.10,
                "deepening": 0.25,
                "closing": 0.55,
                "off_style": 0.10,
            },
            "off_style": {
                "opening": 0.15,
                "deepening": 0.15,
                "closing": 0.10,
                "off_style": 0.60,
            },
        },
        "emission_likelihoods": {
            "opening": {
                "followup": 0.45,
                "reflection": 0.20,
                "generic_shift": 0.35,
            },
            "deepening": {
                "followup": 0.65,
                "reflection": 0.25,
                "generic_shift": 0.10,
            },
            "closing": {
                "followup": 0.20,
                "reflection": 0.65,
                "generic_shift": 0.15,
            },
            "off_style": {
                "followup": 0.10,
                "reflection": 0.15,
                "generic_shift": 0.75,
            },
        },
        "state_descriptions": {
            "opening": "会話の導入状態。",
            "deepening": "文脈を踏まえて深める状態。",
            "closing": "温かくまとめる状態。",
            "off_style": "望ましい進行から外れる状態。",
        },
        "observation_descriptions": {
            "followup": "具体的な追加質問で深めている。",
            "reflection": "温かく要約している。",
            "generic_shift": "一般論や別方向へ移っている。",
        },
        "dataset_hypothesis": "相手の話を受け止め、文脈に沿って深める会話を重視している。",
    }


def test_parse_transition_bayes_model_accepts_valid_payload():
    model = parse_transition_bayes_model(make_transition_payload())

    assert model.model_type == "transition_bayes_network"
    assert model.states == ("opening", "deepening", "closing", "off_style")
    assert model.positive_states == ("deepening", "closing")


def test_parse_transition_bayes_model_rejects_bad_transition_sum():
    payload = make_transition_payload()
    payload["transition_likelihoods"]["opening"]["deepening"] = 0.20

    with pytest.raises(ValueError, match="transition_likelihoods.opening"):
        parse_transition_bayes_model(payload)


def test_parse_transition_bayes_model_rejects_unknown_positive_state():
    payload = make_transition_payload()
    payload["positive_states"] = ["deepening", "unknown"]

    with pytest.raises(ValueError, match="positive_states"):
        parse_transition_bayes_model(payload)


def test_update_state_distribution_uses_transition_and_emission():
    model = parse_transition_bayes_model(make_transition_payload())

    posterior = update_state_distribution(model, None, "followup")

    assert posterior["deepening"] > posterior["off_style"]


def test_score_transition_observation_returns_positive_posterior():
    model = parse_transition_bayes_model(make_transition_payload())

    result = score_transition_observation(
        model,
        TransitionObservationScore("followup", 0.9, "文脈を深めている"),
    )

    assert result["posterior"] > result["prior"]
    assert result["most_likely_state"] in model.states
    assert "state_posteriors" in result


def test_transition_analysis_prompt_keeps_dataset_type_inferred():
    instructions = build_transition_analysis_instructions()

    assert "状態遷移" in instructions
    assert "transition_likelihoods" in instructions
    assert "emission_likelihoods" in instructions
    assert "低頻度" in instructions
    assert "観測ラベルは応答戦略" in instructions
    assert "状態ラベルは会話の局面" in instructions
    assert "そのまま使わず" in instructions
    assert "回想法" not in instructions


def test_transition_scoring_prompt_explains_context_and_negative_labels():
    model = parse_transition_bayes_model(make_transition_payload())

    instructions = build_transition_scoring_instructions(model)

    assert "直前までの会話文脈" in instructions
    assert "状態遷移ベイズモデルの観測" in instructions
    assert "negative/off_style" in instructions
    assert "確信度" in instructions


def test_generate_transition_bayes_model_validates_stub_output():
    payload = make_transition_payload()
    generator = StubGenerator([json.dumps(payload, ensure_ascii=False)])
    records = [{"conversation_id": "c1", "turn_index": 1, "speaker": "user", "text": "昔の話です。"}]

    result = generate_transition_bayes_model(
        records,
        generator=generator,
        model="gpt-5.4-pro",
        max_output_tokens=1024,
    )

    assert result["name"] == "transition_dialogue_model"
    assert generator.calls[0]["model"] == "gpt-5.4-pro"
    assert generator.calls[0]["response_text_format"] == {"type": "json_object"}


def test_generate_transition_bayes_model_repairs_invalid_json_once():
    payload = make_transition_payload()
    generator = StubGenerator(
        [
            '{"name": "broken"',
            json.dumps(payload, ensure_ascii=False),
        ]
    )
    records = [{"conversation_id": "c1", "turn_index": 1, "speaker": "user", "text": "昔の話です。"}]

    result = generate_transition_bayes_model(
        records,
        generator=generator,
        model="gpt-5.4-pro",
        max_output_tokens=1024,
    )

    assert result["name"] == "transition_dialogue_model"
    assert len(generator.calls) == 2
    assert "JSON修復専用" in generator.calls[1]["instructions"]


def test_parse_transition_observation_score_accepts_known_observation():
    model = parse_transition_bayes_model(make_transition_payload())

    score = parse_transition_observation_score({"observation": "followup", "score": 1.2}, model)

    assert score.observation == "followup"
    assert score.score == 1.0


def test_transition_score_records_updates_distribution_per_conversation():
    model = parse_transition_bayes_model(make_transition_payload())
    generator = StubGenerator(
        [
            json.dumps({"observation": "followup", "score": 0.9, "reason": "深めている"}, ensure_ascii=False),
            json.dumps({"observation": "generic_shift", "score": 0.8, "reason": "一般論"}, ensure_ascii=False),
        ]
    )
    records = [
        {"conversation_id": "c1", "turn_index": 1, "prompt": "旅行の話", "response": "どこが印象的でしたか。"},
        {"conversation_id": "c1", "turn_index": 2, "prompt": "旅行の話", "response": "そういうこともありますね。"},
    ]

    scored = score_records(records, bayes_model=model, generator=generator, model="gpt-5.4", max_output_tokens=256)

    assert scored[0]["posterior"] > scored[0]["prior"]
    assert scored[1]["posterior"] < scored[0]["posterior"]
    assert scored[1]["state_posteriors"] != scored[0]["state_posteriors"]
