"""新研究パイプライン用CLI補助関数のテスト。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.generated_bayes_model import parse_bayes_model
from tools.analyze_small_corpus import (
    build_analysis_instructions,
    build_corpus_text,
    build_json_mode_input as build_analysis_json_mode_input,
    extract_json_object,
    extract_response_text,
    generate_bayes_model,
    read_jsonl,
    resolve_analysis_azure_api_key,
    resolve_analysis_azure_endpoint,
    resolve_analysis_model,
    summarize_corpus,
)
from tools.build_dpo_from_bayes_scores import build_preference_records
from tools.score_dialogue_with_bayes_model import (
    build_json_mode_input as build_scoring_json_mode_input,
    parse_observation_score,
    resolve_scoring_azure_api_key,
    resolve_scoring_azure_endpoint,
    resolve_scoring_model,
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


def make_bayes_payload():
    """テスト用ベイズモデルJSONを返す。"""
    return {
        "name": "target_style_model",
        "positive_state": "target_style",
        "negative_state": "non_target_style",
        "observations": ["deepening", "generic", "blocking"],
        "likelihoods": {
            "target_style": {
                "deepening": 0.7,
                "generic": 0.2,
                "blocking": 0.1,
            },
            "non_target_style": {
                "deepening": 0.1,
                "generic": 0.3,
                "blocking": 0.6,
            },
        },
        "prior": 0.5,
        "strategy_descriptions": {
            "deepening": "相手の発話内容を拾って自然に深める",
            "generic": "一般論に戻す",
            "blocking": "会話の継続を妨げる",
        },
    }


def test_read_jsonl_and_summarize_small_corpus(tmp_path: Path):
    path = tmp_path / "small.jsonl"
    rows = [
        {"conversation_id": "c1", "turn_index": 1, "speaker": "user", "text": "旅行が好きです。"},
        {"conversation_id": "c1", "turn_index": 2, "speaker": "assistant", "text": "どんな場所が印象的でしたか。"},
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    records = read_jsonl(path)
    summary = summarize_corpus(records)

    assert summary["records"] == 2
    assert summary["conversations"] == 1
    assert "assistant: どんな場所" in build_corpus_text(records)


def test_generate_bayes_model_extracts_json_from_stub():
    payload = make_bayes_payload()
    generator = StubGenerator([f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"])
    records = [{"conversation_id": "c1", "turn_index": 1, "speaker": "user", "text": "昔の話です。"}]

    result = generate_bayes_model(records, generator=generator, model="gpt-5.4-pro", max_output_tokens=1024)

    assert result["name"] == "target_style_model"
    assert generator.calls[0]["model"] == "gpt-5.4-pro"


def test_analysis_instructions_infer_dataset_purpose_without_fixed_domain():
    instructions = build_analysis_instructions()

    assert "大量" in instructions
    assert "prompt/response" in instructions
    assert "posterior" in instructions
    assert "DPO" in instructions
    assert "そのまま使わず" in instructions
    assert "必ずコーパス分析に基づいて" in instructions
    assert "回想法" not in instructions


def test_generate_bayes_model_validates_generated_payload():
    payload = make_bayes_payload()
    payload["likelihoods"]["target_style"]["deepening"] = 0.2
    generator = StubGenerator([json.dumps(payload, ensure_ascii=False)])
    records = [{"conversation_id": "c1", "turn_index": 1, "speaker": "user", "text": "昔の話です。"}]

    with pytest.raises(ValueError, match="合計"):
        generate_bayes_model(records, generator=generator, model="gpt-5.4-pro", max_output_tokens=1024)


def test_analysis_env_resolution_prefers_gpt54_pro_specific_values(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "generic-key")
    monkeypatch.setenv("AZURE_OPENAI_GPT54_PRO_API_KEY", "pro-key")
    monkeypatch.setenv("AZURE_OPENAI_GPT54_PRO_DEPLOYMENT_NAME", "pro-deployment")
    monkeypatch.delenv("ANALYSIS_LLM_MODEL", raising=False)

    assert resolve_analysis_azure_api_key() == "pro-key"
    assert resolve_analysis_model() == "pro-deployment"


def test_scoring_env_resolution_prefers_gpt54_specific_values(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "generic-key")
    monkeypatch.setenv("AZURE_OPENAI_GPT54_API_KEY", "base-key")
    monkeypatch.setenv("AZURE_OPENAI_GPT54_DEPLOYMENT_NAME", "base-deployment")
    monkeypatch.delenv("SCORING_LLM_MODEL", raising=False)

    assert resolve_scoring_azure_api_key() == "base-key"
    assert resolve_scoring_model() == "base-deployment"


def test_gpt56_azure_resolution_uses_shared_key_and_generation_endpoint(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_GPT56_API_KEY", "gpt56-key")
    monkeypatch.setenv("AZURE_OPENAI_GPT56_ENDPOINT", "https://gpt56.example")
    monkeypatch.setenv("AZURE_OPENAI_GPT56_SOL_DEPLOYMENT", "sol-deployment")
    monkeypatch.setenv("AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT", "terra-deployment")
    monkeypatch.setenv("AZURE_OPENAI_GPT54_PRO_API_KEY", "legacy-pro-key")
    monkeypatch.setenv("AZURE_OPENAI_GPT54_API_KEY", "legacy-score-key")

    assert resolve_analysis_azure_api_key("sol-deployment") == "gpt56-key"
    assert resolve_analysis_azure_endpoint("sol-deployment") == "https://gpt56.example"
    assert resolve_scoring_azure_api_key("terra-deployment") == "gpt56-key"
    assert resolve_scoring_azure_endpoint("terra-deployment") == "https://gpt56.example"


def test_extract_json_object_handles_surrounding_text():
    result = extract_json_object('前置き {"name": "x", "prior": 0.5} 後置き')

    assert result == {"name": "x", "prior": 0.5}


def test_extract_response_text_falls_back_to_output_content():
    response = SimpleNamespace(
        output_text="",
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text='{"name": "x"}'),
                ],
            )
        ],
        status="completed",
    )

    assert extract_response_text(response) == '{"name": "x"}'


def test_extract_response_text_reports_incomplete_reason():
    response = SimpleNamespace(
        output_text="",
        output=[],
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )

    with pytest.raises(RuntimeError, match="max_output_tokens"):
        extract_response_text(response)


def test_json_mode_input_contains_json_keyword():
    assert "json" in build_analysis_json_mode_input("本文").lower()
    assert "json" in build_scoring_json_mode_input("本文").lower()


def test_parse_observation_score_accepts_known_observation():
    model = parse_bayes_model(make_bayes_payload())

    score = parse_observation_score({"observation": "deepening", "score": 1.2, "reason": "自然"}, model)

    assert score.observation == "deepening"
    assert score.score == 1.0


def test_score_records_updates_prior_per_conversation():
    model = parse_bayes_model(make_bayes_payload())
    generator = StubGenerator(
        [
            json.dumps({"observation": "deepening", "score": 0.9, "reason": "深めている"}, ensure_ascii=False),
            json.dumps({"observation": "blocking", "score": 0.8, "reason": "止めている"}, ensure_ascii=False),
        ]
    )
    records = [
        {"conversation_id": "c1", "turn_index": 1, "prompt": "旅行は好きですか", "response": "昔の旅の話を聞かせてください。"},
        {"conversation_id": "c1", "turn_index": 2, "prompt": "旅行は好きですか", "response": "それは普通ですね。"},
    ]

    scored = score_records(records, bayes_model=model, generator=generator, model="gpt-5.4", max_output_tokens=256)

    assert scored[0]["posterior"] > 0.5
    assert scored[1]["prior"] == scored[0]["posterior"]
    assert scored[1]["posterior"] < scored[1]["prior"]


def test_build_preference_records_pairs_same_prompt():
    scored = [
        {
            "conversation_id": "c1",
            "turn_index": 1,
            "prompt": "最近どうですか",
            "response": "その話をもう少し聞かせてください。",
            "posterior": 0.8,
            "observation": "deepening",
        },
        {
            "conversation_id": "c2",
            "turn_index": 1,
            "prompt": "最近どうですか",
            "response": "そういうこともありますね。",
            "posterior": 0.2,
            "observation": "generic",
        },
    ]

    preferences = build_preference_records(scored, min_chosen_posterior=0.65, max_rejected_posterior=0.35)

    assert len(preferences) == 1
    assert preferences[0]["chosen"] == "その話をもう少し聞かせてください。"
    assert preferences[0]["rejected"] == "そういうこともありますね。"
