"""ESConv専用変換・分析パイプラインのテスト。"""

from __future__ import annotations

import json
from pathlib import Path

from tools.analyze_esconv_corpus_transition_bayes import (
    build_esconv_corpus_text,
    build_esconv_transition_analysis_instructions,
    generate_esconv_transition_bayes_model,
    read_esconv_analysis_jsonl,
)
from tools.build_esconv_gold_dpo import collect_gold_candidates
from tools.prepare_esconv_for_analysis import convert_esconv_rows


class StubGenerator:
    """LLM呼び出しを置き換えるテスト用生成器。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


def make_esconv_payload(index: int = 0) -> dict:
    """テスト用ESConv text payloadを返す。"""
    return {
        "experience_type": "Current Experience",
        "emotion_type": "anxiety",
        "problem_type": "job crisis",
        "situation": f"I am worried about my job. #{index}",
        "survey_score": {
            "seeker": {
                "initial_emotion_intensity": "3",
                "empathy": "5",
                "relevance": "5",
                "final_emotion_intensity": "2",
            },
            "supporter": {"relevance": "5"},
        },
        "dialog": [
            {"text": "Hello good afternoon.", "speaker": "usr"},
            {"text": "Hi, good afternoon.", "speaker": "sys", "strategy": "Question"},
            {"text": "I'm feeling anxious about my job.", "speaker": "usr"},
            {
                "text": "It sounds really stressful to feel unsure about your job.",
                "speaker": "sys",
                "strategy": "Reflection of feelings",
            },
        ],
        "seeker_question1": "They reflected my feelings well.",
        "seeker_question2": "N.A",
        "supporter_question1": "I tried to help.",
        "supporter_question2": "",
    }


def make_transition_payload() -> dict:
    """ESConv分析テスト用の状態遷移ベイズモデルJSONを返す。"""
    return {
        "name": "esconv_transition_model",
        "model_type": "transition_bayes_network",
        "states": ["opening", "emotional_exploration", "supportive_guidance", "off_style"],
        "positive_states": ["emotional_exploration", "supportive_guidance"],
        "negative_states": ["off_style"],
        "observations": ["open_question", "feeling_reflection", "practical_suggestion", "generic_shift"],
        "initial_state_prior": {
            "opening": 0.50,
            "emotional_exploration": 0.25,
            "supportive_guidance": 0.15,
            "off_style": 0.10,
        },
        "transition_likelihoods": {
            "opening": {
                "opening": 0.20,
                "emotional_exploration": 0.55,
                "supportive_guidance": 0.15,
                "off_style": 0.10,
            },
            "emotional_exploration": {
                "opening": 0.05,
                "emotional_exploration": 0.45,
                "supportive_guidance": 0.40,
                "off_style": 0.10,
            },
            "supportive_guidance": {
                "opening": 0.05,
                "emotional_exploration": 0.20,
                "supportive_guidance": 0.65,
                "off_style": 0.10,
            },
            "off_style": {
                "opening": 0.10,
                "emotional_exploration": 0.15,
                "supportive_guidance": 0.15,
                "off_style": 0.60,
            },
        },
        "emission_likelihoods": {
            "opening": {
                "open_question": 0.55,
                "feeling_reflection": 0.20,
                "practical_suggestion": 0.15,
                "generic_shift": 0.10,
            },
            "emotional_exploration": {
                "open_question": 0.30,
                "feeling_reflection": 0.50,
                "practical_suggestion": 0.10,
                "generic_shift": 0.10,
            },
            "supportive_guidance": {
                "open_question": 0.15,
                "feeling_reflection": 0.20,
                "practical_suggestion": 0.55,
                "generic_shift": 0.10,
            },
            "off_style": {
                "open_question": 0.10,
                "feeling_reflection": 0.10,
                "practical_suggestion": 0.15,
                "generic_shift": 0.65,
            },
        },
        "state_descriptions": {
            "opening": "会話を開き、相手の状況を聞き始める状態。",
            "emotional_exploration": "相手の感情や背景を受け止めながら探索する状態。",
            "supportive_guidance": "感情を支えつつ現実的な方向づけを行う状態。",
            "off_style": "支援目的から外れ、一般論や表面的返答になる状態。",
        },
        "observation_descriptions": {
            "open_question": "相手が話しやすい質問をしている。",
            "feeling_reflection": "相手の感情を言い換えて受け止めている。",
            "practical_suggestion": "文脈に即した現実的な提案をしている。",
            "generic_shift": "文脈や感情を拾わず一般的に返している。",
        },
        "dataset_hypothesis": "不安や悩みを抱える相手を受け止め、探索と支援を行う会話を重視している。",
    }


def test_convert_esconv_rows_preserves_dialog_and_annotations():
    rows = [{"text": json.dumps(make_esconv_payload(), ensure_ascii=False)}]

    records = convert_esconv_rows(rows, split="train", max_conversations=None, seed=42)

    assert len(records) == 1
    assert records[0]["conversation_id"] == "esconv_train_000000"
    assert records[0]["emotion_type"] == "anxiety"
    assert records[0]["problem_type"] == "job crisis"
    assert records[0]["dialog"][0]["speaker"] == "user"
    assert records[0]["dialog"][1]["speaker"] == "assistant"
    assert records[0]["dialog"][1]["strategy"] == "Question"
    assert records[0]["survey_score"]["seeker"]["empathy"] == "5"


def test_convert_esconv_rows_samples_with_seed():
    rows = [
        {"text": json.dumps(make_esconv_payload(index), ensure_ascii=False)}
        for index in range(5)
    ]

    first = convert_esconv_rows(rows, split="train", max_conversations=2, seed=7)
    second = convert_esconv_rows(rows, split="train", max_conversations=2, seed=7)

    assert [record["conversation_id"] for record in first] == [record["conversation_id"] for record in second]
    assert len(first) == 2


def test_convert_esconv_rows_stratified_prefers_strategy_coverage():
    rows = []
    for index, strategy in enumerate(["Question", "Question", "Providing Suggestions", "Information"]):
        payload = make_esconv_payload(index)
        payload["dialog"][1]["strategy"] = strategy
        payload["dialog"][3]["strategy"] = strategy
        payload["emotion_type"] = f"emotion_{index}"
        rows.append({"text": json.dumps(payload, ensure_ascii=False)})

    records = convert_esconv_rows(
        rows,
        split="train",
        max_conversations=3,
        seed=3,
        sampling="stratified",
    )
    selected_strategies = {record["dialog"][1]["strategy"] for record in records}

    assert len(records) == 3
    assert "Providing Suggestions" in selected_strategies
    assert "Information" in selected_strategies


def test_read_esconv_analysis_jsonl_and_build_text(tmp_path: Path):
    rows = [{"text": json.dumps(make_esconv_payload(), ensure_ascii=False)}]
    records = convert_esconv_rows(rows, split="train", max_conversations=None, seed=42)
    path = tmp_path / "esconv.jsonl"
    path.write_text(json.dumps(records[0], ensure_ascii=False) + "\n", encoding="utf-8")

    loaded = read_esconv_analysis_jsonl(path)
    corpus_text = build_esconv_corpus_text(loaded)

    assert loaded[0]["conversation_id"] == "esconv_train_000000"
    assert "emotion_type: anxiety" in corpus_text
    assert "problem_type: job crisis" in corpus_text
    assert "annotated_strategy=Reflection of feelings" in corpus_text
    assert "survey_score:" in corpus_text


def test_esconv_analysis_prompt_uses_annotations_without_copying_strategy():
    instructions = build_esconv_transition_analysis_instructions()

    assert "emotion_type" in instructions
    assert "problem_type" in instructions
    assert "situation" in instructions
    assert "survey_score" in instructions
    assert "annotated_strategy" in instructions
    assert "そのまま観測ラベルとしてコピーしない" in instructions
    assert "会話本文と整合" in instructions
    assert "transition_likelihoods" in instructions
    assert "emission_likelihoods" in instructions
    assert "Strategy利用方針（結果重視）" in instructions
    assert "強く参照" in instructions


def test_generate_esconv_transition_bayes_model_validates_stub_output():
    rows = [{"text": json.dumps(make_esconv_payload(), ensure_ascii=False)}]
    records = convert_esconv_rows(rows, split="train", max_conversations=None, seed=42)
    payload = make_transition_payload()
    generator = StubGenerator([json.dumps(payload, ensure_ascii=False)])

    result = generate_esconv_transition_bayes_model(
        records,
        generator=generator,
        model="gpt-5.4-pro",
        max_output_tokens=1024,
    )

    assert result["name"] == "esconv_transition_model"
    assert generator.calls[0]["model"] == "gpt-5.4-pro"
    assert generator.calls[0]["response_text_format"] == {"type": "json_object"}
    assert "annotated_strategy=Question" in generator.calls[0]["input_text"]


def test_collect_esconv_gold_candidates_balances_strategies():
    rows = []
    for index, strategy in enumerate(["Question", "Reflection of feelings", "Providing Suggestions"]):
        payload = make_esconv_payload(index)
        payload["dialog"][1]["strategy"] = strategy
        payload["dialog"][3]["strategy"] = strategy
        rows.append({"text": json.dumps(payload, ensure_ascii=False)})
    records = convert_esconv_rows(rows, split="train", max_conversations=None, seed=42)

    candidates = collect_gold_candidates(records, max_context_turns=4)

    assert len(candidates) == 6
    assert {candidate["metadata"]["source_dataset"] for candidate in candidates} == {"ESConv"}
    assert candidates[0]["metadata"]["strategy"] == "Reflection of feelings"
    assert candidates[1]["metadata"]["strategy"] == "Question"
    assert candidates[2]["metadata"]["strategy"] == "Providing Suggestions"
    assert candidates[0]["prompt"].startswith("User:")
    assert candidates[0]["response"]
