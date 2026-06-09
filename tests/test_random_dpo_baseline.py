"""Random-DPO baselineの軽量テスト。"""

import json

from core.dpo_prompting import DPO_PROMPT_TEMPLATE_VERSION
from core.random_dpo_prompting import (
    GENERAL_QUALITY_STYLE_PRESET,
    RANDOM_DPO_PROMPT_TEMPLATE_VERSION,
    build_general_quality_generation_instructions,
    validate_general_quality_payload,
)
from tools.build_random_dailydialog_dpo import (
    build_random_dpo_records,
    count_by_source_dataset,
    randomize_source_records,
)


class StubGenerator:
    """LLM呼び出しを置き換えるテスト用生成器。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


def make_source_record(index: int) -> dict:
    """DailyDialog由来の入力候補を作る。"""
    return {
        "conversation_id": f"train_{index:06d}",
        "turn_index": 3,
        "prompt": "speaker_a: Hi.\nspeaker_b: Hello.",
        "response": f"How are you doing today? {index}",
        "metadata": {
            "source_dataset": "DailyDialog",
            "source_split": "train",
            "source_dialogue_index": index,
            "context_turns": 2,
        },
    }


def generation_output(prompt: str, chosen: str, rejected: str) -> str:
    """Random-DPO生成JSONを返す。"""
    return json.dumps(
        {
            "translated_prompt": prompt,
            "translated_chosen": chosen,
            "rejected_candidates": [
                rejected,
                "そうです。",
                "もう終わりにしましょう。",
                "よく分かりません。",
            ],
            "chosen_quality_score": 0.91,
        },
        ensure_ascii=False,
    )


def test_general_quality_prompt_avoids_dataset_specific_policy_terms():
    instructions = build_general_quality_generation_instructions()
    blocked_terms = [
        "ES" + "Conv",
        "支援対話",
        "感情反映",
        "共感不足",
        "早すぎる助言",
        "strategy",
        "Reflection of feelings",
    ]

    assert "一般的な雑談" in instructions
    assert "文脈を無視する" in instructions
    assert "会話を終わらせてしまう" in instructions
    for term in blocked_terms:
        assert term not in instructions


def test_randomize_source_records_is_seed_reproducible():
    records = [make_source_record(index) for index in range(10)]

    first = randomize_source_records(records, seed=42, max_source_records=5)
    second = randomize_source_records(records, seed=42, max_source_records=5)
    different = randomize_source_records(records, seed=7, max_source_records=5)

    assert [record["conversation_id"] for record in first] == [
        record["conversation_id"] for record in second
    ]
    assert [record["conversation_id"] for record in first] != [
        record["conversation_id"] for record in different
    ]
    assert len(first) == 5


def test_validate_general_quality_payload_requires_distinct_candidates():
    payload = {
        "translated_prompt": "User: こんにちは。",
        "translated_chosen": "今日は何をしていましたか。",
        "rejected_candidates": ["はい。", "はい。", "もういいです。"],
        "chosen_quality_score": 1.2,
    }

    result = validate_general_quality_payload(payload, candidates=2)

    assert result["chosen_quality_score"] == 1.0
    assert result["rejected_candidates"] == ["はい。", "もういいです。"]


def test_build_random_dpo_records_uses_shared_training_prompt_template():
    source_records = [make_source_record(1)]
    generator = StubGenerator(
        [
            generation_output(
                "speaker_a: こんにちは。\nspeaker_b: やあ。",
                "今日はどう過ごしていますか。",
                "はい。",
            )
        ]
    )

    records, skipped = build_random_dpo_records(
        source_records,
        generator=generator,
        model="gpt-5.4",
        max_output_tokens=512,
        candidates=4,
        target_records=1,
        workers=1,
        seed=42,
        skip_sample_errors=False,
    )

    assert skipped == {}
    assert len(records) == 1
    record = records[0]
    assert record["source_dataset"] == "DailyDialog"
    assert record["chosen"] == "今日はどう過ごしていますか。"
    assert record["rejected"] == "はい。"
    assert record["prompt"].startswith("以下の会話の次のAI返答を生成してください。")
    assert record["prompt"].endswith("\n\nAI:")
    assert record["dpo_prompt_template_version"] == DPO_PROMPT_TEMPLATE_VERSION
    assert record["prompt_template_version"] == RANDOM_DPO_PROMPT_TEMPLATE_VERSION
    assert record["metadata"]["style_preset"] == GENERAL_QUALITY_STYLE_PRESET
    assert record["metadata"]["selection_method"] == "random"
    assert record["metadata"]["esconv_gold_records"] == 0
    assert record["model_used_for_scoring"] == "not_used_random_baseline"
    assert count_by_source_dataset(records) == {"DailyDialog": 1}
    assert generator.calls[0]["response_text_format"] == {"type": "json_object"}
