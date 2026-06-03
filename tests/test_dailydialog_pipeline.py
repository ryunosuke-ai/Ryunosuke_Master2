"""DailyDialog抽出・日本語DPO化パイプラインのテスト。"""

import json
from pathlib import Path

from core.transition_bayes_model import parse_transition_bayes_model
from tools.audit_dpo_preferences import audit_records, build_audit_instructions, parse_audit_payload
from tools.compare_scoring_models import compare_scored_records, spearman_rank_correlation, top_k_overlap
from tools.extract_high_posterior_dialogues import select_high_posterior_records
from tools.prepare_dailydialog_for_scoring import build_context_prompt, convert_dailydialog_rows
from tools.translate_and_generate_dpo import (
    PROMPT_TEMPLATE_VERSION,
    build_dpo_records,
    build_translation_rejected_instructions,
    build_translation_rejected_input,
    validate_translation_payload,
)


class StubGenerator:
    """LLM呼び出しを置き換えるテスト用生成器。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def make_transition_payload():
    """DPO生成テスト用の状態遷移ベイズモデルJSONを返す。"""
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
            "opening": {
                "opening": 0.20,
                "deepening": 0.65,
                "off_style": 0.15,
            },
            "deepening": {
                "opening": 0.10,
                "deepening": 0.75,
                "off_style": 0.15,
            },
            "off_style": {
                "opening": 0.20,
                "deepening": 0.20,
                "off_style": 0.60,
            },
        },
        "emission_likelihoods": {
            "opening": {
                "followup": 0.50,
                "reflection": 0.25,
                "generic_shift": 0.25,
            },
            "deepening": {
                "followup": 0.75,
                "reflection": 0.20,
                "generic_shift": 0.05,
            },
            "off_style": {
                "followup": 0.10,
                "reflection": 0.20,
                "generic_shift": 0.70,
            },
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


def test_prepare_dailydialog_builds_context_records():
    rows = [
        {
            "dialog": [
                "Hi.",
                "Hello.",
                "I visited Kyoto long ago.",
                "What do you remember most about it?",
            ]
        }
    ]

    records = convert_dailydialog_rows(rows, split="train", max_dialogues=None, max_context_turns=2)

    assert len(records) == 3
    assert records[0]["conversation_id"] == "train_000000"
    assert records[0]["prompt"] == "speaker_a: Hi."
    assert records[0]["response"] == "Hello."
    assert records[2]["prompt"] == "speaker_b: Hello.\nspeaker_a: I visited Kyoto long ago."
    assert records[2]["metadata"]["context_turns"] == 2


def test_prepare_dailydialog_accepts_convlab_turns_format():
    rows = [
        {
            "turns": [
                {"speaker": "user", "utterance": "Say, Jim, how about going for a few beers?"},
                {"speaker": "system", "utterance": "You know that is tempting but is really not good for our fitness."},
                {"speaker": "user", "utterance": "What do you mean? It will help us relax."},
            ]
        }
    ]

    records = convert_dailydialog_rows(rows, split="train", max_dialogues=None, max_context_turns=2)

    assert len(records) == 2
    assert records[0]["prompt"] == "speaker_a: Say, Jim, how about going for a few beers?"
    assert records[0]["response"] == "You know that is tempting but is really not good for our fitness."
    assert records[1]["prompt"].endswith("speaker_b: You know that is tempting but is really not good for our fitness.")


def test_build_context_prompt_limits_previous_turns():
    prompt = build_context_prompt(["a", "b", "c", "d"], target_index=3, max_context_turns=2)

    assert prompt == "speaker_b: b\nspeaker_a: c"


def test_select_high_posterior_records_sorts_and_limits():
    records = [
        {"conversation_id": "c1", "turn_index": 1, "posterior": 0.8},
        {"conversation_id": "c1", "turn_index": 2, "posterior": 0.9},
        {"conversation_id": "c1", "turn_index": 3, "posterior": 0.4},
    ]

    selected = select_high_posterior_records(
        records,
        min_posterior=0.75,
        max_records=1,
        sort_by_posterior=True,
    )

    assert len(selected) == 1
    assert selected[0]["turn_index"] == 2


def test_select_high_posterior_records_prefers_reminiscence_labels():
    records = [
        {
            "conversation_id": "c1",
            "turn_index": 1,
            "posterior": 0.95,
            "most_likely_state": "warm_closure",
            "observation": "warm_summary_close",
            "metadata": {"context_turns": 2},
        },
        {
            "conversation_id": "c2",
            "turn_index": 1,
            "posterior": 0.82,
            "most_likely_state": "setting_sensory_detail",
            "observation": "sensory_setting_focus",
            "metadata": {"context_turns": 2},
        },
        {
            "conversation_id": "c3",
            "turn_index": 1,
            "posterior": 0.99,
            "most_likely_state": "off_style",
            "observation": "generic_or_unrelated",
            "metadata": {"context_turns": 2},
        },
    ]

    selected = select_high_posterior_records(
        records,
        min_posterior=0.75,
        max_records=None,
        sort_by_posterior=False,
        sort_by_selection=True,
        require_preferred=True,
    )

    assert len(selected) == 1
    assert selected[0]["conversation_id"] == "c2"
    assert selected[0]["selection_score"] > 1.0
    assert "preferred_observation=sensory_setting_focus" in selected[0]["selection_reason"]


def test_select_high_posterior_records_limits_per_dialogue():
    records = [
        {
            "conversation_id": "c1",
            "turn_index": 1,
            "posterior": 0.9,
            "most_likely_state": "activity_social_detail",
            "observation": "activity_social_focus",
            "metadata": {"context_turns": 2},
        },
        {
            "conversation_id": "c1",
            "turn_index": 2,
            "posterior": 0.88,
            "most_likely_state": "activity_social_detail",
            "observation": "activity_social_focus",
            "metadata": {"context_turns": 2},
        },
        {
            "conversation_id": "c2",
            "turn_index": 1,
            "posterior": 0.86,
            "most_likely_state": "opening_invitation",
            "observation": "ack_open_probe",
            "metadata": {"context_turns": 1},
        },
    ]

    selected = select_high_posterior_records(
        records,
        min_posterior=0.75,
        max_records=None,
        sort_by_posterior=False,
        sort_by_selection=True,
        per_dialogue_limit=1,
    )

    assert [record["conversation_id"] for record in selected] == ["c1", "c2"]
    assert [record["turn_index"] for record in selected] == [1, 1]


def test_validate_translation_payload_requires_candidates():
    payload = {
        "translated_prompt": "昔の旅行の話ですね。",
        "translated_chosen": "その時の景色で、特に覚えているものはありますか。",
        "rejected_candidates": ["そうなんですね。", "旅行はいいですよね。"],
        "translation_quality_score": 1.2,
    }

    result = validate_translation_payload(payload, candidates=2)

    assert result["translation_quality_score"] == 1.0
    assert len(result["rejected_candidates"]) == 2


def test_translation_rejected_input_records_seed_and_counts():
    record = {"conversation_id": "c1", "turn_index": 3, "prompt": "A: hello", "response": "B: hi"}

    text = build_translation_rejected_input(record, candidates=4, seed=43)

    assert "seed: 43" in text
    assert "rejected_candidates_count: 4" in text
    assert "english_prompt" in text


def test_translation_rejected_prompt_controls_natural_low_score_candidates():
    model = parse_transition_bayes_model(make_transition_payload())

    instructions = build_translation_rejected_instructions(model)

    assert "直訳ではなく" in instructions
    assert "意図" in instructions
    assert "感情" in instructions
    assert "会話戦略" in instructions
    assert "一見自然" in instructions
    assert "chosenの単なる短縮" in instructions
    assert "候補ごとに低評価になりやすい理由" in instructions
    assert "過去の経験" in instructions
    assert "思い出の情景" in instructions
    assert "昔の経験や情景を深めない" in instructions


def test_build_dpo_records_keeps_research_tracking_top_level_keys(tmp_path: Path):
    payload = make_transition_payload()
    model_path = tmp_path / "transition_model.json"
    model_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    model = parse_transition_bayes_model(payload)
    generator = StubGenerator(
        [
            json.dumps(
                {
                    "translated_prompt": "speaker_a: 昔、京都に行ったことがあります。",
                    "translated_chosen": "その京都で、特に印象に残っている景色はありますか。",
                    "rejected_candidates": [
                        "京都は有名な観光地ですよね。",
                        "そうなんですね。旅行は楽しいですよね。",
                    ],
                    "translation_quality_score": 0.92,
                },
                ensure_ascii=False,
            ),
            json.dumps({"observation": "followup", "score": 0.93, "reason": "具体的に深めている"}, ensure_ascii=False),
            json.dumps({"observation": "reflection", "score": 0.65, "reason": "受け止め中心"}, ensure_ascii=False),
            json.dumps({"observation": "generic_shift", "score": 0.88, "reason": "一般論に寄っている"}, ensure_ascii=False),
        ]
    )
    selected_records = [
        {
            "conversation_id": "train_000001",
            "turn_index": 3,
            "prompt": "speaker_a: I went to Kyoto long ago.",
            "response": "What scenery do you remember most?",
            "posterior": 0.91,
            "metadata": {
                "source_dataset": "DailyDialog",
                "source_split": "train",
                "context_turns": 2,
            },
        }
    ]

    records = build_dpo_records(
        selected_records,
        bayes_model=model,
        bayes_model_path=model_path,
        generator=generator,
        model="gpt-5.4",
        score_model="gpt-5.4",
        max_output_tokens=512,
        candidates=2,
        min_score_gap=0.0,
        min_chosen_posterior=0.0,
        max_rejected_posterior=1.0,
        seed=42,
        max_records=None,
    )

    assert len(records) == 1
    record = records[0]
    assert record["source_dataset"] == "DailyDialog"
    assert record["history_turns"] == 2
    assert record["model_used_for_scoring"] == "gpt-5.4"
    assert record["model_used_for_translation"] == "gpt-5.4"
    assert record["model_used_for_rejected_generation"] == "gpt-5.4"
    assert record["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert record["bayesian_model_version"]
    assert isinstance(record["state_sequence"], list)
    assert isinstance(record["strategy_sequence"], list)
    assert record["score_gap"] > 0.0
    assert record["rejected"] == "そうなんですね。旅行は楽しいですよね。"


def test_build_dpo_records_skips_content_filter_generation(tmp_path: Path):
    payload = make_transition_payload()
    model_path = tmp_path / "transition_model.json"
    model_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    model = parse_transition_bayes_model(payload)
    generator = StubGenerator(
        [
            RuntimeError("content_filter: prompt triggered content management policy"),
            json.dumps(
                {
                    "translated_prompt": "speaker_a: 昔、京都に行ったことがあります。",
                    "translated_chosen": "その京都で、特に印象に残っている景色はありますか。",
                    "rejected_candidates": [
                        "京都は有名な観光地ですよね。",
                        "そうなんですね。旅行は楽しいですよね。",
                    ],
                    "translation_quality_score": 0.92,
                },
                ensure_ascii=False,
            ),
            json.dumps({"observation": "followup", "score": 0.93, "reason": "具体的に深めている"}, ensure_ascii=False),
            json.dumps({"observation": "reflection", "score": 0.65, "reason": "受け止め中心"}, ensure_ascii=False),
            json.dumps({"observation": "generic_shift", "score": 0.88, "reason": "一般論に寄っている"}, ensure_ascii=False),
        ]
    )
    selected_records = [
        {
            "conversation_id": "train_cf",
            "turn_index": 1,
            "prompt": "speaker_a: filtered source",
            "response": "filtered response",
            "posterior": 0.90,
            "metadata": {"source_dataset": "DailyDialog", "context_turns": 1},
        },
        {
            "conversation_id": "train_ok",
            "turn_index": 2,
            "prompt": "speaker_a: I went to Kyoto long ago.",
            "response": "What scenery do you remember most?",
            "posterior": 0.91,
            "metadata": {"source_dataset": "DailyDialog", "context_turns": 1},
        },
    ]

    records = build_dpo_records(
        selected_records,
        bayes_model=model,
        bayes_model_path=model_path,
        generator=generator,
        model="gpt-5.4",
        score_model="gpt-5.4",
        max_output_tokens=512,
        candidates=2,
        min_score_gap=0.0,
        min_chosen_posterior=0.0,
        max_rejected_posterior=1.0,
        seed=42,
        max_records=None,
    )

    assert len(records) == 1
    assert records[0]["source_dialogue_id"] == "train_ok"


def test_build_dpo_records_stops_at_target_records(tmp_path: Path):
    payload = make_transition_payload()
    model_path = tmp_path / "transition_model.json"
    model_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    model = parse_transition_bayes_model(payload)
    generator = StubGenerator(
        [
            json.dumps(
                {
                    "translated_prompt": "speaker_a: 昔、京都に行ったことがあります。",
                    "translated_chosen": "その京都で、特に印象に残っている景色はありますか。",
                    "rejected_candidates": [
                        "京都は有名な観光地ですよね。",
                        "そうなんですね。旅行は楽しいですよね。",
                    ],
                    "translation_quality_score": 0.92,
                },
                ensure_ascii=False,
            ),
            json.dumps({"observation": "followup", "score": 0.93, "reason": "具体的に深めている"}, ensure_ascii=False),
            json.dumps({"observation": "reflection", "score": 0.65, "reason": "受け止め中心"}, ensure_ascii=False),
            json.dumps({"observation": "generic_shift", "score": 0.88, "reason": "一般論に寄っている"}, ensure_ascii=False),
        ]
    )
    selected_records = [
        {
            "conversation_id": "train_1",
            "turn_index": 2,
            "prompt": "speaker_a: I went to Kyoto long ago.",
            "response": "What scenery do you remember most?",
            "posterior": 0.91,
            "metadata": {"source_dataset": "DailyDialog", "context_turns": 1},
        },
        {
            "conversation_id": "train_2",
            "turn_index": 2,
            "prompt": "speaker_a: I went to Nara long ago.",
            "response": "Who were you with?",
            "posterior": 0.90,
            "metadata": {"source_dataset": "DailyDialog", "context_turns": 1},
        },
    ]

    records = build_dpo_records(
        selected_records,
        bayes_model=model,
        bayes_model_path=model_path,
        generator=generator,
        model="gpt-5.4",
        score_model="gpt-5.4",
        max_output_tokens=512,
        candidates=2,
        min_score_gap=0.0,
        min_chosen_posterior=0.0,
        max_rejected_posterior=1.0,
        seed=42,
        max_records=None,
        target_records=1,
    )

    assert len(records) == 1
    assert len(generator.calls) == 4


def test_parse_audit_payload_requires_quality_thresholds():
    payload = {
        "pass": True,
        "quality_score": 0.90,
        "chosen_alignment_score": 0.80,
        "rejected_contrast_score": 0.50,
        "japanese_naturalness_score": 0.95,
        "issues": ["rejectedの差が弱い"],
        "reason": "chosenは良いがrejectedとの差が弱い。",
    }

    audit = parse_audit_payload(payload, min_quality_score=0.78)

    assert audit["pass"] is False
    assert audit["model_pass"] is True
    assert audit["rejected_contrast_score"] == 0.50


def test_audit_prompt_checks_reminiscence_dpo_quality():
    instructions = build_audit_instructions()

    assert "回想支援型" in instructions
    assert "過去の経験" in instructions
    assert "思い出の情景" in instructions
    assert "一見自然" in instructions


def test_audit_records_keeps_only_passed_records(tmp_path: Path):
    generator = StubGenerator(
        [
            json.dumps(
                {
                    "pass": True,
                    "quality_score": 0.91,
                    "chosen_alignment_score": 0.88,
                    "rejected_contrast_score": 0.80,
                    "japanese_naturalness_score": 0.92,
                    "issues": [],
                    "reason": "chosenが思い出を深め、rejectedとの差も明確。",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "pass": False,
                    "quality_score": 0.55,
                    "chosen_alignment_score": 0.40,
                    "rejected_contrast_score": 0.70,
                    "japanese_naturalness_score": 0.90,
                    "issues": ["chosenが一般論"],
                    "reason": "chosenが回想を深めていない。",
                },
                ensure_ascii=False,
            ),
        ]
    )
    records = [
        {
            "prompt": "話し手A: 昔、京都に行きました。",
            "chosen": "その時の景色で、今も覚えているものはありますか。",
            "rejected": "京都は有名な観光地ですよね。",
            "score_gap": 0.7,
            "source_dialogue_id": "c1",
            "turn_index": 2,
        },
        {
            "prompt": "話し手A: 昔、奈良に行きました。",
            "chosen": "奈良は観光地として人気ですね。",
            "rejected": "そうなんですね。",
            "score_gap": 0.4,
            "source_dialogue_id": "c2",
            "turn_index": 2,
        },
    ]
    output_path = tmp_path / "audited.jsonl"
    report_path = tmp_path / "audit.jsonl"

    accepted = audit_records(
        records,
        generator=generator,
        model="gpt-5.4-pro",
        max_output_tokens=512,
        min_quality_score=0.78,
        max_records=None,
        workers=1,
        output_path=output_path,
        report_path=report_path,
    )

    assert len(accepted) == 1
    assert accepted[0]["source_dialogue_id"] == "c1"
    assert accepted[0]["model_used_for_quality_audit"] == "gpt-5.4-pro"
    assert accepted[0]["quality_audit"]["pass"] is True
    assert len(report_path.read_text(encoding="utf-8").splitlines()) == 2


def test_compare_scored_records_reports_overlap_and_correlation():
    scored_a = [
        {"conversation_id": "c1", "turn_index": 1, "posterior": 0.9},
        {"conversation_id": "c2", "turn_index": 1, "posterior": 0.7},
        {"conversation_id": "c3", "turn_index": 1, "posterior": 0.2},
    ]
    scored_b = [
        {"conversation_id": "c1", "turn_index": 1, "posterior": 0.85},
        {"conversation_id": "c2", "turn_index": 1, "posterior": 0.65},
        {"conversation_id": "c3", "turn_index": 1, "posterior": 0.25},
    ]

    report = compare_scored_records(scored_a, scored_b, model_a="gpt-5.4", model_b="gpt-5.4-pro", top_k=2)

    assert report["records_compared"] == 3
    assert report["top_k_overlap"] == 1.0
    assert report["spearman_rank_correlation"] == 1.0
    assert report["recommendation"] == "use_model_a_for_bulk_scoring"


def test_rank_metrics_handle_different_orders():
    scores_a = {"a": 0.9, "b": 0.5, "c": 0.1}
    scores_b = {"a": 0.1, "b": 0.5, "c": 0.9}

    assert top_k_overlap(scores_a, scores_b, top_k=1) == 0.0
    assert spearman_rank_correlation(scores_a, scores_b) < 0.0
