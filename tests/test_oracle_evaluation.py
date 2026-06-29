"""Oracle評価パイプラインの軽量テスト。"""

import json
from pathlib import Path

import pytest

from core.transition_bayes_model import parse_transition_bayes_model
from tools.run_oracle_evaluation import (
    ESCONV_STRATEGY_V3_AXIS_KEYS,
    ESCONV_STRATEGY_V3_PRESET,
    EvaluationPrompt,
    OracleRetryConfig,
    append_failure_record,
    append_jsonl_record,
    build_partial_summary,
    build_context_only_prompt,
    build_judge_input,
    build_judge_instructions,
    build_local_model_prompt,
    build_reference_input,
    build_reference_instructions,
    format_judgment_progress,
    model_order_for_prompt,
    parse_category_filter,
    parse_args,
    parse_judge_payload,
    parse_reference_payload,
    read_jsonl_lenient,
    read_evaluation_prompts,
    records_by_sample_key,
    retry_config_from_env,
    run_oracle_judgment,
    run_with_retry,
    summarize_judgments,
)
from tools.run_oracle_evaluation_prompt_only import (
    FewShotExample,
    build_prompt_only_fewshot_prompt,
    parse_single_judge_payload,
    select_balanced_fewshot_examples,
    summarize_prompt_only_judgments,
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


class OracleStubGenerator:
    """Oracle API呼び出しを置き換えるテスト用生成器。"""

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if "oracle_response_100_points" in kwargs["input_text"]:
            return json.dumps(
                {
                    "score_a": 80,
                    "score_b": 90,
                    "winner": "response_b",
                    "rubric_scores": {
                        "context_understanding": 85,
                        "concrete_pickup": 86,
                        "experiential_deepening": 87,
                        "emotion_and_scene": 88,
                        "conversation_continuity": 89,
                        "avoids_generic_advice": 90,
                        "japanese_naturalness": 91,
                    },
                    "reason": "response_bの方が近い。",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "oracle_response": "その時の景色で、今も覚えているものはありますか。",
                "reason": "具体的な情景を深めている。",
            },
            ensure_ascii=False,
        )


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


def test_oracle_instructions_have_esconv_support_preset():
    model = parse_transition_bayes_model(make_transition_payload())

    reference = build_reference_instructions(model, style_preset="esconv_support")
    judge = build_judge_instructions(model, style_preset="esconv_support")

    assert "支援的対話" in reference
    assert "ESConv由来の支援Strategy" in reference
    assert "感情" in reference
    assert "早すぎる助言" in reference
    assert "支援的対話スタイル" in judge
    assert "Strategy選択" in judge
    assert "相談の進展" in judge


def test_esconv_oracle_eval_v2_prompts_are_contextual_and_balanced():
    path = Path("configs/evaluation_prompts/esconv_oracle_eval_v2_100.jsonl")

    prompts = read_evaluation_prompts(path)
    categories = {}
    for prompt in prompts:
        categories[prompt.category] = categories.get(prompt.category, 0) + 1

    assert len(prompts) == 100
    assert len({prompt.prompt_id for prompt in prompts}) == 100
    assert min(categories.values()) >= 10
    assert "emotion_reflection" in categories
    assert "suggestion_timing" in categories
    assert "strategy_contrast" in categories
    assert all(len(prompt.history) >= 2 for prompt in prompts)


def test_esconv_oracle_eval_v3_prompts_are_contextual_and_axis_focused():
    path = Path("configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl")

    prompts = read_evaluation_prompts(path)
    categories = {}
    for prompt in prompts:
        categories[prompt.category] = categories.get(prompt.category, 0) + 1

    assert len(prompts) == 100
    assert len({prompt.prompt_id for prompt in prompts}) == 100
    assert set(categories.values()) == {10}
    assert "emotional_reflection_validation" in categories
    assert "premature_advice_avoidance" in categories
    assert "strategy_contrast_core" in categories
    assert all(prompt.prompt_id.startswith("esconv_v3_") for prompt in prompts)
    assert all(len(prompt.history) >= 2 for prompt in prompts)
    assert all(prompt.axis_focus for prompt in prompts)


def test_oracle_instructions_have_esconv_strategy_v3_preset():
    model = parse_transition_bayes_model(make_transition_payload())

    reference = build_reference_instructions(model, style_preset=ESCONV_STRATEGY_V3_PRESET)
    judge = build_judge_instructions(model, style_preset=ESCONV_STRATEGY_V3_PRESET)

    assert "感情反映・感情の受容・早すぎる助言の抑制を最優先" in reference
    assert "質問がない応答でも、感情反映と受容が十分なら良い応答" in reference
    assert "ESConvらしさの主要軸" in judge
    assert "Conversational progression として別軸" in judge
    assert "weighted_esconv_overall" in judge
    for axis_key in ESCONV_STRATEGY_V3_AXIS_KEYS:
        assert axis_key in judge


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


def test_read_evaluation_prompts_accepts_history(tmp_path: Path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "p1",
                "category": "sensory_setting",
                "history": [
                    {"speaker": "User", "text": "昔は川でよく遊びました。"},
                    {"speaker": "AI", "text": "どんな遊びをしていましたか。"},
                ],
                "prompt": "魚を捕まえていました。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = read_evaluation_prompts(path)

    assert prompts[0].history[0]["speaker"] == "User"
    assert prompts[0].history[1]["speaker"] == "AI"
    assert "conversation_context" in build_reference_input(prompts[0])
    assert "latest_user_prompt" in build_judge_input(
        prompt=prompts[0],
        oracle_response="川で魚を捕まえた時間、楽しそうですね。",
        response_a="川で魚を捕まえるのは楽しそうですね。",
        response_b="いいですね。",
    )


def test_read_evaluation_prompts_can_skip_and_filter_categories(tmp_path: Path):
    path = tmp_path / "prompts.jsonl"
    records = [
        {"id": "p1", "category": "opening_invitation", "prompt": "昔、川で遊びました。"},
        {"id": "p2", "category": "sensory_setting", "prompt": "水が冷たかったです。"},
        {"id": "p3", "category": "activity_social", "prompt": "友だちと遊びました。"},
        {"id": "p4", "category": "activity_social", "prompt": "兄弟と出かけました。"},
    ]
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")

    categories = parse_category_filter("activity_social, warm_closure")
    prompts = read_evaluation_prompts(path, categories=categories, skip_prompts=1, max_prompts=1)

    assert categories == {"activity_social", "warm_closure"}
    assert [prompt.prompt_id for prompt in prompts] == ["p4"]


def test_build_local_model_prompt_context_only_avoids_instruction_bias():
    prompt = EvaluationPrompt(
        prompt_id="p1",
        category="activity_social",
        history=(
            {"speaker": "User", "text": "昔は川で遊びました。"},
            {"speaker": "AI", "text": "どんな遊びをしていましたか。"},
        ),
        prompt="魚を捕まえていました。",
    )

    context_only = build_context_only_prompt(prompt)
    instruction = build_local_model_prompt(prompt, mode="instruction")

    assert context_only == (
        "User: 昔は川で遊びました。\n"
        "AI: どんな遊びをしていましたか。\n"
        "User: 魚を捕まえていました。\n"
        "AI:"
    )
    assert "返答は日本語で1〜2文" not in context_only
    assert "ユーザーが話し続けやすいように" not in context_only
    assert "返答は日本語で1〜2文" in instruction


def test_prompt_only_fewshot_selection_is_seeded_and_strategy_balanced():
    examples = [
        FewShotExample(f"e{i}", strategy, f"User: 入力{i}\nAI:", f"良い返答{i}")
        for i, strategy in enumerate(
            [
                "Reflection of feelings",
                "Reflection of feelings",
                "Question",
                "Question",
                "Information",
                "Self-disclosure",
            ],
            start=1,
        )
    ]

    first = select_balanced_fewshot_examples(examples, count=4, seed=42)
    second = select_balanced_fewshot_examples(examples, count=4, seed=42)
    other_seed = select_balanced_fewshot_examples(examples, count=4, seed=7)

    assert [item.example_id for item in first] == [item.example_id for item in second]
    assert [item.example_id for item in first] != [item.example_id for item in other_seed]
    assert len({item.source_strategy for item in first}) == 4


def test_prompt_only_fewshot_prompt_uses_only_chosen_examples():
    prompt = EvaluationPrompt(
        prompt_id="p1",
        category="emotional_reflection_validation",
        history=(
            {"speaker": "User", "text": "仕事のことを考えるだけで苦しいです。"},
            {"speaker": "AI", "text": "かなり張りつめているのですね。"},
        ),
        prompt="スマホを見るのも怖いです。",
        axis_focus=("emotional_reflection_validation",),
    )
    fewshot_examples = [
        FewShotExample(
            "esconv_train_000000:4",
            "Reflection of feelings",
            (
                "以下の会話の次のAI返答を生成してください。\n"
                "返答は日本語で1〜2文にしてください。\n\n"
                "これまでの会話:\n"
                "User: 仕事を失ってしまうんじゃないかと思って、不安です。\n"
                "\nAI:"
            ),
            "仕事を失うかもしれないと思うと、すごく不安になりますよね。",
        )
    ]

    model_prompt = build_prompt_only_fewshot_prompt(prompt, fewshot_examples)

    assert "仕事を失うかもしれないと思うと、すごく不安になりますよね。" in model_prompt
    assert "仕事のことを考えるだけで苦しいです。" in model_prompt
    assert "スマホを見るのも怖いです。" in model_prompt
    assert "rejected" not in model_prompt
    assert "posterior" not in model_prompt
    assert "score_chosen" not in model_prompt
    assert "reward_breakdown" not in model_prompt


def test_parse_single_judge_payload_maps_v3_scores():
    payload = {
        "scores": {
            "esconv_strategy_adherence": 90,
            "emotional_reflection_validation": 80,
            "premature_advice_avoidance": 70,
            "supportive_tone": 60,
            "contextual_grounding": 50,
            "conversational_progression": 40,
            "overall_helpfulness": 30,
        },
        "reason": "感情反映はあるが進行は控えめ。",
    }

    result = parse_single_judge_payload(payload)

    assert result["axis_scores"]["esconv_strategy_adherence"] == 90
    assert result["esconv_core_score"] == pytest.approx(81.5)
    assert result["weighted_esconv_overall_score"] == pytest.approx(71.0)
    assert result["reason"] == "感情反映はあるが進行は控えめ。"


def test_summarize_prompt_only_judgments_reports_axes_and_baseline_gap():
    judgments = [
        make_prompt_only_judgment("p1", "emotion", core=80, overall=75, axis_value=70),
        make_prompt_only_judgment("p2", "emotion", core=90, overall=85, axis_value=80),
    ]
    baseline_summary = {
        "esconv_core_score": {"mean_dpo": 90},
        "weighted_esconv_overall": {"mean_dpo": 88},
        "axis_scores": {
            axis_key: {"mean_dpo": 82}
            for axis_key in ESCONV_STRATEGY_V3_AXIS_KEYS
        },
    }

    summary = summarize_prompt_only_judgments(
        judgments,
        baseline_summary=baseline_summary,
        baseline_summary_path="baseline/summary.json",
    )

    assert summary["records"] == 2
    assert summary["esconv_core_score"]["mean_prompt_only"] == pytest.approx(85)
    assert summary["weighted_esconv_overall"]["mean_prompt_only"] == pytest.approx(80)
    assert summary["axis_scores"]["supportive_tone"]["mean_prompt_only"] == pytest.approx(75)
    assert summary["by_category"]["emotion"]["count"] == 2
    assert summary["baseline_comparison"]["esconv_core_score"]["gap"] == pytest.approx(-5)
    assert summary["baseline_comparison"]["weighted_esconv_overall"]["gap"] == pytest.approx(-8)


def test_parse_args_defaults_to_instruction_prompt_mode(monkeypatch):
    monkeypatch.setattr("sys.argv", ["tools.run_oracle_evaluation"])

    args = parse_args()

    assert args.local_prompt_mode == "instruction"


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


def test_parse_judge_payload_maps_v3_axis_scores_and_weighted_winner():
    payload = {
        "scores": {
            "response_a": {
                "esconv_strategy_adherence": 80,
                "emotional_reflection_validation": 80,
                "premature_advice_avoidance": 80,
                "supportive_tone": 100,
                "contextual_grounding": 100,
                "conversational_progression": 100,
                "overall_helpfulness": 100,
            },
            "response_b": {
                "esconv_strategy_adherence": 90,
                "emotional_reflection_validation": 90,
                "premature_advice_avoidance": 90,
                "supportive_tone": 40,
                "contextual_grounding": 40,
                "conversational_progression": 20,
                "overall_helpfulness": 30,
            },
        },
        "winner": "response_b",
        "reason": "response_bはESConv主要軸は高いがweightedでは弱い。",
    }

    result = parse_judge_payload(payload, style_preset=ESCONV_STRATEGY_V3_PRESET)

    assert result["esconv_core_score_a"] == pytest.approx(80)
    assert result["esconv_core_score_b"] == pytest.approx(90)
    assert result["weighted_esconv_overall_score_a"] == pytest.approx(86)
    assert result["weighted_esconv_overall_score_b"] == pytest.approx(73.5)
    assert result["winner"] == "response_a"
    assert result["raw_winner"] == "response_b"


def test_model_order_for_prompt_is_deterministic_and_varies_by_seed():
    first = model_order_for_prompt("eval_001", seed=42)
    second = model_order_for_prompt("eval_001", seed=42)
    other_seed = model_order_for_prompt("eval_001", seed=43)

    assert first == second
    assert set(first) == {"base", "dpo"}
    assert set(other_seed) == {"base", "dpo"}


def test_run_oracle_judgment_parallel_preserves_prompt_order():
    model = parse_transition_bayes_model(make_transition_payload())
    generator = OracleStubGenerator()
    response_records = [
        {
            "prompt_id": "p1",
            "category": "memory",
            "prompt": "昔、川で遊びました。",
            "history": [],
            "base_response": "川で遊ぶのは楽しいですね。",
            "dpo_response": "川の水の冷たさなど、覚えていることはありますか。",
        },
        {
            "prompt_id": "p2",
            "category": "memory",
            "prompt": "友人と旅行しました。",
            "history": [],
            "base_response": "旅行はいいですね。",
            "dpo_response": "ご友人と行った場所で、特に印象に残っている場面はありますか。",
        },
    ]

    responses, judgments = run_oracle_judgment(
        response_records,
        bayes_model=model,
        small_corpus_text="user: 昔の話です。",
        oracle_model="gpt-5.4-pro",
        max_output_tokens=512,
        seed=42,
        style_preset="reminiscence",
        generator=generator,
        oracle_workers=2,
    )

    assert [record["prompt_id"] for record in responses] == ["p1", "p2"]
    assert [record["prompt_id"] for record in judgments] == ["p1", "p2"]
    assert len(generator.calls) == 4


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


def make_v3_judgment(prompt_id, category, *, core_gap, weighted_gap, winner):
    """v3 summaryテスト用judgmentを返す。"""
    base_axis = {
        "esconv_strategy_adherence": 80,
        "emotional_reflection_validation": 80,
        "premature_advice_avoidance": 80,
        "supportive_tone": 80,
        "contextual_grounding": 80,
        "conversational_progression": 80,
        "overall_helpfulness": 80,
    }
    dpo_axis = dict(base_axis)
    dpo_axis["esconv_strategy_adherence"] += core_gap
    dpo_axis["emotional_reflection_validation"] += core_gap
    dpo_axis["premature_advice_avoidance"] += core_gap
    dpo_axis["supportive_tone"] += weighted_gap - (core_gap * 0.70)
    return {
        "prompt_id": prompt_id,
        "category": category,
        "prompt": "つらいです。",
        "axis_scores": {"base": base_axis, "dpo": dpo_axis},
        "esconv_core_score_base": 80,
        "esconv_core_score_dpo": 80 + core_gap,
        "esconv_core_score_gap": core_gap,
        "weighted_esconv_overall_score_base": 80,
        "weighted_esconv_overall_score_dpo": 80 + weighted_gap,
        "weighted_esconv_overall_score_gap": weighted_gap,
        "score_base": 80,
        "score_dpo": 80 + weighted_gap,
        "score_gap": weighted_gap,
        "winner": winner,
        "reason": "summary用の理由。",
    }


def make_prompt_only_judgment(prompt_id, category, *, core, overall, axis_value):
    """prompt-only summaryテスト用judgmentを返す。"""
    axis_scores = {
        axis_key: axis_value
        for axis_key in ESCONV_STRATEGY_V3_AXIS_KEYS
    }
    return {
        "prompt_id": prompt_id,
        "category": category,
        "prompt": "つらいです。",
        "axis_scores": axis_scores,
        "esconv_core_score": core,
        "weighted_esconv_overall_score": overall,
        "score_prompt_only": overall,
        "reason": "prompt-only summary用の理由。",
    }


def test_summarize_judgments_reports_v3_axes_and_split_examples():
    judgments = [
        make_v3_judgment("p1", "emotion", core_gap=10, weighted_gap=8, winner="dpo"),
        make_v3_judgment("p2", "emotion", core_gap=8, weighted_gap=-5, winner="base"),
        make_v3_judgment("p3", "progression", core_gap=-7, weighted_gap=-6, winner="base"),
    ]

    summary = summarize_judgments(judgments)

    assert "mean_score_base" not in summary
    assert summary["weighted_esconv_overall"]["mean_gap"] == pytest.approx(-1)
    assert summary["esconv_core_score"]["mean_gap"] == pytest.approx(11 / 3)
    assert "emotional_reflection_validation" in summary["axis_scores"]
    assert summary["by_category"]["emotion"]["count"] == 2
    assert summary["dpo_esconv_core_win_examples"][0]["prompt_id"] == "p1"
    assert summary["dpo_weighted_overall_win_examples"][0]["prompt_id"] == "p1"
    assert summary["dpo_esconv_core_win_overall_loss_examples"][0]["prompt_id"] == "p2"
    assert summary["base_esconv_core_win_examples"][0]["prompt_id"] == "p3"
    assert summary["base_weighted_overall_win_examples"][0]["prompt_id"] == "p3"


def test_lenient_jsonl_reader_skips_broken_lines(tmp_path: Path, capsys):
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"prompt_id": "p1"}),
                "{broken",
                json.dumps(["not", "object"]),
                json.dumps({"sample_id": "s2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = read_jsonl_lenient(path)

    assert [record.get("prompt_id") or record.get("sample_id") for record in records] == ["p1", "s2"]
    assert "スキップ" in capsys.readouterr().err


def test_records_by_sample_key_supports_prompt_id_and_sample_id():
    records = [
        {"prompt_id": "p1", "value": 1},
        {"sample_id": "s2", "value": 2},
        {"prompt_id": "p1", "value": 3},
    ]

    indexed = records_by_sample_key(records)

    assert indexed["p1"]["value"] == 3
    assert indexed["s2"]["value"] == 2


def test_append_jsonl_record_and_failure_record(tmp_path: Path):
    records_path = tmp_path / "records.jsonl"
    failures_path = tmp_path / "failures.jsonl"

    append_jsonl_record({"prompt_id": "p1"}, records_path)
    append_failure_record(
        path=failures_path,
        prompt_id="p2",
        stage="judgment",
        error=TimeoutError("timeout"),
        attempts=3,
    )

    assert read_jsonl_lenient(records_path)[0]["prompt_id"] == "p1"
    failure = read_jsonl_lenient(failures_path)[0]
    assert failure["prompt_id"] == "p2"
    assert failure["status"] == "failed"
    assert failure["error_type"] == "TimeoutError"


def test_progress_formatter_reports_scores_axes_and_truncates_reason():
    judgment = make_v3_judgment("p1", "emotion", core_gap=10, weighted_gap=8, winner="dpo")
    judgment["reason"] = "a" * 80

    message = format_judgment_progress(judgment, completed=1, total=3, reason_max_chars=30)

    assert "completed 1/3 p1" in message
    assert "winner=dpo" in message
    assert "core_gap=+10.0" in message
    assert "strategy" in message
    assert "reflection" in message
    assert "premature_advice" in message
    assert "helpfulness" in message
    assert "..." in message


def test_progress_formatter_tolerates_missing_optional_keys():
    message = format_judgment_progress(
        {"prompt_id": "p1", "winner": "tie", "reason": "短い理由"},
        completed=1,
        total=1,
        reason_max_chars=140,
    )

    assert "winner=tie" in message
    assert "短い理由" in message


def test_partial_summary_reports_scores_axes_and_lora_pair_aliases():
    judgments = [
        make_v3_judgment("p1", "emotion", core_gap=10, weighted_gap=8, winner="dpo"),
        make_v3_judgment("p2", "emotion", core_gap=-5, weighted_gap=-4, winner="base"),
    ]

    summary = build_partial_summary(
        judgments,
        total_prompts=5,
        extra_metadata={
            "comparison_kind": "lora_pair",
            "base_field_label": "bayes_dpo",
            "dpo_field_label": "random_dpo",
        },
    )

    assert summary["completed_judgments"] == 2
    assert summary["total_prompts"] == 5
    assert summary["base_mean_so_far"] == pytest.approx(80)
    assert summary["dpo_mean_so_far"] == pytest.approx(82)
    assert "emotional_reflection_validation" in summary["axis_scores_so_far"]
    assert summary["bayes_dpo_win_rate"] == summary["base_win_rate_so_far"]
    assert summary["random_dpo_win_rate"] == summary["dpo_win_rate_so_far"]


def test_retry_config_reads_environment(monkeypatch):
    monkeypatch.setenv("ORACLE_MAX_RETRIES", "2")
    monkeypatch.setenv("ORACLE_RETRY_BASE_SECONDS", "0.5")
    monkeypatch.setenv("ORACLE_RETRY_MAX_SECONDS", "3")

    config = retry_config_from_env()

    assert config.max_retries == 2
    assert config.base_seconds == pytest.approx(0.5)
    assert config.max_seconds == pytest.approx(3)


def test_run_with_retry_retries_json_parse_like_errors():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] == 1:
            raise json.JSONDecodeError("bad", "x", 0)
        return {"ok": True}

    result = run_with_retry(
        flaky,
        prompt_id="p1",
        stage="judgment",
        retry_config=OracleRetryConfig(max_retries=1, base_seconds=0, max_seconds=0),
    )

    assert result == {"ok": True}
    assert calls["count"] == 2


def test_run_oracle_judgment_skips_existing_judgment_without_api_call():
    model = parse_transition_bayes_model(make_transition_payload())
    generator = OracleStubGenerator()
    response_records = [
        {
            "prompt_id": "p1",
            "category": "memory",
            "prompt": "昔、川で遊びました。",
            "history": [],
            "base_response": "川で遊ぶのは楽しいですね。",
            "dpo_response": "川の水の冷たさなど、覚えていることはありますか。",
        }
    ]
    existing_judgment = {
        "prompt_id": "p1",
        "category": "memory",
        "prompt": "昔、川で遊びました。",
        "oracle_response": "川の水の冷たさを覚えていますか。",
        "score_base": 60,
        "score_dpo": 80,
        "score_gap": 20,
        "winner": "dpo",
        "rubric_scores": {},
        "reason": "既存判定。",
    }

    responses, judgments = run_oracle_judgment(
        response_records,
        bayes_model=model,
        small_corpus_text="user: 昔の話です。",
        oracle_model="gpt-5.4-pro",
        max_output_tokens=512,
        seed=42,
        style_preset="reminiscence",
        generator=generator,
        oracle_workers=1,
        existing_judgment_records=[existing_judgment],
    )

    assert judgments == [existing_judgment]
    assert responses[0]["oracle_response"] == "川の水の冷たさを覚えていますか。"
    assert generator.calls == []
