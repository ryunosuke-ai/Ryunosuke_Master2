"""DPO共通prompt整形の軽量テスト。"""

from core.dpo_prompting import (
    DPO_PROMPT_TEMPLATE_VERSION,
    build_dpo_prompt_from_context_text,
    context_text_to_user_ai_turns,
)


def test_context_text_to_user_ai_turns_maps_last_context_to_user():
    turns = context_text_to_user_ai_turns(
        "話し手A: 仕事のことを考えるだけで苦しいです。\n"
        "話し手B: かなり張りつめているのですね。\n"
        "話し手A: スマホを見るのも怖いです。"
    )

    assert turns == [
        {"speaker": "User", "text": "仕事のことを考えるだけで苦しいです。"},
        {"speaker": "AI", "text": "かなり張りつめているのですね。"},
        {"speaker": "User", "text": "スマホを見るのも怖いです。"},
    ]


def test_build_dpo_prompt_from_context_text_uses_shared_instruction_template():
    prompt = build_dpo_prompt_from_context_text("speaker_a: I am worried.\nspeaker_b: I see.")

    assert DPO_PROMPT_TEMPLATE_VERSION == "dpo_user_ai_instruction.v1"
    assert "以下の会話の次のAI返答を生成してください。" in prompt
    assert "返答は日本語で1〜2文" in prompt
    assert "AI: I am worried." in prompt
    assert "User: I see." in prompt
    assert prompt.endswith("\n\nAI:")

