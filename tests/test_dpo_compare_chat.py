"""DPO前後比較UIの軽量テスト。"""

from apps.dpo_compare_chat import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_LORA_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    build_dpo_compare_prompt,
    cleanup_generated_text,
    disable_peft_bitsandbytes_dispatch,
    list_lora_adapter_paths,
    strip_prompt_prefix,
    write_streamlit_session_header,
)
from apps.dpo_text_chat import build_dpo_generation_prompt


class FakeQwenTokenizer:
    """Qwenチャットテンプレート呼び出しを記録するスタブ。"""

    def __init__(self):
        self.last_messages = None
        self.last_enable_thinking = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        self.last_messages = messages
        self.last_enable_thinking = enable_thinking
        assert tokenize is False
        assert add_generation_prompt is True
        return "templated streamlit prompt"


def test_build_dpo_compare_prompt_matches_training_template():
    prompt = build_dpo_compare_prompt("最近、またギターを弾きたくなってきました。")

    assert prompt == (
        "以下の会話の次のAI返答を生成してください。\n"
        "返答は日本語で1〜2文にしてください。\n"
        "ユーザーが話し続けやすいように、共感や具体語の拾いを使い、必要な時だけ質問を1つ添えてください。\n\n"
        "これまでの会話:\n"
        "User: 最近、またギターを弾きたくなってきました。\n\n"
        "AI:"
    )


def test_build_dpo_compare_prompt_includes_history_for_streamlit_context():
    history = [
        {"speaker": "User", "text": "今日は庭仕事をしました。"},
        {"speaker": "AI", "text": "どんな作業をされたんですか？"},
    ]

    prompt = build_dpo_compare_prompt("草取りです。", history)

    assert "User: 今日は庭仕事をしました。" in prompt
    assert "AI: どんな作業をされたんですか？" in prompt
    assert "User: 草取りです。" in prompt
    assert "AI(base)" not in prompt
    assert prompt.endswith("\n\nAI:")


def test_write_streamlit_session_header_records_compare_log_metadata(tmp_path):
    history_file = tmp_path / "log.txt"

    write_streamlit_session_header(
        str(history_file),
        base_model_id=DEFAULT_BASE_MODEL_ID,
        lora_path=DEFAULT_LORA_PATH,
    )

    content = history_file.read_text(encoding="utf-8")
    assert "# mode: streamlit_compare" in content
    assert "# base_model_id: Qwen/Qwen3.5-27B" in content
    assert "# lora_path: artifacts/training_runs/qwen35_dpo_lora_200samples_ep1_lr5e-6_r8_a16_no4bit" in content
    assert "# thinking: disabled" in content
    assert "# prompt_history_ai: dpo" in content


def test_build_dpo_compare_generation_prompt_disables_qwen_thinking():
    tokenizer = FakeQwenTokenizer()
    prompt = build_dpo_compare_prompt("旅行の話をしたいです。")

    result = build_dpo_generation_prompt(tokenizer, prompt)

    assert result == "templated streamlit prompt"
    assert tokenizer.last_enable_thinking is False
    assert tokenizer.last_messages == [{"role": "user", "content": prompt}]


def test_strip_prompt_prefix_removes_prompt_when_present():
    prompt = build_dpo_compare_prompt("旅行の話をしたいです。")
    decoded = prompt + "いいですね。どんな場所が気になっていますか？"

    assert strip_prompt_prefix(decoded, prompt) == "いいですね。どんな場所が気になっていますか？"


def test_strip_prompt_prefix_uses_last_ai_marker_as_fallback():
    decoded = "User: 旅行の話をしたいです。\n\nAI: いいですね。どんな場所が気になっていますか？"

    assert strip_prompt_prefix(decoded, "missing prompt") == "いいですね。どんな場所が気になっていますか？"


def test_cleanup_generated_text_removes_qwen_special_tokens():
    prompt = build_dpo_compare_prompt("お酒の話をしたいです。")
    decoded = "バーボン、お好きなんですね。<|im_end|>"

    assert cleanup_generated_text(decoded, prompt) == "バーボン、お好きなんですね。"


def test_default_paths():
    assert DEFAULT_BASE_MODEL_ID == "Qwen/Qwen3.5-27B"
    assert DEFAULT_LORA_PATH == "artifacts/training_runs/qwen35_dpo_lora_200samples_ep1_lr5e-6_r8_a16_no4bit"
    assert DEFAULT_MAX_NEW_TOKENS == 192


def test_list_lora_adapter_paths_returns_only_adapter_dirs(tmp_path):
    valid = tmp_path / "qwen35_dpo_lora_200samples_ep1_lr5e-6_r8_a16_no4bit"
    valid.mkdir()
    (valid / "adapter_config.json").write_text("{}", encoding="utf-8")
    (valid / "adapter_model.safetensors").write_text("dummy", encoding="utf-8")
    invalid = tmp_path / "checkpoint-25"
    invalid.mkdir()
    (invalid / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert list_lora_adapter_paths(tmp_path) == [valid.as_posix()]


def test_disable_peft_bitsandbytes_dispatch_forces_detectors_false():
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.model as peft_lora_model
    except ImportError:
        return

    disable_peft_bitsandbytes_dispatch()

    assert peft_import_utils.is_bnb_available() is False
    assert peft_import_utils.is_bnb_4bit_available() is False
    assert peft_lora_model.is_bnb_available() is False
    assert peft_lora_model.is_bnb_4bit_available() is False
