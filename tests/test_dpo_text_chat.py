"""DPO後モデルのターミナルチャットの軽量テスト。"""

import sys
from pathlib import Path

from apps.dpo_text_chat import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_LORA_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    append_history_line,
    append_prompt_history_turn,
    build_dpo_generation_prompt,
    build_dpo_prompt,
    create_run_dir,
    cleanup_generated_text,
    parse_args,
    write_session_header,
    strip_prompt_prefix,
)


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
        return "templated prompt"


def test_build_dpo_prompt_matches_dataset_style():
    prompt = build_dpo_prompt("最近、またギターを弾きたくなってきました。")

    assert prompt == (
        "以下の会話の次のAI返答を生成してください。\n"
        "返答は日本語で1〜2文にしてください。\n"
        "ユーザーが話し続けやすいように、共感や具体語の拾いを使い、必要な時だけ質問を1つ添えてください。\n\n"
        "これまでの会話:\n"
        "User: 最近、またギターを弾きたくなってきました。\n\n"
        "AI:"
    )


def test_build_dpo_prompt_includes_recent_history_before_latest_user():
    history = [
        {"speaker": "User", "text": "昨日は散歩しました。"},
        {"speaker": "AI", "text": "どのあたりを歩かれたんですか？"},
    ]

    prompt = build_dpo_prompt("近所の公園です。", history)

    assert prompt == (
        "以下の会話の次のAI返答を生成してください。\n"
        "返答は日本語で1〜2文にしてください。\n"
        "ユーザーが話し続けやすいように、共感や具体語の拾いを使い、必要な時だけ質問を1つ添えてください。\n\n"
        "これまでの会話:\n"
        "User: 昨日は散歩しました。\n"
        "AI: どのあたりを歩かれたんですか？\n"
        "User: 近所の公園です。\n\n"
        "AI:"
    )


def test_append_prompt_history_turn_limits_recent_turns():
    history = []

    append_prompt_history_turn(history, "User", "1", max_history_turns=3)
    append_prompt_history_turn(history, "AI", "2", max_history_turns=3)
    append_prompt_history_turn(history, "User", "3", max_history_turns=3)
    append_prompt_history_turn(history, "AI", "4", max_history_turns=3)

    assert history == [
        {"speaker": "AI", "text": "2"},
        {"speaker": "User", "text": "3"},
        {"speaker": "AI", "text": "4"},
    ]


def test_build_dpo_generation_prompt_disables_qwen_thinking():
    tokenizer = FakeQwenTokenizer()
    prompt = build_dpo_prompt("旅行の話をしたいです。")

    result = build_dpo_generation_prompt(tokenizer, prompt)

    assert result == "templated prompt"
    assert tokenizer.last_enable_thinking is False
    assert tokenizer.last_messages == [{"role": "user", "content": prompt}]


def test_strip_prompt_prefix_removes_prompt_when_present():
    prompt = build_dpo_prompt("旅行の話をしたいです。")
    decoded = prompt + "いいですね。どんな場所が気になっていますか？"

    assert strip_prompt_prefix(decoded, prompt) == "いいですね。どんな場所が気になっていますか？"


def test_cleanup_generated_text_removes_qwen_special_tokens():
    prompt = build_dpo_prompt("お酒の話をしたいです。")
    decoded = "バーボン、お好きなんですね。<|im_end|>"

    assert cleanup_generated_text(decoded, prompt) == "バーボン、お好きなんですね。"


def test_default_paths():
    assert DEFAULT_BASE_MODEL_ID == "Qwen/Qwen3.5-27B"
    assert DEFAULT_LORA_PATH == (
        "artifacts/training_runs/"
        "qwen35_bayes_dpo_lora_reminiscence_5000_to_2000_ep1_lr5e-6_r8_a16_no4bit"
    )
    assert DEFAULT_MAX_NEW_TOKENS == 192


def test_parse_args_defaults_to_non_4bit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["apps.dpo_text_chat"])

    args = parse_args()

    assert args.use_4bit is False


def test_append_history_line_writes_timestamped_line(tmp_path):
    history_file = tmp_path / "log.txt"

    append_history_line(str(history_file), "User", "こんにちは\n元気ですか?")

    content = history_file.read_text(encoding="utf-8").strip()
    assert content.startswith("[")
    assert "] User: こんにちは 元気ですか?" in content


def test_write_session_header_writes_metadata(tmp_path):
    history_file = tmp_path / "log.txt"
    args = type(
        "Args",
        (),
        {
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "temperature": 0.7,
            "top_p": 0.8,
            "repetition_penalty": 1.0,
            "seed": 42,
        },
    )()

    write_session_header(
        str(history_file),
        base_model_id=DEFAULT_BASE_MODEL_ID,
        lora_path=DEFAULT_LORA_PATH,
        use_4bit=False,
        args=args,
    )

    content = history_file.read_text(encoding="utf-8")
    assert "# base_model_id: Qwen/Qwen3.5-27B" in content
    assert f"# lora_path: {DEFAULT_LORA_PATH}" in content
    assert "# use_4bit: False" in content
    assert "# thinking: disabled" in content
    assert "# max_new_tokens: 192" in content


def test_create_run_dir_creates_log_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    run_dir, history_file = create_run_dir()

    assert Path(run_dir).exists()
    assert Path(history_file).parent == Path(run_dir)
    assert Path(run_dir).parts[-3:-1] == (
        "dpo_text_chat",
        f"dpo__Qwen_Qwen3.5-27B__{Path(DEFAULT_LORA_PATH).name}",
    )
    assert (Path(run_dir) / "run_meta.json").exists()
