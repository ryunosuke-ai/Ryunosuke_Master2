"""DPO比較用の単独Streamlitチャットの軽量テスト。"""

from pathlib import Path

from apps.dpo_base_chat import (
    DEFAULT_BASE_MODEL_ID,
    build_base_log_model_id,
    create_run_dir as create_base_run_dir,
    write_streamlit_session_header as write_base_session_header,
)
from apps.dpo_text_chat import build_dpo_prompt
from apps.dpo_trained_chat import (
    DEFAULT_LORA_PATH,
    create_run_dir as create_trained_run_dir,
    write_streamlit_session_header as write_trained_session_header,
)


def test_base_log_model_id_has_no_lora_segment():
    model_id = build_base_log_model_id(DEFAULT_BASE_MODEL_ID)

    assert model_id == "base__Qwen_Qwen3.5-27B"


def test_base_session_header_records_independent_chat_metadata(tmp_path):
    history_file = tmp_path / "log.txt"

    write_base_session_header(str(history_file), base_model_id=DEFAULT_BASE_MODEL_ID)

    content = history_file.read_text(encoding="utf-8")
    assert "# mode: streamlit_base_chat" in content
    assert "# base_model_id: Qwen/Qwen3.5-27B" in content
    assert "# lora_path:" not in content
    assert "# prompt_template: dpo" in content


def test_trained_session_header_records_independent_chat_metadata(tmp_path):
    history_file = tmp_path / "log.txt"

    write_trained_session_header(
        str(history_file),
        base_model_id=DEFAULT_BASE_MODEL_ID,
        lora_path=DEFAULT_LORA_PATH,
    )

    content = history_file.read_text(encoding="utf-8")
    assert "# mode: streamlit_dpo_trained_chat" in content
    assert "# base_model_id: Qwen/Qwen3.5-27B" in content
    assert f"# lora_path: {DEFAULT_LORA_PATH}" in content
    assert "# prompt_template: dpo" in content


def test_base_run_dir_uses_base_code_and_model_without_lora(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    run_dir, history_file = create_base_run_dir(base_model_id=DEFAULT_BASE_MODEL_ID)

    assert Path(run_dir).exists()
    assert Path(history_file).parent == Path(run_dir)
    assert Path(run_dir).parts[-3:-1] == ("dpo_base_chat", "base__Qwen_Qwen3.5-27B")
    assert (Path(run_dir) / "run_meta.json").exists()


def test_trained_run_dir_uses_dpo_code_and_lora_segment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    run_dir, history_file = create_trained_run_dir(
        base_model_id=DEFAULT_BASE_MODEL_ID,
        lora_path=DEFAULT_LORA_PATH,
    )

    assert Path(run_dir).exists()
    assert Path(history_file).parent == Path(run_dir)
    assert Path(run_dir).parts[-3:-1] == (
        "dpo_trained_chat",
        "dpo__Qwen_Qwen3.5-27B__qwen35_dpo_lora_300samples_ep1_lr5e-6_r8_a16_no4bit",
    )
    assert (Path(run_dir) / "run_meta.json").exists()


def test_single_chat_apps_use_dpo_prompt_template():
    prompt = build_dpo_prompt("最近、またギターを弾きたくなってきました。")

    assert "以下の会話の次のAI返答を生成してください。" in prompt
    assert "User: 最近、またギターを弾きたくなってきました。" in prompt
    assert prompt.endswith("\n\nAI:")
