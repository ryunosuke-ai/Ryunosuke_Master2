"""DPO比較ログビューアの軽量テスト。"""

from pathlib import Path

from apps.dpo_log_viewer import (
    build_compare_turns,
    filter_turns,
    is_compare_log,
    list_compare_log_paths,
    load_compare_log,
    parse_log_messages,
    parse_metadata,
    render_diff_html,
    summarize_turns,
)


SAMPLE_LOG = """# session_start: 2026-05-14T15:00:03
# mode: streamlit_compare
# base_model_id: Qwen/Qwen3.5-27B
# lora_path: artifacts/training_runs/qwen35_dpo_lora_200samples_ep1_lr5e-6_r8_a16_no4bit
# thinking: disabled

[15:00:57] User: こんにちは！
[15:00:57] AI(base): こんにちは！今日はどんなことを話してみたいですか？
[15:00:57] AI(dpo): こんにちは！今日はどんなことを話してみたいですか？
[15:01:20] User: GWの話をしたいです。
[15:01:20] AI(base): GWですね。旅行や趣味など、どんな予定でしたか？
[15:01:20] AI(dpo): GWの話、いいですね！どんな過ごし方をしましたか？
[15:02:24] User: 川でBBQをしました。
[15:02:24] AI(base): 川でのBBQ、風情があって楽しかったでしょうね！
その時の料理は何でしたか？
[15:02:24] AI(dpo): 川でのBBQ、にぎやかで楽しかったでしょうね！
特に印象に残った料理はありましたか？
"""


def test_parse_metadata_reads_session_header():
    metadata = parse_metadata(SAMPLE_LOG)

    assert metadata["mode"] == "streamlit_compare"
    assert metadata["base_model_id"] == "Qwen/Qwen3.5-27B"
    assert metadata["thinking"] == "disabled"


def test_parse_log_messages_keeps_multiline_content():
    messages = parse_log_messages(SAMPLE_LOG)

    assert len(messages) == 9
    assert messages[-2].role == "AI(base)"
    assert "その時の料理は何でしたか？" in messages[-2].content
    assert messages[-1].role == "AI(dpo)"
    assert "特に印象に残った料理" in messages[-1].content


def test_build_compare_turns_groups_user_base_and_dpo():
    turns = build_compare_turns(parse_log_messages(SAMPLE_LOG))

    assert len(turns) == 3
    assert turns[0].user == "こんにちは！"
    assert turns[0].base == turns[0].dpo
    assert turns[1].base != turns[1].dpo
    assert turns[2].timestamp == "15:02:24"


def test_load_compare_log_reads_file(tmp_path):
    log_path = tmp_path / "log_20260514_150003.txt"
    log_path.write_text(SAMPLE_LOG, encoding="utf-8")

    metadata, turns = load_compare_log(log_path)

    assert metadata["lora_path"].startswith("artifacts/training_runs")
    assert len(turns) == 3


def test_summarize_turns_counts_changes_and_lengths():
    turns = build_compare_turns(parse_log_messages(SAMPLE_LOG))

    summary = summarize_turns(turns)

    assert summary.total_turns == 3
    assert summary.same_turns == 1
    assert summary.changed_turns == 2
    assert summary.dpo_longer_turns >= 1
    assert summary.average_base_length > 0
    assert summary.average_dpo_length > 0


def test_filter_turns_selects_expected_categories():
    turns = build_compare_turns(parse_log_messages(SAMPLE_LOG))

    assert len(filter_turns(turns, "全ターン")) == 3
    assert len(filter_turns(turns, "差分ありのみ")) == 2
    assert len(filter_turns(turns, "完全一致のみ")) == 1
    assert all(len(turn.dpo) > len(turn.base) for turn in filter_turns(turns, "DPOが長い"))
    assert all(len(turn.dpo) < len(turn.base) for turn in filter_turns(turns, "DPOが短い"))


def test_render_diff_html_marks_insert_and_delete():
    diff_html = render_diff_html("旅行や趣味など、どんな予定でしたか？", "海鮮旅行はどうでしたか？")

    assert "diff-insert" in diff_html
    assert "diff-delete" in diff_html
    assert "海鮮旅行" in diff_html


def test_is_compare_log_requires_compare_mode_and_outputs(tmp_path):
    compare_log = tmp_path / "log_compare.txt"
    compare_log.write_text(SAMPLE_LOG, encoding="utf-8")
    normal_log = tmp_path / "log_normal.txt"
    normal_log.write_text("# mode: normal\n[12:00:00] AI: こんにちは", encoding="utf-8")

    assert is_compare_log(compare_log) is True
    assert is_compare_log(normal_log) is False


def test_list_compare_log_paths_returns_newest_compare_logs_first(tmp_path):
    old_dir = tmp_path / "run_old"
    new_dir = tmp_path / "run_new"
    old_dir.mkdir()
    new_dir.mkdir()
    old_log = old_dir / "log_20260513_183840.txt"
    new_log = new_dir / "log_20260514_150003.txt"
    ignored_log = new_dir / "log_ignored.txt"
    old_log.write_text(SAMPLE_LOG, encoding="utf-8")
    new_log.write_text(SAMPLE_LOG.replace("2026-05-14T15:00:03", "2026-05-14T16:00:03"), encoding="utf-8")
    ignored_log.write_text("# mode: text_chat\n[12:00:00] AI: こんにちは", encoding="utf-8")
    old_time = 1_700_000_000
    new_time = 1_800_000_000
    Path(old_log).touch()
    Path(new_log).touch()
    import os

    os.utime(old_log, (old_time, old_time))
    os.utime(new_log, (new_time, new_time))

    assert list_compare_log_paths(tmp_path) == [new_log, old_log]
