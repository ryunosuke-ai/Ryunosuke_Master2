"""ログ管理ユーティリティのテスト。"""

import json
from pathlib import Path

from core.log_manager import (
    build_model_segment,
    create_log_run_dir,
    find_latest_run_dir,
    sanitize_log_segment,
)


def test_sanitize_log_segment_removes_path_and_special_chars():
    assert sanitize_log_segment("Qwen/Qwen3.5-27B") == "Qwen_Qwen3.5-27B"
    assert sanitize_log_segment("  !!  ", fallback="unknown") == "unknown"


def test_build_model_segment_joins_clean_parts():
    assert build_model_segment("dpo", "Qwen/Qwen3.5-27B", "adapter path") == "dpo__Qwen_Qwen3.5-27B__adapter_path"


def test_create_log_run_dir_creates_classified_run_dir(tmp_path):
    run_dir, history_file, ts = create_log_run_dir(
        "dpo_compare",
        "dpo__Qwen_Qwen3.5-27B__adapter",
        ts="20260514_150003",
        logs_base=tmp_path / "logs",
        metadata={"base_model_id": "Qwen/Qwen3.5-27B"},
    )

    run_path = Path(run_dir)
    assert run_path == tmp_path / "logs" / "dpo_compare" / "dpo__Qwen_Qwen3.5-27B__adapter" / "run_20260514_150003"
    assert Path(history_file) == run_path / "log_20260514_150003.txt"
    assert ts == "20260514_150003"
    meta = json.loads((run_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["code_id"] == "dpo_compare"
    assert meta["model_id"] == "dpo__Qwen_Qwen3.5-27B__adapter"
    assert meta["base_model_id"] == "Qwen/Qwen3.5-27B"
    assert "git_commit" in meta


def test_find_latest_run_dir_searches_recursively(tmp_path):
    old_run, _history, _ts = create_log_run_dir(
        "text_chat",
        "azure_old",
        ts="20260513_120000",
        logs_base=tmp_path / "logs",
    )
    new_run, _history, _ts = create_log_run_dir(
        "bayes_v3",
        "azure_new",
        ts="20260514_120000",
        logs_base=tmp_path / "logs",
    )

    Path(old_run).touch()
    Path(new_run).touch()

    assert find_latest_run_dir(tmp_path / "logs") == Path(new_run)
