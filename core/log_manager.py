"""実行ログの保存先とメタ情報を管理する共通ユーティリティ。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


LOGS_BASE = Path("logs")
RUN_DIR_PATTERN = re.compile(r"^run_\d{8}_\d{6}$")
UNSAFE_SEGMENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_log_segment(value: object, *, fallback: str = "unknown") -> str:
    """ログ階層に使う文字列を安全なディレクトリ名に変換する。"""
    text = str(value or "").strip()
    if not text:
        return fallback
    text = text.replace(os.sep, "_")
    if os.altsep:
        text = text.replace(os.altsep, "_")
    text = UNSAFE_SEGMENT_PATTERN.sub("_", text)
    text = text.strip("._-")
    return text[:120] or fallback


def build_model_segment(*parts: object) -> str:
    """モデル名・LoRA名などを結合してログ用モデルセグメントを作る。"""
    cleaned = [sanitize_log_segment(part) for part in parts if str(part or "").strip()]
    return "__".join(cleaned) if cleaned else "unknown_model"


def get_git_commit() -> str:
    """現在のgit commit hashを取得する。失敗時はunknownを返す。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def create_log_run_dir(
    code_id: str,
    model_id: str,
    *,
    ts: str | None = None,
    logs_base: Path | str = LOGS_BASE,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    """分類済みrunディレクトリを作り、履歴ファイルとtimestampを返す。"""
    timestamp = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    code_segment = sanitize_log_segment(code_id, fallback="unknown_code")
    model_segment = sanitize_log_segment(model_id, fallback="unknown_model")
    run_dir = Path(logs_base) / code_segment / model_segment / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_file = run_dir / f"log_{timestamp}.txt"
    write_run_metadata(
        run_dir,
        code_id=code_segment,
        model_id=model_segment,
        timestamp=timestamp,
        extra=metadata,
    )
    return run_dir.as_posix(), history_file.as_posix(), timestamp


def write_run_metadata(
    run_dir: Path | str,
    *,
    code_id: str,
    model_id: str,
    timestamp: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """run_meta.jsonを書き出す。"""
    path = Path(run_dir) / "run_meta.json"
    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "code_id": code_id,
        "model_id": model_id,
        "entrypoint": sys.argv[0] if sys.argv else "",
        "git_commit": get_git_commit(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def is_run_dir(path: Path) -> bool:
    """run_YYYYMMDD_HHMMSS 形式のディレクトリか判定する。"""
    return path.is_dir() and RUN_DIR_PATTERN.match(path.name) is not None


def find_latest_run_dir(logs_base: Path | str = LOGS_BASE) -> Path | None:
    """logs配下から最新のrunディレクトリを再帰的に探す。"""
    base = Path(logs_base)
    if not base.exists():
        return None
    run_dirs = [path for path in base.rglob("run_*") if is_run_dir(path)]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: (path.stat().st_mtime, path.name))
