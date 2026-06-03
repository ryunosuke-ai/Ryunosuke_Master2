"""研究用の重要操作をaudit_log.mdへ追記する補助関数。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo


DEFAULT_AUDIT_LOG_PATH = "audit_log.md"


def now_jst() -> str:
    """JSTの現在時刻を文字列で返す。"""
    return datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S JST")


def _format_list(items: Iterable[str]) -> str:
    """Markdownの箇条書き本文を作る。"""
    values = [str(item) for item in items if str(item)]
    if not values:
        return "  - なし"
    return "\n".join(f"  - {value}" for value in values)


def append_audit_log(
    *,
    title: str,
    target_files: Iterable[str],
    operation: str,
    reason: str,
    alternatives: Iterable[str],
    command: str,
    before_after: Iterable[str],
    risks: Iterable[str],
    audit_log_path: str | Path = DEFAULT_AUDIT_LOG_PATH,
    details: Iterable[str] | None = None,
) -> None:
    """重要操作の要約をaudit_log.mdへ追記する。

    個人情報やAPIキーを残さないため、入力本文ではなく件数・閾値・出力パスなどの
    追跡可能なメタ情報だけを書く。
    """
    path = Path(audit_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"\n## {now_jst()}: {title}",
        "",
        "- 対象ファイル:",
        _format_list(target_files),
        "- 実行した操作:",
        f"  - {operation}",
        "- なぜその操作が必要だったか:",
        f"  - {reason}",
        "- 代替案があったか:",
        _format_list(alternatives),
        "- 実行したコマンド:",
        f"  - `{command}`",
        "- 変更前後の要約:",
        _format_list(before_after),
        "- リスクや注意点:",
        _format_list(risks),
    ]
    if details:
        lines.extend(["- 追加詳細:", _format_list(details)])
    body = "\n".join(lines)
    if path.exists():
        body = path.read_text(encoding="utf-8") + body
    else:
        body = body.lstrip()
    path.write_text(body + "\n", encoding="utf-8")
