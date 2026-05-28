"""DPO前後比較ログを読み込んで差分を表示するStreamlitアプリ。"""

from __future__ import annotations

import glob
import html
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOGS_BASE = Path("logs")
COMPARE_MODES = {"compare", "streamlit_compare"}
LOG_ENTRY_PATTERN = re.compile(
    r"^\[(?P<timestamp>\d{2}:\d{2}:\d{2})\]\s+"
    r"(?P<role>User|AI\(base\)|AI\(dpo\)):\s*(?P<content>.*)$"
)
TOKEN_PATTERN = re.compile(r"\s+|[A-Za-z0-9_]+|[ぁ-んァ-ン一-龥]+|[^\sA-Za-z0-9_ぁ-んァ-ン一-龥]+")


@dataclass(frozen=True)
class LogMessage:
    """ログ中の1メッセージ。"""

    timestamp: str
    role: str
    content: str


@dataclass(frozen=True)
class CompareTurn:
    """User入力とベース/DPO後返答の1ターン。"""

    index: int
    timestamp: str
    user: str
    base: str
    dpo: str


@dataclass(frozen=True)
class LogSummary:
    """比較ログ全体の集計結果。"""

    total_turns: int
    same_turns: int
    changed_turns: int
    dpo_longer_turns: int
    dpo_shorter_turns: int
    average_base_length: float
    average_dpo_length: float


def parse_metadata(raw_text: str) -> dict[str, str]:
    """ログ先頭の # key: value 形式のメタ情報を読む。"""
    metadata: dict[str, str] = {}
    for line in raw_text.splitlines():
        if not line.startswith("#"):
            continue
        body = line[1:].strip()
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def parse_log_messages(raw_text: str) -> list[LogMessage]:
    """比較ログ本文をメッセージ単位に分解する。"""
    messages: list[LogMessage] = []
    current_timestamp = ""
    current_role = ""
    current_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current_timestamp, current_role, current_lines
        if not current_role:
            return
        content = "\n".join(current_lines).strip()
        messages.append(LogMessage(current_timestamp, current_role, content))
        current_timestamp = ""
        current_role = ""
        current_lines = []

    for line in raw_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        match = LOG_ENTRY_PATTERN.match(line)
        if match:
            flush_current()
            current_timestamp = match.group("timestamp")
            current_role = match.group("role")
            current_lines = [match.group("content")]
            continue
        if current_role:
            current_lines.append(line)

    flush_current()
    return messages


def build_compare_turns(messages: list[LogMessage]) -> list[CompareTurn]:
    """メッセージ列をUser/base/dpoの比較ターンにまとめる。"""
    turns: list[CompareTurn] = []
    current_user = ""
    current_timestamp = ""
    current_base = ""

    for message in messages:
        if message.role == "User":
            current_user = message.content
            current_timestamp = message.timestamp
            current_base = ""
        elif message.role == "AI(base)":
            current_base = message.content
        elif message.role == "AI(dpo)" and current_user:
            turns.append(
                CompareTurn(
                    index=len(turns) + 1,
                    timestamp=current_timestamp or message.timestamp,
                    user=current_user,
                    base=current_base,
                    dpo=message.content,
                )
            )
            current_user = ""
            current_timestamp = ""
            current_base = ""
    return turns


def load_compare_log(log_path: Path) -> tuple[dict[str, str], list[CompareTurn]]:
    """比較ログファイルを読み込む。"""
    raw_text = log_path.read_text(encoding="utf-8")
    metadata = parse_metadata(raw_text)
    return metadata, build_compare_turns(parse_log_messages(raw_text))


def is_compare_log(log_path: Path) -> bool:
    """DPO比較ログとして扱えるファイルか判定する。"""
    try:
        raw_text = log_path.read_text(encoding="utf-8")
    except OSError:
        return False
    metadata = parse_metadata(raw_text)
    mode = metadata.get("mode", "")
    return mode in COMPARE_MODES and "AI(base)" in raw_text and "AI(dpo)" in raw_text


def list_compare_log_paths(logs_base: Path = LOGS_BASE) -> list[Path]:
    """logs配下から比較ログ候補を新しい順に列挙する。"""
    pattern = str(logs_base / "**" / "log_*.txt")
    candidates = [Path(path) for path in glob.glob(pattern, recursive=True)]
    compare_logs = [path for path in candidates if path.is_file() and is_compare_log(path)]
    return sorted(compare_logs, key=lambda path: path.stat().st_mtime, reverse=True)


def count_questions(text: str) -> int:
    """返答中の質問らしさを簡易的に数える。"""
    return text.count("?") + text.count("？")


def summarize_turns(turns: list[CompareTurn]) -> LogSummary:
    """比較ターンの概要を集計する。"""
    total_turns = len(turns)
    same_turns = sum(1 for turn in turns if turn.base == turn.dpo)
    changed_turns = total_turns - same_turns
    dpo_longer_turns = sum(1 for turn in turns if len(turn.dpo) > len(turn.base))
    dpo_shorter_turns = sum(1 for turn in turns if len(turn.dpo) < len(turn.base))
    average_base_length = sum(len(turn.base) for turn in turns) / total_turns if total_turns else 0.0
    average_dpo_length = sum(len(turn.dpo) for turn in turns) / total_turns if total_turns else 0.0
    return LogSummary(
        total_turns=total_turns,
        same_turns=same_turns,
        changed_turns=changed_turns,
        dpo_longer_turns=dpo_longer_turns,
        dpo_shorter_turns=dpo_shorter_turns,
        average_base_length=average_base_length,
        average_dpo_length=average_dpo_length,
    )


def filter_turns(turns: list[CompareTurn], filter_name: str) -> list[CompareTurn]:
    """選択された条件でターンを絞り込む。"""
    if filter_name == "差分ありのみ":
        return [turn for turn in turns if turn.base != turn.dpo]
    if filter_name == "DPOが長い":
        return [turn for turn in turns if len(turn.dpo) > len(turn.base)]
    if filter_name == "DPOが短い":
        return [turn for turn in turns if len(turn.dpo) < len(turn.base)]
    if filter_name == "完全一致のみ":
        return [turn for turn in turns if turn.base == turn.dpo]
    return turns


def _tokenize_for_diff(text: str) -> list[str]:
    """日本語ログを差分表示しやすい粒度に分ける。"""
    return TOKEN_PATTERN.findall(text)


def render_diff_html(base: str, dpo: str) -> str:
    """ベース返答からDPO後返答への差分をHTMLで表現する。"""
    base_tokens = _tokenize_for_diff(base)
    dpo_tokens = _tokenize_for_diff(dpo)
    matcher = SequenceMatcher(a=base_tokens, b=dpo_tokens)
    parts: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        base_part = html.escape("".join(base_tokens[i1:i2]))
        dpo_part = html.escape("".join(dpo_tokens[j1:j2]))
        if tag == "equal":
            parts.append(dpo_part)
        elif tag == "insert":
            parts.append(f'<span class="diff-insert">{dpo_part}</span>')
        elif tag == "delete":
            parts.append(f'<span class="diff-delete">{base_part}</span>')
        elif tag == "replace":
            if base_part:
                parts.append(f'<span class="diff-delete">{base_part}</span>')
            if dpo_part:
                parts.append(f'<span class="diff-insert">{dpo_part}</span>')

    return "".join(parts).replace("\n", "<br>")


def _short_path(path: Path) -> str:
    """画面表示用にパスを短くする。"""
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def render_metadata(metadata: dict[str, str], log_path: Path) -> None:
    """ログファイルのメタ情報を表示する。"""
    columns = st.columns(4)
    columns[0].metric("ログ", log_path.name)
    columns[1].metric("mode", metadata.get("mode", "---"))
    columns[2].metric("開始", metadata.get("session_start", "---"))
    columns[3].metric("thinking", metadata.get("thinking", "---"))
    with st.expander("モデル情報", expanded=False):
        st.text(f"base_model_id: {metadata.get('base_model_id', '---')}")
        st.text(f"lora_path: {metadata.get('lora_path', '---')}")


def render_summary(summary: LogSummary) -> None:
    """集計値を表示する。"""
    columns = st.columns(6)
    columns[0].metric("ターン数", summary.total_turns)
    columns[1].metric("差分あり", summary.changed_turns)
    columns[2].metric("完全一致", summary.same_turns)
    columns[3].metric("DPO長文化", summary.dpo_longer_turns)
    columns[4].metric("DPO短文化", summary.dpo_shorter_turns)
    columns[5].metric("平均文字数", f"{summary.average_base_length:.1f} → {summary.average_dpo_length:.1f}")


def render_turn(turn: CompareTurn) -> None:
    """1ターン分の比較を表示する。"""
    base_length = len(turn.base)
    dpo_length = len(turn.dpo)
    length_delta = dpo_length - base_length
    question_delta = count_questions(turn.dpo) - count_questions(turn.base)
    status = "完全一致" if turn.base == turn.dpo else "差分あり"

    with st.container(border=True):
        st.markdown(f"#### Turn {turn.index} / {html.escape(turn.timestamp)} / {status}")
        st.markdown(f"**User**: {html.escape(turn.user)}")
        metric_columns = st.columns(3)
        metric_columns[0].metric("文字数", f"{base_length} → {dpo_length}", delta=length_delta)
        metric_columns[1].metric("質問数", f"{count_questions(turn.base)} → {count_questions(turn.dpo)}", delta=question_delta)
        metric_columns[2].metric("一致", "Yes" if turn.base == turn.dpo else "No")

        left, right = st.columns(2)
        with left:
            st.markdown("**ベースモデル**")
            st.markdown(html.escape(turn.base).replace("\n", "<br>"), unsafe_allow_html=True)
        with right:
            st.markdown("**DPO後**")
            st.markdown(html.escape(turn.dpo).replace("\n", "<br>"), unsafe_allow_html=True)

        with st.expander("差分を見る", expanded=turn.base != turn.dpo):
            st.markdown(render_diff_html(turn.base, turn.dpo), unsafe_allow_html=True)


def render_app() -> None:
    """Streamlit UIを描画する。"""
    st.set_page_config(page_title="DPO比較ログビューア", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; max-width: 1280px; }
        .diff-insert { background: #d7f7df; color: #14532d; padding: 0.05rem 0.12rem; border-radius: 3px; }
        .diff-delete { background: #ffe0e0; color: #7f1d1d; padding: 0.05rem 0.12rem; border-radius: 3px; text-decoration: line-through; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("DPO比較ログビューア")

    log_paths = list_compare_log_paths()
    if not log_paths:
        st.warning("比較ログが見つかりません。`AI(base)` と `AI(dpo)` を含む `log_*.txt` を確認してください。")
        return

    labels = [_short_path(path) for path in log_paths]
    with st.sidebar:
        st.header("ログ選択")
        selected_label = st.selectbox("比較ログ", labels)
        filter_name = st.radio(
            "表示条件",
            ["全ターン", "差分ありのみ", "DPOが長い", "DPOが短い", "完全一致のみ"],
            index=0,
        )

    selected_path = log_paths[labels.index(selected_label)]
    metadata, turns = load_compare_log(selected_path)
    summary = summarize_turns(turns)
    filtered_turns = filter_turns(turns, filter_name)

    render_metadata(metadata, selected_path)
    render_summary(summary)
    st.caption(f"表示中: {_short_path(selected_path)} / {len(filtered_turns)}件")

    for turn in filtered_turns:
        render_turn(turn)


if __name__ == "__main__":
    render_app()
