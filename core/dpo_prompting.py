"""DPO学習・評価で共有するprompt整形ユーティリティ。"""

from __future__ import annotations

import re
from typing import Any


DPO_PROMPT_TEMPLATE_VERSION = "dpo_user_ai_instruction.v1"
DEFAULT_MAX_HISTORY_TURNS = 10

INSTRUCTION_LINES = [
    "以下の会話の次のAI返答を生成してください。",
    "返答は日本語で1〜2文にしてください。",
    "ユーザーが話し続けやすいように、共感や具体語の拾いを使い、必要な時だけ質問を1つ添えてください。",
]
MATHDIAL_INSTRUCTION_LINES = [
    "以下の個別指導対話の次の教師返答を生成してください。",
    "返答は自然な日本語で、問題とこれまでの学習者の考えに即して簡潔に書いてください。",
    "必要に応じて質問、焦点化、段階的ヒント、説明、理解確認のいずれかを選んでください。",
]

ROLE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:"
    r"User|ユーザー|usr|user|"
    r"AI|assistant|アシスタント|sys|system|"
    r"speaker_a|speaker_b|話し手A|話し手B|A|B"
    r")\s*[:：]\s*",
    re.IGNORECASE,
)


def clean_turn_text(text: Any) -> str:
    """発話本文を1行の学習用テキストへ整える。"""
    return str(text).replace("\n", " ").strip()


def normalize_speaker(speaker: Any) -> str:
    """話者表記をDPO学習用のUser/AIへ正規化する。"""
    value = str(speaker).strip().lower()
    if value in {"ai", "assistant", "アシスタント", "sys", "system"}:
        return "AI"
    return "User"


def strip_role_prefix(line: str) -> str:
    """行頭の既存話者ラベルを取り除く。"""
    return ROLE_PREFIX_PATTERN.sub("", line, count=1).strip()


def context_text_to_user_ai_turns(context_text: str) -> list[dict[str, str]]:
    """既存の文脈文字列を、次のAI応答用のUser/AI履歴へ変換する。

    DailyDialog由来のspeaker_a/speaker_bや、翻訳LLMが出した話し手A/B表記は、
    ターゲット応答直前の発話をUserとして、後ろから交互にUser/AIへ割り当てる。
    """
    raw_lines = [line.strip() for line in str(context_text).splitlines() if line.strip()]
    texts = [strip_role_prefix(line) for line in raw_lines]
    texts = [text for text in texts if text]
    turns: list[dict[str, str]] = []
    for reverse_index, text in enumerate(reversed(texts)):
        speaker = "User" if reverse_index % 2 == 0 else "AI"
        turns.append({"speaker": speaker, "text": clean_turn_text(text)})
    turns.reverse()
    return turns


def build_dpo_prompt(
    user_text: str = "",
    history_turns: list[dict[str, str]] | tuple[dict[str, str], ...] | None = None,
    *,
    max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
) -> str:
    """DPO学習・比較・Oracle評価で使う共通promptを作る。"""
    lines = [
        *INSTRUCTION_LINES,
        "",
        "これまでの会話:",
    ]
    for turn in list(history_turns or [])[-max_history_turns:]:
        speaker = normalize_speaker(turn.get("speaker", "User"))
        text = clean_turn_text(turn.get("text", ""))
        if text:
            lines.append(f"{speaker}: {text}")
    clean_user_text = clean_turn_text(user_text)
    if clean_user_text:
        lines.append(f"User: {clean_user_text}")
    lines.extend(["", "AI:"])
    return "\n".join(lines)


def build_mathdial_dpo_prompt(
    user_text: str = "",
    history_turns: list[dict[str, str]] | tuple[dict[str, str], ...] | None = None,
    *,
    max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
) -> str:
    """MathDial日本語個別指導で共有するpromptを作る。"""
    lines = [*MATHDIAL_INSTRUCTION_LINES, "", "これまでの学習対話:"]
    for turn in list(history_turns or [])[-max_history_turns:]:
        speaker = normalize_speaker(turn.get("speaker", "User"))
        text = clean_turn_text(turn.get("text", ""))
        if text:
            lines.append(f"{speaker}: {text}")
    clean_user_text = clean_turn_text(user_text)
    if clean_user_text:
        lines.append(f"User: {clean_user_text}")
    lines.extend(["", "AI:"])
    return "\n".join(lines)


def build_dpo_prompt_from_context_text(
    context_text: str,
    *,
    max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
) -> str:
    """文脈文字列だけから、次のAI応答を生成する共通promptを作る。"""
    return build_dpo_prompt(
        history_turns=context_text_to_user_ai_turns(context_text),
        max_history_turns=max_history_turns,
    )


def build_mathdial_dpo_prompt_from_context_text(
    context_text: str,
    *,
    max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
) -> str:
    """翻訳済み文脈からMathDial用DPO promptを作る。"""
    return build_mathdial_dpo_prompt(
        history_turns=context_text_to_user_ai_turns(context_text),
        max_history_turns=max_history_turns,
    )
