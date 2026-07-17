"""DPO学習・評価で共有するprompt整形ユーティリティ。"""

from __future__ import annotations

import re
from typing import Any


DPO_PROMPT_TEMPLATE_VERSION = "dpo_user_ai_instruction.v1"
CONTEXT_ONLY_DPO_PROMPT_TEMPLATE_VERSION = "dpo_user_ai_context_only.v1"
NEUTRAL_CONVERSATION_DPO_PROMPT_TEMPLATE_VERSION = (
    "dpo_user_ai_neutral_instruction.v2"
)
DEFAULT_MAX_HISTORY_TURNS = 10
MATHDIAL_CONTEXT_MARKER = "\n\nこれまでの学習対話:\n"
NEUTRAL_CONVERSATION_INSTRUCTION = (
    "以下の会話の文脈に沿って、次のAIの応答を自然な日本語で生成してください。"
)

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


def build_context_only_dpo_prompt(
    user_text: str = "",
    history_turns: list[dict[str, str]] | tuple[dict[str, str], ...] | None = None,
    *,
    max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
) -> str:
    """指示を加えず、User/AI履歴だけから次応答用promptを作る。"""
    if max_history_turns <= 0:
        raise ValueError("max_history_turnsは正数である必要があります。")
    lines: list[str] = []
    for turn in list(history_turns or [])[-max_history_turns:]:
        speaker = normalize_speaker(turn.get("speaker", "User"))
        text = clean_turn_text(turn.get("text", ""))
        if text:
            lines.append(f"{speaker}: {text}")
    clean_user_text = clean_turn_text(user_text)
    if clean_user_text:
        lines.append(f"User: {clean_user_text}")
    if not lines:
        raise ValueError("context-only promptに有効な会話履歴がありません。")
    lines.append("AI:")
    return "\n".join(lines)


def build_neutral_conversation_dpo_prompt(
    user_text: str = "",
    history_turns: list[dict[str, str]] | tuple[dict[str, str], ...] | None = None,
    *,
    max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
) -> str:
    """会話タスクだけを明示し、目的スタイルを指定しないpromptを作る。"""
    context = build_context_only_dpo_prompt(
        user_text=user_text,
        history_turns=history_turns,
        max_history_turns=max_history_turns,
    )
    return f"{NEUTRAL_CONVERSATION_INSTRUCTION}\n\n{context}"


def _mathdial_instruction_prompt_turns(
    prompt: str,
) -> list[dict[str, str]]:
    """旧MathDial promptからUser/AI会話行だけを読む。"""
    text = str(prompt)
    if MATHDIAL_CONTEXT_MARKER not in text:
        raise ValueError("旧MathDial promptの学習対話markerが見つかりません。")
    _, context = text.split(MATHDIAL_CONTEXT_MARKER, 1)
    context = context.strip()
    if not context:
        raise ValueError("旧MathDial promptの会話本文が空です。")
    if not context.endswith("AI:"):
        raise ValueError("旧MathDial promptが末尾の`AI:`で終わっていません。")
    lines = [line for line in context.splitlines() if line.strip()]
    if not lines or lines[-1] != "AI:":
        raise ValueError("旧MathDial promptの末尾roleが不正です。")
    turns: list[dict[str, str]] = []
    for line in lines[:-1]:
        match = re.fullmatch(r"(User|AI):(?: (.*))?", line)
        if match is None:
            raise ValueError(
                "旧MathDial promptにUser/AI形式ではない会話行があります。"
            )
        turns.append(
            {
                "speaker": match.group(1),
                "text": match.group(2) or "",
            }
        )
    return turns


def convert_mathdial_instruction_prompt_to_context_only(prompt: str) -> str:
    """旧MathDial promptからinstructionを除去し、共通builderで再構築する。"""
    turns = _mathdial_instruction_prompt_turns(prompt)
    return build_context_only_dpo_prompt(
        history_turns=turns,
        max_history_turns=len(turns),
    )


def convert_mathdial_instruction_prompt_to_neutral_conversation(
    prompt: str,
) -> str:
    """旧MathDial promptを中立的な最小会話指示へ変換する。"""
    turns = _mathdial_instruction_prompt_turns(prompt)
    return build_neutral_conversation_dpo_prompt(
        history_turns=turns,
        max_history_turns=len(turns),
    )


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
