"""Random-DPO baseline用の一般会話品質prompt。"""

from __future__ import annotations

from typing import Any


RANDOM_DPO_PROMPT_TEMPLATE_VERSION = "random_dailydialog_general_quality.v1"
GENERAL_QUALITY_STYLE_PRESET = "general_conversation_quality"


def build_general_quality_generation_instructions() -> str:
    """DailyDialogのRandom-DPO生成で使う翻訳・低品質応答生成指示を返す。"""
    return (
        "あなたはDPO学習用の日本語雑談データ作成者です。"
        "英語の会話文脈と次の応答を、日本人同士の自然な雑談として使える"
        "prompt/chosen/rejected候補へ変換してください。\n\n"
        "翻訳方針:\n"
        "- 直訳ではなく、日本語として自然な会話にしてください。\n"
        "- 元のchosenが持つ話題、相手への返答意図、会話の流れは保ってください。\n"
        "- promptは過去の会話文脈として自然に読めるよう日本語化し、話者ラベルは意味が分かる形にしてください。\n"
        "- chosenは同じ文脈に対する自然で短い次の返答にしてください。長すぎる説明や過度な創作は避けてください。\n\n"
        "rejected候補の生成方針:\n"
        "- rejectedは同じtranslated_promptに対する返答として作ってください。\n"
        "- 文法的に破綻した返答、攻撃的な返答、安全性に問題がある返答は作らないでください。\n"
        "- 一般的な雑談としてchosenより明確に品質が低いが、DPO学習の比較対象として読める返答にしてください。\n"
        "- chosenの単なる短縮、同義表現、語尾だけの変更は禁止です。\n"
        "- 候補ごとに弱点を分散してください。例: 文脈を無視する、短すぎて情報が不足する、"
        "相手の発話に答えていない、不自然または機械的、根拠のない決めつけを含む、"
        "会話を終わらせてしまう、相手の意図とずれている。\n"
        "- rejectedも日本語としては自然で、chosenとの品質差が分かる1〜2文にしてください。\n\n"
        "出力はJSONのみで、translated_prompt, translated_chosen, rejected_candidates, "
        "chosen_quality_score を含めてください。rejected_candidatesは文字列配列です。"
        "chosen_quality_scoreは、文脈に沿った自然な返答としての品質を0.0〜1.0で付けてください。"
    )


def build_medical_general_quality_generation_instructions() -> str:
    """MediTOD Random-DPO用に、選別戦略を教えず医療情報だけ保つ指示を返す。"""
    return (
        "あなたはDPO学習用の日本語医療相談データ作成者です。"
        "英語の医療相談文脈と次の応答を、自然な日本語のprompt/chosen/rejectedへ変換してください。\n\n"
        "翻訳方針:\n"
        "- 元の話題、応答意図、会話順序を保持してください。\n"
        "- 否定、発症時期、期間、数値、単位、薬剤名、症状名を省略・反転しないでください。\n"
        "- 原文にない診断、助言、緊急性、安全網を追加しないでください。\n"
        "- 薬剤名や固有の医学用語は必要なら原語を括弧内に残してください。\n"
        "- chosenは同じ文脈に対する自然で簡潔な医療者側の応答にしてください。\n\n"
        "rejected候補の生成方針:\n"
        "- rejectedは同じtranslated_promptに対する応答としてください。\n"
        "- 危険な診断・投薬、攻撃的表現、虚偽情報を作らないでください。\n"
        "- 一般的な応答品質がchosenより弱いが、日本語として読める安全な応答を作ってください。\n"
        "- chosenの単なる言い換えや壊れた文章は禁止です。\n\n"
        "出力はJSONのみで、translated_prompt, translated_chosen, rejected_candidates, "
        "chosen_quality_scoreを含めてください。"
    )


def build_general_quality_generation_input(
    record: dict[str, Any],
    *,
    candidates: int,
    seed: int,
) -> str:
    """翻訳・rejected生成用の入力を作る。"""
    return (
        "json output only.\n"
        f"seed: {seed}\n"
        f"rejected_candidates_count: {candidates}\n"
        f"source_dialogue_id: {record.get('conversation_id')}\n"
        f"turn_index: {record.get('turn_index')}\n\n"
        f"english_prompt:\n{record.get('prompt', '')}\n\n"
        f"english_chosen_response:\n{record.get('response', '')}"
    )


def validate_general_quality_payload(
    payload: dict[str, Any],
    *,
    candidates: int,
) -> dict[str, Any]:
    """LLMが返したRandom-DPO生成JSONを検証する。"""
    translated_prompt = str(payload.get("translated_prompt", "")).strip()
    translated_chosen = str(payload.get("translated_chosen", "")).strip()
    rejected_candidates = payload.get("rejected_candidates")
    if not translated_prompt:
        raise ValueError("`translated_prompt` が空です。")
    if not translated_chosen:
        raise ValueError("`translated_chosen` が空です。")
    if not isinstance(rejected_candidates, list):
        raise ValueError("`rejected_candidates` は配列である必要があります。")

    rejected_texts: list[str] = []
    seen: set[str] = set()
    for item in rejected_candidates:
        text = str(item).strip()
        if not text or text == translated_chosen or text in seen:
            continue
        rejected_texts.append(text)
        seen.add(text)
    if len(rejected_texts) < candidates:
        raise ValueError("rejected候補数が不足しています。")

    quality = payload.get("chosen_quality_score", 0.0)
    if not isinstance(quality, (int, float)):
        raise ValueError("`chosen_quality_score` は数値である必要があります。")
    return {
        "translated_prompt": translated_prompt,
        "translated_chosen": translated_chosen,
        "rejected_candidates": rejected_texts,
        "chosen_quality_score": max(0.0, min(1.0, float(quality))),
    }
