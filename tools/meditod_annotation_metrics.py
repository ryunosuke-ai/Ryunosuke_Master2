"""MediTOD公式annotationに基づく副次的な応答構造指標を算出する。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tools.wildchat_tutoring import tokenize


QUESTION_PATTERN = re.compile(r"[?？]|(?:ですか|ますか|でしょうか|教えてください|ありますか)[。！]?$")


def dialogue_tokens(text: str) -> set[str]:
    """英数字語と日本語を含む文字bigramを重複質問比較へ使う。"""
    normalized = re.sub(r"[^\w一-龥ぁ-んァ-ヶ]+", "", text.casefold())
    character_bigrams = {
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
    }
    return tokenize(text) | character_bigrams


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def duplicate_question_similarity(history: list[dict[str, Any]], response: str) -> float:
    response_tokens = dialogue_tokens(response)
    previous = [
        dialogue_tokens(str(turn.get("text", "")))
        for turn in history
        if turn.get("role") == "assistant" and QUESTION_PATTERN.search(str(turn.get("text", "")))
    ]
    similarities = []
    for tokens in previous:
        union = response_tokens | tokens
        similarities.append(len(response_tokens & tokens) / len(union) if union else 0.0)
    return max(similarities, default=0.0)


def infer_action(response: str) -> str:
    text = response.strip()
    if QUESTION_PATTERN.search(text):
        return "inquire"
    if re.search(r"(?:可能性|考えられ|疑われ|診断)", text):
        return "assessment"
    if re.search(r"(?:まとめると|つまり|これまで)", text):
        return "summary"
    return "inform"


def compute(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        source_intents = {str(value).lower() for value in row.get("source_response_intents", [])}
        source_inquire = "inquire" in source_intents
        for model, key in (
            ("base", "base_response"),
            ("basis", "basis_response"),
            ("random_dpo", "random_dpo_response"),
        ):
            response = str(row.get(key, ""))
            action = infer_action(response)
            output.append(
                {
                    "sample_id": row["sample_id"],
                    "conversation_id": row["conversation_id"],
                    "selection_stratum": row["selection_stratum"],
                    "model_name": model,
                    "source_intents": ",".join(sorted(source_intents)),
                    "source_slots": ",".join(row.get("source_response_slots", [])),
                    "predicted_action": action,
                    "intent_alignment": int((action == "inquire") == source_inquire),
                    "question_count": len(re.findall(r"[?？]", response)),
                    "duplicate_question_similarity": duplicate_question_similarity(row.get("history_ja", []), response),
                    "duplicate_question_flag": int(duplicate_question_similarity(row.get("history_ja", []), response) >= 0.75),
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="MediTOD annotation副次指標")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    rows = compute(read_jsonl(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    summary = {}
    for model in ("base", "basis", "random_dpo"):
        values = [row for row in rows if row["model_name"] == model]
        summary[model] = {
            "records": len(values),
            "intent_alignment_rate": sum(row["intent_alignment"] for row in values) / len(values) if values else 0.0,
            "mean_duplicate_question_similarity": sum(row["duplicate_question_similarity"] for row in values) / len(values) if values else 0.0,
            "duplicate_question_rate": sum(row["duplicate_question_flag"] for row in values) / len(values) if values else 0.0,
            "action_distribution": dict(Counter(row["predicted_action"] for row in values)),
        }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
