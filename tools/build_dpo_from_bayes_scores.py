"""ベイズスコア済み対話からDPO学習用JSONLを作成する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = "artifacts/scored_dialogues/bayes_scored_dialogue.jsonl"
DEFAULT_OUTPUT_PATH = "artifacts/datasets/bayes_dpo_preferences.jsonl"
DEFAULT_MIN_CHOSEN_POSTERIOR = 0.65
DEFAULT_MAX_REJECTED_POSTERIOR = 0.35


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="ベイズスコア済み対話からDPO JSONLを作成します。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help=f"入力JSONL（既定: {DEFAULT_INPUT_PATH}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"出力DPO JSONL（既定: {DEFAULT_OUTPUT_PATH}）。")
    parser.add_argument("--min-chosen-posterior", type=float, default=DEFAULT_MIN_CHOSEN_POSTERIOR, help="chosen候補の最低posterior。")
    parser.add_argument("--max-rejected-posterior", type=float, default=DEFAULT_MAX_REJECTED_POSTERIOR, help="rejected候補の最高posterior。")
    parser.add_argument("--dry-run", action="store_true", help="書き出さず、作成件数だけ表示します。")
    return parser.parse_args()


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """JSONLを読み込む。"""
    input_path = Path(path)
    records: list[dict[str, Any]] = []
    try:
        with input_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{line_number}行目をJSONとして読めません: {exc}") from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"スコア済みJSONLが見つかりません: {input_path}") from exc
    if not records:
        raise ValueError("スコア済みJSONLに有効なレコードがありません。")
    return records


def _require_text(record: dict[str, Any], key: str) -> str:
    """必須テキスト列を取り出す。"""
    value = str(record.get(key, "")).strip()
    if not value:
        raise ValueError(f"`{key}` が空のレコードがあります。")
    return value


def _posterior(record: dict[str, Any]) -> float:
    """posteriorを数値として取り出す。"""
    value = record.get("posterior")
    if not isinstance(value, (int, float)):
        raise ValueError("`posterior` が数値でないレコードがあります。")
    return float(value)


def build_preference_records(
    scored_records: list[dict[str, Any]],
    *,
    min_chosen_posterior: float,
    max_rejected_posterior: float,
) -> list[dict[str, Any]]:
    """同じprompt内の高スコア応答と低スコア応答を組にしてDPOレコードを作る。"""
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for record in scored_records:
        prompt = _require_text(record, "prompt")
        by_prompt.setdefault(prompt, []).append(record)

    preferences: list[dict[str, Any]] = []
    for prompt, candidates in by_prompt.items():
        chosen_candidates = [
            record for record in candidates if _posterior(record) >= min_chosen_posterior
        ]
        rejected_candidates = [
            record for record in candidates if _posterior(record) <= max_rejected_posterior
        ]
        if not chosen_candidates or not rejected_candidates:
            continue
        chosen = max(chosen_candidates, key=_posterior)
        rejected = min(rejected_candidates, key=_posterior)
        chosen_text = _require_text(chosen, "response")
        rejected_text = _require_text(rejected, "response")
        if chosen_text == rejected_text:
            continue
        preferences.append(
            {
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "metadata": {
                    "chosen_conversation_id": chosen.get("conversation_id"),
                    "chosen_turn_index": chosen.get("turn_index"),
                    "chosen_posterior": _posterior(chosen),
                    "chosen_observation": chosen.get("observation"),
                    "rejected_conversation_id": rejected.get("conversation_id"),
                    "rejected_turn_index": rejected.get("turn_index"),
                    "rejected_posterior": _posterior(rejected),
                    "rejected_observation": rejected.get("observation"),
                },
            }
        )
    return preferences


def write_jsonl(records: list[dict[str, Any]], path: Path | str) -> None:
    """JSONLを書き出す。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    scored_records = read_jsonl(args.input)
    preferences = build_preference_records(
        scored_records,
        min_chosen_posterior=args.min_chosen_posterior,
        max_rejected_posterior=args.max_rejected_posterior,
    )
    if args.dry_run:
        print("bayes DPO dry-run")
        print(f"  scored_records: {len(scored_records)}")
        print(f"  preference_records: {len(preferences)}")
        return 0
    write_jsonl(preferences, args.output)
    print(f"DPO JSONLを書き出しました: {args.output} ({len(preferences)} 件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
