"""ESConvを専用分析用の会話単位JSONLへ変換する。"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


DEFAULT_DATASET_NAME = "thu-coai/esconv"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_PATH = "data/esconv_analysis_corpus.jsonl"
DEFAULT_MAX_CONVERSATIONS = 30
DEFAULT_SEED = 42
DEFAULT_SAMPLING = "stratified"
SPLIT_ORDER = ("train", "validation", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="ESConvを専用分析用JSONLへ変換します。")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help=f"Hugging Face dataset名（既定: {DEFAULT_DATASET_NAME}）。")
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=(*SPLIT_ORDER, "all"),
        help=f"読み込むsplit（既定: {DEFAULT_SPLIT}）。",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"出力JSONL（既定: {DEFAULT_OUTPUT_PATH}）。")
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=DEFAULT_MAX_CONVERSATIONS,
        help="変換する会話数の上限。0以下で全件を変換します。",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"サンプリング乱数seed（既定: {DEFAULT_SEED}）。")
    parser.add_argument(
        "--sampling",
        choices=("stratified", "random"),
        default=DEFAULT_SAMPLING,
        help="会話抽出方法。stratifiedはstrategy/emotion/problemの網羅を優先します。",
    )
    parser.add_argument("--dry-run", action="store_true", help="書き出さず、作成件数だけ確認します。")
    return parser.parse_args()


def load_esconv_dataset(dataset_name: str = DEFAULT_DATASET_NAME) -> Any:
    """Hugging Face datasetsからESConvを読み込む。"""
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "hf_cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "hf_cache" / "datasets"))
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("ESConvの読み込みには `datasets` パッケージが必要です。") from exc
    return load_dataset(dataset_name)


def _parse_text_payload(row: dict[str, Any], *, row_index: int) -> dict[str, Any]:
    """Hugging Face版ESConvのtext列に入ったJSON文字列をdictへ変換する。"""
    raw_text = row.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError(f"{row_index}件目の `text` が空です。")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{row_index}件目の `text` をJSONとして読めません: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{row_index}件目の `text` JSONはオブジェクトである必要があります。")
    return payload


def _normalise_speaker(raw_speaker: Any) -> str:
    """ESConvの話者表記を既存研究用表記へ変換する。"""
    speaker = str(raw_speaker).strip().lower()
    if speaker == "usr":
        return "user"
    if speaker == "sys":
        return "assistant"
    return speaker or "unknown"


def _normalise_dialog(payload: dict[str, Any], *, row_index: int) -> list[dict[str, Any]]:
    """ESConvのdialog配列を分析用turn配列へ変換する。"""
    raw_dialog = payload.get("dialog")
    if not isinstance(raw_dialog, list) or not raw_dialog:
        raise ValueError(f"{row_index}件目に有効な `dialog` 配列がありません。")
    turns: list[dict[str, Any]] = []
    for turn_index, raw_turn in enumerate(raw_dialog, start=1):
        if not isinstance(raw_turn, dict):
            raise ValueError(f"{row_index}件目 dialog[{turn_index}] はオブジェクトである必要があります。")
        text = str(raw_turn.get("text", "")).strip()
        if not text:
            continue
        speaker = _normalise_speaker(raw_turn.get("speaker"))
        turn: dict[str, Any] = {
            "turn_index": turn_index,
            "speaker": speaker,
            "text": text,
        }
        strategy = str(raw_turn.get("strategy", "")).strip()
        if speaker == "assistant" and strategy:
            turn["strategy"] = strategy
        turns.append(turn)
    if len(turns) < 2:
        raise ValueError(f"{row_index}件目の有効発話数が少なすぎます。")
    return turns


def parse_esconv_row(row: dict[str, Any], *, split: str, row_index: int) -> dict[str, Any]:
    """ESConvの1会話を分析用の会話単位レコードへ変換する。"""
    payload = _parse_text_payload(row, row_index=row_index)
    return {
        "conversation_id": f"esconv_{split}_{row_index:06d}",
        "source_dataset": "ESConv",
        "source_split": split,
        "source_dialogue_index": row_index,
        "experience_type": payload.get("experience_type", ""),
        "emotion_type": payload.get("emotion_type", ""),
        "problem_type": payload.get("problem_type", ""),
        "situation": payload.get("situation", ""),
        "survey_score": payload.get("survey_score", {}),
        "seeker_question1": payload.get("seeker_question1", ""),
        "seeker_question2": payload.get("seeker_question2", ""),
        "supporter_question1": payload.get("supporter_question1", ""),
        "supporter_question2": payload.get("supporter_question2", ""),
        "dialog": _normalise_dialog(payload, row_index=row_index),
    }


def _row_strata(row: dict[str, Any], *, row_index: int) -> set[str]:
    """層化サンプリング用に、1会話が含むstrategy/emotion/problemラベルを返す。"""
    payload = _parse_text_payload(row, row_index=row_index)
    strata: set[str] = set()
    for key in ("emotion_type", "problem_type", "experience_type"):
        value = str(payload.get(key, "")).strip().lower()
        if value:
            strata.add(f"{key}:{value}")
    dialog = payload.get("dialog", [])
    if isinstance(dialog, list):
        for turn in dialog:
            if not isinstance(turn, dict):
                continue
            if _normalise_speaker(turn.get("speaker")) != "assistant":
                continue
            strategy = str(turn.get("strategy", "")).strip().lower()
            if strategy:
                strata.add(f"strategy:{strategy}")
    return strata or {"unlabeled"}


def _selected_indices(total: int, *, max_conversations: int | None, seed: int) -> set[int]:
    """再現可能な会話サンプルのindex集合を返す。"""
    if max_conversations is None or max_conversations <= 0 or max_conversations >= total:
        return set(range(total))
    rng = random.Random(seed)
    return set(rng.sample(range(total), max_conversations))


def _stratified_selected_indices(
    rows: list[dict[str, Any]],
    *,
    max_conversations: int | None,
    seed: int,
) -> set[int]:
    """strategy/emotion/problemの網羅を優先した会話index集合を返す。"""
    total = len(rows)
    if max_conversations is None or max_conversations <= 0 or max_conversations >= total:
        return set(range(total))

    rng = random.Random(seed)
    labels_by_index = {
        row_index: _row_strata(row, row_index=row_index)
        for row_index, row in enumerate(rows)
    }
    index_by_label: dict[str, list[int]] = {}
    for row_index, labels in labels_by_index.items():
        for label in labels:
            index_by_label.setdefault(label, []).append(row_index)
    for candidates in index_by_label.values():
        rng.shuffle(candidates)

    selected: set[int] = set()
    labels_by_coverage_priority = sorted(
        index_by_label,
        key=lambda label: (0 if label.startswith("strategy:") else 1, len(index_by_label[label]), label),
    )
    while len(selected) < max_conversations:
        changed = False
        for label in labels_by_coverage_priority:
            if len(selected) >= max_conversations:
                break
            candidate = next((item for item in index_by_label[label] if item not in selected), None)
            if candidate is None:
                continue
            selected.add(candidate)
            changed = True
        if not changed:
            break

    if len(selected) < max_conversations:
        remaining = [row_index for row_index in range(total) if row_index not in selected]
        selected.update(rng.sample(remaining, max_conversations - len(selected)))
    return selected


def convert_esconv_rows(
    rows: Any,
    *,
    split: str,
    max_conversations: int | None,
    seed: int,
    sampling: str = DEFAULT_SAMPLING,
) -> list[dict[str, Any]]:
    """ESConvのsplit行を会話単位JSONLレコードへ変換する。"""
    row_list = [dict(row) for row in rows]
    if sampling == "stratified":
        selected = _stratified_selected_indices(row_list, max_conversations=max_conversations, seed=seed)
    else:
        selected = _selected_indices(len(row_list), max_conversations=max_conversations, seed=seed)
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(row_list):
        if row_index not in selected:
            continue
        records.append(parse_esconv_row(row, split=split, row_index=row_index))
    return records


def convert_esconv_dataset(
    dataset: Any,
    *,
    split: str,
    max_conversations: int | None,
    seed: int,
    sampling: str = DEFAULT_SAMPLING,
) -> list[dict[str, Any]]:
    """DatasetDictから指定splitのESConv会話を変換する。"""
    if split != "all":
        return convert_esconv_rows(
            dataset[split],
            split=split,
            max_conversations=max_conversations,
            seed=seed,
            sampling=sampling,
        )

    all_rows: list[tuple[str, int, dict[str, Any]]] = []
    for split_name in SPLIT_ORDER:
        for row_index, row in enumerate(dataset[split_name]):
            all_rows.append((split_name, row_index, dict(row)))
    if sampling == "stratified":
        selected = _stratified_selected_indices(
            [row for _, _, row in all_rows],
            max_conversations=max_conversations,
            seed=seed,
        )
    else:
        selected = _selected_indices(len(all_rows), max_conversations=max_conversations, seed=seed)
    records: list[dict[str, Any]] = []
    for combined_index, (split_name, row_index, row) in enumerate(all_rows):
        if combined_index not in selected:
            continue
        records.append(parse_esconv_row(row, split=split_name, row_index=row_index))
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, int]:
    """変換結果の概要を返す。"""
    turns = [turn for record in records for turn in record.get("dialog", [])]
    assistant_turns = [turn for turn in turns if turn.get("speaker") == "assistant"]
    strategies = {turn.get("strategy") for turn in assistant_turns if turn.get("strategy")}
    return {
        "conversations": len(records),
        "turns": len(turns),
        "assistant_turns": len(assistant_turns),
        "user_turns": len([turn for turn in turns if turn.get("speaker") == "user"]),
        "strategy_types": len(strategies),
    }


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
    dataset = load_esconv_dataset(args.dataset_name)
    records = convert_esconv_dataset(
        dataset,
        split=args.split,
        max_conversations=args.max_conversations,
        seed=args.seed,
        sampling=args.sampling,
    )
    summary = summarize_records(records)
    if args.dry_run:
        print("ESConv変換 dry-run")
        print(f"  dataset: {args.dataset_name}")
        print(f"  split: {args.split}")
        print(f"  seed: {args.seed}")
        print(f"  sampling: {args.sampling}")
        print(f"  max_conversations: {args.max_conversations}")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return 0
    write_jsonl(records, args.output)
    print(f"ESConv分析用JSONLを書き出しました: {args.output} ({summary['conversations']} 会話, {summary['turns']} 発話)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
