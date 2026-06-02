"""DailyDialogを状態遷移ベイズスコアリング用JSONLへ変換する。"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_DATASET_NAME = "ConvLab/dailydialog"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_PATH = "data/dailydialog_for_scoring.jsonl"
DEFAULT_MAX_CONTEXT_TURNS = 8


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="DailyDialogを文脈付き応答評価用JSONLへ変換します。")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help=f"Hugging Face dataset名（既定: {DEFAULT_DATASET_NAME}）。")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help=f"読み込むsplit（既定: {DEFAULT_SPLIT}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"出力JSONL（既定: {DEFAULT_OUTPUT_PATH}）。")
    parser.add_argument("--max-dialogues", type=int, default=None, help="処理する対話数の上限。")
    parser.add_argument("--max-context-turns", type=int, default=DEFAULT_MAX_CONTEXT_TURNS, help="promptに含める直前発話数。")
    parser.add_argument("--dry-run", action="store_true", help="書き出さず、作成件数だけ確認します。")
    return parser.parse_args()


def load_dailydialog_dataset(dataset_name: str, split: str) -> Any:
    """Hugging Face datasetsからDailyDialogを読み込む。"""
    if dataset_name == "ConvLab/dailydialog":
        return load_convlab_dailydialog_split(dataset_name, split)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("DailyDialogの読み込みには `datasets` パッケージが必要です。") from exc
    try:
        return load_dataset(dataset_name, split=split)
    except RuntimeError as exc:
        if dataset_name == "daily_dialog" and "Dataset scripts are no longer supported" in str(exc):
            raise RuntimeError(
                "`daily_dialog` は現在の datasets では読み込めない形式です。"
                "`--dataset-name ConvLab/dailydialog` を指定するか、既定値のまま実行してください。"
            ) from exc
        raise


def load_convlab_dailydialog_split(dataset_name: str, split: str) -> list[dict[str, Any]]:
    """ConvLab版DailyDialogをdata.zipから直接読み込む。"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("ConvLab版DailyDialogの読み込みには `huggingface_hub` パッケージが必要です。") from exc

    zip_path = hf_hub_download(repo_id=dataset_name, filename="data.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("data/dialogues.json") as file:
            rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError("ConvLab版DailyDialogの `data/dialogues.json` は配列である必要があります。")
    split_rows = [dict(row) for row in rows if dict(row).get("data_split") == split]
    if not split_rows:
        raise ValueError(f"ConvLab版DailyDialogに split={split!r} のデータがありません。")
    return split_rows


def _normalise_turn(turn: Any) -> str:
    """DailyDialogの1発話を文字列へ正規化する。"""
    if isinstance(turn, dict):
        for key in ("utterance", "text", "dialogue", "response"):
            value = turn.get(key)
            if value is not None:
                return str(value).strip()
        return ""
    return str(turn).strip()


def _normalise_dialogue(row: dict[str, Any], *, row_index: int) -> list[str]:
    """DailyDialog系データセットの対話本文リストを取り出す。"""
    if "dialog" in row:
        raw_dialogue = row.get("dialog")
    elif "dialogue" in row:
        raw_dialogue = row.get("dialogue")
    else:
        raw_dialogue = row.get("turns")
    if raw_dialogue is None:
        raise ValueError(f"{row_index}件目に `dialog`, `dialogue`, `turns` のいずれもありません。")
    if not isinstance(raw_dialogue, list):
        raise ValueError(f"{row_index}件目の対話本文はリストである必要があります。")
    dialogue = [_normalise_turn(turn) for turn in raw_dialogue]
    dialogue = [turn for turn in dialogue if turn]
    if len(dialogue) < 2:
        return []
    return dialogue


def _speaker_label(turn_index_zero_based: int) -> str:
    """DailyDialogの交互発話に話者ラベルを付ける。"""
    return "speaker_a" if turn_index_zero_based % 2 == 0 else "speaker_b"


def build_context_prompt(dialogue: list[str], *, target_index: int, max_context_turns: int) -> str:
    """target_index直前までの文脈をpromptへ整形する。"""
    start_index = max(0, target_index - max_context_turns)
    lines = []
    for index in range(start_index, target_index):
        lines.append(f"{_speaker_label(index)}: {dialogue[index]}")
    return "\n".join(lines)


def convert_dailydialog_rows(
    rows: Any,
    *,
    split: str,
    max_dialogues: int | None,
    max_context_turns: int,
) -> list[dict[str, Any]]:
    """DailyDialog行を既存スコアリング用JSONLレコードへ変換する。"""
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if max_dialogues is not None and row_index >= max_dialogues:
            break
        row_dict = dict(row)
        dialogue = _normalise_dialogue(row_dict, row_index=row_index)
        if not dialogue:
            continue
        conversation_id = f"{split}_{row_index:06d}"
        for target_index in range(1, len(dialogue)):
            prompt = build_context_prompt(
                dialogue,
                target_index=target_index,
                max_context_turns=max_context_turns,
            )
            records.append(
                {
                    "conversation_id": conversation_id,
                    "turn_index": target_index + 1,
                    "prompt": prompt,
                    "response": dialogue[target_index],
                    "metadata": {
                        "source_dataset": "DailyDialog",
                        "source_split": split,
                        "source_dialogue_index": row_index,
                        "response_speaker": _speaker_label(target_index),
                        "context_turns": target_index - max(0, target_index - max_context_turns),
                    },
                }
            )
    return records


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
    dataset = load_dailydialog_dataset(args.dataset_name, args.split)
    records = convert_dailydialog_rows(
        dataset,
        split=args.split,
        max_dialogues=args.max_dialogues,
        max_context_turns=args.max_context_turns,
    )
    if args.dry_run:
        print("DailyDialog変換 dry-run")
        print(f"  dataset: {args.dataset_name}")
        print(f"  split: {args.split}")
        print(f"  records: {len(records)}")
        return 0
    write_jsonl(records, args.output)
    print(f"DailyDialog評価用JSONLを書き出しました: {args.output} ({len(records)} 件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
