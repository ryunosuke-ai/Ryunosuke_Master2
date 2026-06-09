"""DailyDialogをランダム抽出してRandom-DPO baselineデータを作る。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.dpo_prompting import DPO_PROMPT_TEMPLATE_VERSION, build_dpo_prompt_from_context_text
from core.random_dpo_prompting import (
    GENERAL_QUALITY_STYLE_PRESET,
    RANDOM_DPO_PROMPT_TEMPLATE_VERSION,
    build_general_quality_generation_input,
    build_general_quality_generation_instructions,
    validate_general_quality_payload,
)
from tools.analyze_small_corpus import TextGenerator
from tools.prepare_dailydialog_for_scoring import (
    DEFAULT_DATASET_NAME,
    DEFAULT_MAX_CONTEXT_TURNS,
    DEFAULT_SPLIT,
    convert_dailydialog_rows,
    load_dailydialog_dataset,
)
from tools.score_dialogue_with_bayes_model import (
    OpenAIResponsesGenerator,
    extract_json_object,
    load_env_file,
    resolve_scoring_model,
)


DEFAULT_RUN_TAG = "esconv_5000_to_2000_random2500"
DEFAULT_TARGET_RECORDS = 2500
DEFAULT_MAX_DIALOGUES = 8000
DEFAULT_CANDIDATES = 4
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_SEED = 42
DEFAULT_DAILY_OUTPUT_PATH = (
    "artifacts/datasets/"
    f"dailydialog_ja_dpo_preferences_random2500_{DEFAULT_RUN_TAG}_daily.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    "artifacts/datasets/"
    f"dailydialog_random2500_ja_dpo_preferences_{DEFAULT_RUN_TAG}.jsonl"
)
NOT_USED_RANDOM_BASELINE = "not_used_random_baseline"


def env_int(name: str, default: int) -> int:
    """整数環境変数を読む。"""
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    load_env_file()
    target_records = env_int("RANDOM_DPO_TARGET_RECORDS", DEFAULT_TARGET_RECORDS)
    seed = env_int("RANDOM_DPO_SEED", DEFAULT_SEED)
    parser = argparse.ArgumentParser(
        description="DailyDialogからランダム抽出したRandom-DPO baseline JSONLを作成します。"
    )
    parser.add_argument("--input", default="", help="準備済みDailyDialog JSONL。未指定ならHF datasetから読み込みます。")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--start-dialogue", type=int, default=0)
    parser.add_argument("--max-dialogues", type=int, default=DEFAULT_MAX_DIALOGUES)
    parser.add_argument("--max-context-turns", type=int, default=DEFAULT_MAX_CONTEXT_TURNS)
    parser.add_argument("--daily-output", default=DEFAULT_DAILY_OUTPUT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest-output", default="")
    parser.add_argument("--model", default=resolve_scoring_model())
    parser.add_argument("--target-records", type=int, default=target_records)
    parser.add_argument("--max-source-records", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--skip-sample-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、抽出対象だけ確認します。")
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
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{line_number}行目をJSONとして読めません: {exc}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"{line_number}行目はJSON objectである必要があります。")
                records.append(validate_source_record(payload, line_number=line_number))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"入力JSONLが見つかりません: {input_path}") from exc
    if not records:
        raise ValueError("入力JSONLに有効なレコードがありません。")
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path | str) -> None:
    """JSONLを書き出す。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(payload: dict[str, Any], path: Path | str) -> None:
    """JSONを書き出す。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_source_record(record: dict[str, Any], *, line_number: int) -> dict[str, Any]:
    """DailyDialog候補レコードの必須列を検証する。"""
    for key in ("conversation_id", "turn_index", "prompt", "response"):
        if not str(record.get(key, "")).strip():
            raise ValueError(f"{line_number}行目の `{key}` が空です。")
    metadata = record.setdefault("metadata", {})
    metadata.setdefault("source_dataset", "DailyDialog")
    return record


def source_key(record: dict[str, Any]) -> tuple[str, int]:
    """元候補の一意キーを返す。"""
    return str(record["conversation_id"]), int(record["turn_index"])


def stable_source_hash(record: dict[str, Any]) -> str:
    """source keyの短い安定ハッシュを返す。"""
    key = f"{record.get('conversation_id')}:{record.get('turn_index')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def load_source_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """DailyDialog候補を読み込む。"""
    if args.input:
        return read_jsonl(args.input)
    rows = load_dailydialog_dataset(args.dataset_name, args.split)
    return convert_dailydialog_rows(
        rows,
        split=args.split,
        start_dialogue=args.start_dialogue,
        max_dialogues=args.max_dialogues,
        max_context_turns=args.max_context_turns,
    )


def randomize_source_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    max_source_records: int | None,
) -> list[dict[str, Any]]:
    """候補をseed固定でランダム順に並べる。"""
    randomized = list(records)
    random.Random(seed).shuffle(randomized)
    if max_source_records is not None:
        randomized = randomized[:max_source_records]
    return randomized


def generate_general_quality_payload(
    *,
    source_record: dict[str, Any],
    index: int,
    generator: TextGenerator,
    instructions: str,
    model: str,
    max_output_tokens: int,
    candidates: int,
    seed: int,
) -> dict[str, Any]:
    """1件分の日本語chosen/rejected候補を生成する。"""
    output_text = generator.generate(
        instructions=instructions,
        input_text=build_general_quality_generation_input(
            source_record,
            candidates=candidates,
            seed=seed + index,
        ),
        model=model,
        max_output_tokens=max_output_tokens,
        response_text_format={"type": "json_object"},
    )
    return validate_general_quality_payload(
        extract_json_object(output_text),
        candidates=candidates,
    )


def build_random_dpo_record(
    source_record: dict[str, Any],
    *,
    index: int,
    generation_payload: dict[str, Any],
    model: str,
    seed: int,
    candidates: int,
) -> dict[str, Any]:
    """Random-DPOの1レコードを既存DPO学習schemaへ変換する。"""
    translated_prompt = generation_payload["translated_prompt"]
    translated_chosen = generation_payload["translated_chosen"]
    rejected_text = generation_payload["rejected_candidates"][0]
    dpo_prompt = build_dpo_prompt_from_context_text(translated_prompt)
    metadata = dict(source_record.get("metadata", {}))
    metadata.update(
        {
            "source_dataset": "DailyDialog",
            "selection_method": "random",
            "random_seed": seed,
            "random_index": index,
            "source_hash": stable_source_hash(source_record),
            "style_preset": GENERAL_QUALITY_STYLE_PRESET,
            "generation_model": model,
            "prompt_template": RANDOM_DPO_PROMPT_TEMPLATE_VERSION,
            "dpo_prompt_template": DPO_PROMPT_TEMPLATE_VERSION,
            "rejected_candidates": candidates,
            "raw_translated_prompt": translated_prompt,
            "esconv_gold_records": 0,
        }
    )
    return {
        "prompt": dpo_prompt,
        "chosen": translated_chosen,
        "rejected": rejected_text,
        "score_chosen": 1.0,
        "score_rejected": 0.0,
        "score_gap": 1.0,
        "source_dataset": "DailyDialog",
        "source_dialogue_id": source_record.get("conversation_id"),
        "turn_index": source_record.get("turn_index"),
        "history_turns": source_record.get("metadata", {}).get("context_turns"),
        "translated_chosen": translated_chosen,
        "translated_rejected": rejected_text,
        "state_sequence": [],
        "strategy_sequence": [],
        "reward_breakdown": {
            "chosen": {"general_quality_score": generation_payload["chosen_quality_score"]},
            "rejected": {"general_quality_score": 0.0},
        },
        "translation_quality_score": generation_payload["chosen_quality_score"],
        "raw_translated_prompt": translated_prompt,
        "generation_retry": None,
        "model_used_for_scoring": NOT_USED_RANDOM_BASELINE,
        "model_used_for_translation": model,
        "model_used_for_rejected_generation": model,
        "bayesian_model_version": NOT_USED_RANDOM_BASELINE,
        "prompt_template_version": RANDOM_DPO_PROMPT_TEMPLATE_VERSION,
        "dpo_prompt_template_version": DPO_PROMPT_TEMPLATE_VERSION,
        "source_prompt_en": source_record.get("prompt"),
        "source_chosen_en": source_record.get("response"),
        "metadata": metadata,
    }


def build_random_dpo_records(
    source_records: list[dict[str, Any]],
    *,
    generator: TextGenerator,
    model: str,
    max_output_tokens: int,
    candidates: int,
    target_records: int,
    workers: int,
    seed: int,
    skip_sample_errors: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """ランダム順のDailyDialog候補からDPOレコードを作る。"""
    instructions = build_general_quality_generation_instructions()
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    processed: set[tuple[str, int]] = set()
    indexed_records = list(enumerate(source_records, start=1))

    def build_one(index: int, source_record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        try:
            payload = generate_general_quality_payload(
                source_record=source_record,
                index=index,
                generator=generator,
                instructions=instructions,
                model=model,
                max_output_tokens=max_output_tokens,
                candidates=candidates,
                seed=seed,
            )
            return (
                build_random_dpo_record(
                    source_record,
                    index=index,
                    generation_payload=payload,
                    model=model,
                    seed=seed,
                    candidates=candidates,
                ),
                None,
            )
        except Exception as exc:
            if not skip_sample_errors:
                raise
            print(
                "[RANDOM-DPO] skip sample "
                f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return None, "sample_error"

    for chunk_start in range(0, len(indexed_records), max(1, workers)):
        if len(records) >= target_records:
            break
        chunk = indexed_records[chunk_start : chunk_start + max(1, workers)]
        if workers <= 1:
            results = [(source_record, *build_one(index, source_record)) for index, source_record in chunk]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(build_one, index, source_record): source_record
                    for index, source_record in chunk
                }
                for future in as_completed(futures):
                    source_record = futures[future]
                    record, skip_reason = future.result()
                    results.append((source_record, record, skip_reason))
        for source_record, record, skip_reason in results:
            key = source_key(source_record)
            if key in processed:
                continue
            processed.add(key)
            if record is None:
                skipped[skip_reason or "unknown"] += 1
                continue
            records.append(record)
            print(
                "[RANDOM-DPO] accepted "
                f"{len(records)}/{target_records} "
                f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}",
                flush=True,
            )
            if len(records) >= target_records:
                break
    return records[:target_records], skipped


def count_by_source_dataset(records: list[dict[str, Any]]) -> dict[str, int]:
    """source_dataset別件数を返す。"""
    counter: Counter[str] = Counter()
    for record in records:
        source_dataset = str(
            record.get("source_dataset")
            or record.get("metadata", {}).get("source_dataset")
            or ""
        )
        counter[source_dataset] += 1
    return dict(counter)


def manifest_payload(
    *,
    args: argparse.Namespace,
    source_count: int,
    randomized_count: int,
    records: list[dict[str, Any]],
    skipped: Counter[str],
) -> dict[str, Any]:
    """再現性manifestを作る。"""
    return {
        "mode": "random_dailydialog_dpo_baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "start_dialogue": args.start_dialogue,
        "max_dialogues": args.max_dialogues,
        "max_context_turns": args.max_context_turns,
        "daily_output": args.daily_output,
        "output": args.output,
        "generation_model": args.model,
        "target_records": args.target_records,
        "records_written": len(records),
        "source_candidates": source_count,
        "randomized_candidates": randomized_count,
        "seed": args.seed,
        "workers": max(1, args.workers),
        "candidates": args.candidates,
        "prompt_template": RANDOM_DPO_PROMPT_TEMPLATE_VERSION,
        "dpo_prompt_template": DPO_PROMPT_TEMPLATE_VERSION,
        "style_preset": GENERAL_QUALITY_STYLE_PRESET,
        "selection_method": "random",
        "source_dataset_counts": count_by_source_dataset(records),
        "daily_dialog_random_records": count_by_source_dataset(records).get("DailyDialog", 0),
        "esconv_gold_records": 0,
        "bayes_selection_used": False,
        "bayes_model_used": False,
        "skip_sample_errors": args.skip_sample_errors,
        "skipped": dict(skipped),
    }


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    source_records = load_source_records(args)
    randomized_records = randomize_source_records(
        source_records,
        seed=args.seed,
        max_source_records=args.max_source_records,
    )
    if args.dry_run:
        print("Random DailyDialog DPO dry-run")
        print(f"  source_candidates: {len(source_records)}")
        print(f"  randomized_candidates: {len(randomized_records)}")
        print(f"  target_records: {args.target_records}")
        print(f"  seed: {args.seed}")
        print(f"  model: {args.model}")
        print(f"  dpo_prompt_template: {DPO_PROMPT_TEMPLATE_VERSION}")
        print("  esconv_gold_records: 0")
        return 0

    records, skipped = build_random_dpo_records(
        randomized_records,
        generator=OpenAIResponsesGenerator(),
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        candidates=args.candidates,
        target_records=args.target_records,
        workers=max(1, args.workers),
        seed=args.seed,
        skip_sample_errors=args.skip_sample_errors,
    )
    if len(records) < args.target_records:
        raise RuntimeError(
            f"target_records={args.target_records} に届きませんでした: records={len(records)}"
        )
    write_jsonl(records, args.output)
    if args.daily_output and Path(args.daily_output) != Path(args.output):
        write_jsonl(records, args.daily_output)
    manifest_path = args.manifest_output or str(Path(args.output).with_suffix(".manifest.json"))
    write_json(
        manifest_payload(
            args=args,
            source_count=len(source_records),
            randomized_count=len(randomized_records),
            records=records,
            skipped=skipped,
        ),
        manifest_path,
    )
    print(f"Random DailyDialog DPO JSONLを書き出しました: {args.output} ({len(records)} 件)")
    if args.daily_output and Path(args.daily_output) != Path(args.output):
        print(f"DailyDialog random DPO JSONLを書き出しました: {args.daily_output}")
    print(f"再現性manifestを書き出しました: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
