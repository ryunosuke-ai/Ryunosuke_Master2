"""追記中のスコア済みJSONLからDPOデータを逐次生成する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.transition_bayes_model import TransitionBayesModel, load_transition_bayes_model
from tools.extract_high_posterior_dialogues import (
    DEFAULT_EXCLUDED_OBSERVATIONS,
    DEFAULT_EXCLUDED_STATES,
    DEFAULT_LOW_PRIORITY_STATES,
    DEFAULT_MIN_POSTERIOR,
    DEFAULT_PREFERRED_OBSERVATIONS,
    DEFAULT_PREFERRED_STATES,
    select_high_posterior_records,
)
from tools.score_dialogue_with_bayes_model import (
    OpenAIResponsesGenerator,
    load_env_file,
    resolve_scoring_model,
)
from tools.translate_and_generate_dpo import (
    DEFAULT_CANDIDATES,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_REJECTED_POSTERIOR,
    DEFAULT_MIN_CHOSEN_POSTERIOR,
    DEFAULT_MIN_SCORE_GAP,
    DEFAULT_SEED,
    DEFAULT_STYLE_PRESET,
    PROMPT_TEMPLATE_VERSION,
    bayes_model_version,
    build_one_dpo_record,
    build_translation_rejected_instructions,
    read_existing_dpo_records,
)
from tools.jsonl_utils import ensure_jsonl_append_boundary, read_jsonl_records


DEFAULT_SCORED_INPUT = "artifacts/scored_dialogues/dailydialog_transition_scored.jsonl"
DEFAULT_SELECTED_OUTPUT = "artifacts/datasets/dailydialog_selected_en_streaming.jsonl"
DEFAULT_DPO_OUTPUT = "artifacts/datasets/dailydialog_ja_dpo_preferences_streaming.jsonl"
DEFAULT_BAYES_MODEL = "artifacts/bayes_models/generated_transition_bayes_model.json"
DEFAULT_POLL_SECONDS = 10.0
DEFAULT_STREAM_BATCH_SIZE = 1
EXIT_TARGET_REACHED = 0
EXIT_SOURCE_EXHAUSTED = 2


@dataclass
class StreamDpoConfig:
    """ストリーミングDPO生成の設定。"""

    scored_input: Path
    selected_output: Path
    dpo_output: Path
    bayes_model_path: Path
    generation_model: str
    score_model: str
    target_records: int | None
    workers: int = 1
    batch_size: int = DEFAULT_STREAM_BATCH_SIZE
    poll_seconds: float = DEFAULT_POLL_SECONDS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    candidates: int = DEFAULT_CANDIDATES
    min_score_gap: float = DEFAULT_MIN_SCORE_GAP
    min_chosen_posterior: float = DEFAULT_MIN_CHOSEN_POSTERIOR
    max_rejected_posterior: float = DEFAULT_MAX_REJECTED_POSTERIOR
    seed: int = DEFAULT_SEED
    style_preset: str = DEFAULT_STYLE_PRESET
    min_posterior: float = DEFAULT_MIN_POSTERIOR
    min_context_turns: int = 0
    per_dialogue_limit: int | None = None
    prefer_states: str = ",".join(DEFAULT_PREFERRED_STATES)
    prefer_observations: str = ",".join(DEFAULT_PREFERRED_OBSERVATIONS)
    low_priority_states: str = ",".join(DEFAULT_LOW_PRIORITY_STATES)
    exclude_states: str = ",".join(DEFAULT_EXCLUDED_STATES)
    exclude_observations: str = ",".join(DEFAULT_EXCLUDED_OBSERVATIONS)
    require_preferred: bool = False
    ledger_path: Path | None = None
    done_file: Path | None = None
    heartbeat_file: Path | None = None


@dataclass
class StreamState:
    """再開に必要な状態。"""

    dpo_records: list[dict[str, Any]] = field(default_factory=list)
    selected_keys: set[tuple[str, int]] = field(default_factory=set)
    processed_keys: set[tuple[str, int]] = field(default_factory=set)
    selected_counts_by_dialogue: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    seen_scored: int = 0
    selected_appended: int = 0
    attempted: int = 0
    accepted: int = 0
    skipped: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))


class JsonlTailReader:
    """追記中JSONLから、まだ読んでいない完全な行だけを読む。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0

    def read_available(self) -> list[dict[str, Any]]:
        """現在読める完全なJSONL行を返す。"""
        if not self.path.exists():
            return []
        if self.path.stat().st_size < self.offset:
            self.offset = 0

        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as file:
            file.seek(self.offset)
            while True:
                position = file.tell()
                line = file.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    file.seek(position)
                    break
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        print(
                            f"[STREAM] skip invalid scored JSONL line {self.path}: {exc}",
                            flush=True,
                        )
            self.offset = file.tell()
        return records


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    load_env_file()
    parser = argparse.ArgumentParser(
        description="追記中のスコア済みJSONLから日本語DPO JSONLを逐次生成します。"
    )
    parser.add_argument("--scored-input", default=DEFAULT_SCORED_INPUT)
    parser.add_argument("--selected-output", default=DEFAULT_SELECTED_OUTPUT)
    parser.add_argument("--dpo-output", default=DEFAULT_DPO_OUTPUT)
    parser.add_argument("--bayes-model", default=DEFAULT_BAYES_MODEL)
    parser.add_argument("--model", default=resolve_scoring_model())
    parser.add_argument("--score-model", default=resolve_scoring_model())
    parser.add_argument("--target-records", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_STREAM_BATCH_SIZE)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--min-score-gap", type=float, default=DEFAULT_MIN_SCORE_GAP)
    parser.add_argument(
        "--min-chosen-posterior",
        type=float,
        default=DEFAULT_MIN_CHOSEN_POSTERIOR,
    )
    parser.add_argument(
        "--max-rejected-posterior",
        type=float,
        default=DEFAULT_MAX_REJECTED_POSTERIOR,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--style-preset",
        choices=("reminiscence", "esconv_support"),
        default=DEFAULT_STYLE_PRESET,
    )
    parser.add_argument("--min-posterior", type=float, default=DEFAULT_MIN_POSTERIOR)
    parser.add_argument("--min-context-turns", type=int, default=0)
    parser.add_argument("--per-dialogue-limit", type=int, default=None)
    parser.add_argument("--prefer-states", default=",".join(DEFAULT_PREFERRED_STATES))
    parser.add_argument(
        "--prefer-observations",
        default=",".join(DEFAULT_PREFERRED_OBSERVATIONS),
    )
    parser.add_argument(
        "--low-priority-states",
        default=",".join(DEFAULT_LOW_PRIORITY_STATES),
    )
    parser.add_argument("--exclude-states", default=",".join(DEFAULT_EXCLUDED_STATES))
    parser.add_argument(
        "--exclude-observations",
        default=",".join(DEFAULT_EXCLUDED_OBSERVATIONS),
    )
    parser.add_argument("--require-preferred", action="store_true")
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--done-file", default=None)
    parser.add_argument("--heartbeat-file", default=None)
    parser.add_argument(
        "--once",
        action="store_true",
        help="現在読めるスコア済み行だけを処理して終了します。主にテスト用です。",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> StreamDpoConfig:
    """CLI引数から設定を作る。"""
    dpo_output = Path(args.dpo_output)
    ledger_path = Path(args.ledger) if args.ledger else dpo_output.with_suffix(".progress.jsonl")
    return StreamDpoConfig(
        scored_input=Path(args.scored_input),
        selected_output=Path(args.selected_output),
        dpo_output=dpo_output,
        bayes_model_path=Path(args.bayes_model),
        generation_model=args.model,
        score_model=args.score_model,
        target_records=args.target_records,
        workers=max(1, args.workers),
        batch_size=max(1, args.batch_size),
        poll_seconds=max(0.1, args.poll_seconds),
        max_output_tokens=args.max_output_tokens,
        candidates=args.candidates,
        min_score_gap=args.min_score_gap,
        min_chosen_posterior=args.min_chosen_posterior,
        max_rejected_posterior=args.max_rejected_posterior,
        seed=args.seed,
        style_preset=args.style_preset,
        min_posterior=args.min_posterior,
        min_context_turns=args.min_context_turns,
        per_dialogue_limit=args.per_dialogue_limit,
        prefer_states=args.prefer_states,
        prefer_observations=args.prefer_observations,
        low_priority_states=args.low_priority_states,
        exclude_states=args.exclude_states,
        exclude_observations=args.exclude_observations,
        require_preferred=args.require_preferred,
        ledger_path=ledger_path,
        done_file=Path(args.done_file) if args.done_file else None,
        heartbeat_file=Path(args.heartbeat_file) if args.heartbeat_file else None,
    )


def read_jsonl_if_exists(path: Path | str) -> list[dict[str, Any]]:
    """存在するJSONLを読む。存在しなければ空配列を返す。"""
    records, skipped = read_jsonl_records(
        path,
        missing_ok=True,
        strict=False,
        label="既存ストリーミングJSONL",
    )
    if skipped:
        print(f"[STREAM] skip invalid existing JSONL lines path={path} skipped={skipped}", flush=True)
    return [record for record in records if isinstance(record, dict)]


def append_jsonl(record: dict[str, Any], path: Path | str) -> None:
    """JSONLへ1行追記する。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_jsonl_append_boundary(output_path)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(records: list[dict[str, Any]], path: Path | str) -> None:
    """JSONLを書き換える。"""
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


def source_key(record: dict[str, Any]) -> tuple[str, int]:
    """スコア済み・抽出済みレコードのキーを返す。"""
    return str(record["conversation_id"]), int(record["turn_index"])


def dpo_source_key(record: dict[str, Any]) -> tuple[str, int]:
    """DPOレコードから元サンプルのキーを返す。"""
    return str(record["source_dialogue_id"]), int(record["turn_index"])


def stable_source_index(record: dict[str, Any]) -> int:
    """seedずらしに使う安定した整数を返す。"""
    key = f"{record.get('conversation_id')}:{record.get('turn_index')}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def target_reached(config: StreamDpoConfig, state: StreamState) -> bool:
    """目標DPO件数に達しているかを返す。"""
    return config.target_records is not None and len(state.dpo_records) >= config.target_records


def load_stream_state(config: StreamDpoConfig) -> StreamState:
    """既存出力とledgerから再開状態を復元する。"""
    state = StreamState()

    for record in read_jsonl_if_exists(config.selected_output):
        key = source_key(record)
        if key in state.selected_keys:
            continue
        state.selected_keys.add(key)
        state.selected_counts_by_dialogue[key[0]] += 1

    state.dpo_records = dedupe_dpo_records(read_existing_dpo_records(config.dpo_output))
    for record in state.dpo_records:
        state.processed_keys.add(dpo_source_key(record))

    if config.ledger_path is not None:
        for record in read_jsonl_if_exists(config.ledger_path):
            conversation_id = record.get("conversation_id")
            turn_index = record.get("turn_index")
            if conversation_id is None or turn_index is None:
                continue
            state.processed_keys.add((str(conversation_id), int(turn_index)))

    state.accepted = len(state.dpo_records)
    return state


def select_stream_candidate(
    record: dict[str, Any],
    *,
    config: StreamDpoConfig,
    state: StreamState,
) -> dict[str, Any] | None:
    """スコア済み1件をストリーミングDPO候補にするか判定する。"""
    key = source_key(record)
    if key in state.processed_keys:
        return None

    selected = select_high_posterior_records(
        [record],
        min_posterior=config.min_posterior,
        max_records=1,
        sort_by_posterior=False,
        min_context_turns=config.min_context_turns,
        target_records=None,
        per_dialogue_limit=None,
        prefer_states=config.prefer_states,
        prefer_observations=config.prefer_observations,
        low_priority_states=config.low_priority_states,
        exclude_states=config.exclude_states,
        exclude_observations=config.exclude_observations,
        require_preferred=config.require_preferred,
        sort_by_selection=False,
    )
    if not selected:
        return None

    selected_record = selected[0]
    if key not in state.selected_keys:
        count = state.selected_counts_by_dialogue[key[0]]
        if config.per_dialogue_limit is not None and count >= config.per_dialogue_limit:
            return None
        append_jsonl(selected_record, config.selected_output)
        state.selected_keys.add(key)
        state.selected_counts_by_dialogue[key[0]] += 1
        state.selected_appended += 1

    return selected_record


def append_ledger(
    *,
    config: StreamDpoConfig,
    source_record: dict[str, Any],
    status: str,
    skip_reason: str | None = None,
    dpo_record: dict[str, Any] | None = None,
) -> None:
    """処理済みsource keyをledgerへ追記する。"""
    if config.ledger_path is None:
        return
    payload = {
        "conversation_id": source_record.get("conversation_id"),
        "turn_index": source_record.get("turn_index"),
        "status": status,
        "skip_reason": skip_reason,
        "score_gap": dpo_record.get("score_gap") if dpo_record else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(payload, config.ledger_path)


def handle_dpo_result(
    *,
    config: StreamDpoConfig,
    state: StreamState,
    source_record: dict[str, Any],
    dpo_record: dict[str, Any] | None,
    skip_reason: str | None,
) -> None:
    """DPO生成結果を出力とledgerへ反映する。"""
    key = source_key(source_record)
    state.attempted += 1
    if dpo_record is None:
        reason = skip_reason or "unknown"
        state.skipped[reason] += 1
        state.processed_keys.add(key)
        append_ledger(
            config=config,
            source_record=source_record,
            status="skipped",
            skip_reason=reason,
        )
        print(
            "[STREAM] skipped "
            f"{key[0]}#{key[1]} reason={reason} accepted={len(state.dpo_records)}",
            flush=True,
        )
        return

    if target_reached(config, state):
        return

    append_jsonl(dpo_record, config.dpo_output)
    state.dpo_records.append(dpo_record)
    state.processed_keys.add(key)
    state.accepted = len(state.dpo_records)
    append_ledger(
        config=config,
        source_record=source_record,
        status="accepted",
        dpo_record=dpo_record,
    )
    print(
        "[STREAM] accepted "
        f"{key[0]}#{key[1]} score_gap={float(dpo_record['score_gap']):.3f} "
        f"accepted={len(state.dpo_records)}",
        flush=True,
    )


def process_candidate_batch(
    candidates: list[dict[str, Any]],
    *,
    config: StreamDpoConfig,
    state: StreamState,
    bayes_model: TransitionBayesModel,
    generator: Any,
    instructions: str,
    model_version: str,
) -> None:
    """候補バッチをDPO生成へ回す。"""
    if not candidates or target_reached(config, state):
        return

    def build_for_record(source_record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        try:
            dpo_record, skip_reason, _skip_record = build_one_dpo_record(
                source_record,
                index=stable_source_index(source_record),
                bayes_model=bayes_model,
                model_version=model_version,
                generator=generator,
                instructions=instructions,
                model=config.generation_model,
                score_model=config.score_model,
                max_output_tokens=config.max_output_tokens,
                candidates=config.candidates,
                min_score_gap=config.min_score_gap,
                min_chosen_posterior=config.min_chosen_posterior,
                max_rejected_posterior=config.max_rejected_posterior,
                gap_rescue_max_rejected_posterior=None,
                gap_rescue_min_score_gap=None,
                seed=config.seed,
                style_preset=config.style_preset,
            )
            return dpo_record, skip_reason
        except Exception as exc:
            print(
                "[STREAM] skip dpo_generation_error "
                f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return None, "dpo_generation_error"

    if config.workers <= 1:
        for source_record in candidates:
            if target_reached(config, state):
                break
            dpo_record, skip_reason = build_for_record(source_record)
            handle_dpo_result(
                config=config,
                state=state,
                source_record=source_record,
                dpo_record=dpo_record,
                skip_reason=skip_reason,
            )
        return

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(build_for_record, source_record): source_record
            for source_record in candidates
        }
        for future in as_completed(futures):
            source_record = futures[future]
            dpo_record, skip_reason = future.result()
            handle_dpo_result(
                config=config,
                state=state,
                source_record=source_record,
                dpo_record=dpo_record,
                skip_reason=skip_reason,
            )


def dedupe_dpo_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """source keyごとにscore_gap最大のDPOレコードだけを残す。"""
    best_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = dpo_source_key(record)
        current = best_by_key.get(key)
        if current is None or float(record.get("score_gap", 0.0)) > float(
            current.get("score_gap", 0.0)
        ):
            best_by_key[key] = record
    return list(best_by_key.values())


def finalize_dpo_output(
    *,
    config: StreamDpoConfig,
    state: StreamState,
    model_version: str,
) -> None:
    """DPO出力をscore_gap順に整列し、manifestを書く。"""
    records = dedupe_dpo_records(read_jsonl_if_exists(config.dpo_output))
    records = sorted(records, key=lambda record: float(record["score_gap"]), reverse=True)
    if config.target_records is not None:
        records = records[: config.target_records]
    write_jsonl(records, config.dpo_output)
    state.dpo_records = records
    state.accepted = len(records)

    manifest_path = config.dpo_output.with_suffix(".manifest.json")
    write_json(
        {
            "mode": "streaming",
            "scored_input": str(config.scored_input),
            "selected_output": str(config.selected_output),
            "dpo_output": str(config.dpo_output),
            "bayes_model": str(config.bayes_model_path),
            "bayes_model_version": model_version,
            "generation_model": config.generation_model,
            "score_model": config.score_model,
            "seed": config.seed,
            "style_preset": config.style_preset,
            "prompt_template": PROMPT_TEMPLATE_VERSION,
            "workers": config.workers,
            "batch_size": config.batch_size,
            "target_records": config.target_records,
            "records_written": len(records),
            "ledger": str(config.ledger_path) if config.ledger_path else None,
        },
        manifest_path,
    )
    print(f"[STREAM] finalized DPO JSONL: {config.dpo_output} ({len(records)} 件)")
    print(f"[STREAM] manifest: {manifest_path}")


def write_heartbeat(
    *,
    config: StreamDpoConfig,
    state: StreamState,
    status: str,
) -> None:
    """watchdog用の進捗ファイルを書く。"""
    if config.heartbeat_file is None:
        return
    payload = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "seen_scored": state.seen_scored,
        "selected_appended": state.selected_appended,
        "attempted": state.attempted,
        "accepted": len(state.dpo_records),
        "target_records": config.target_records,
        "skipped": dict(state.skipped),
    }
    write_json(payload, config.heartbeat_file)


def stream_dpo_from_scored(
    config: StreamDpoConfig,
    *,
    generator: Any,
    once: bool = False,
) -> int:
    """スコア済みJSONLを監視し、条件を満たした候補からDPOを生成する。"""
    bayes_model = load_transition_bayes_model(config.bayes_model_path)
    instructions = build_translation_rejected_instructions(
        bayes_model,
        style_preset=config.style_preset,
    )
    model_version = bayes_model_version(config.bayes_model_path)
    state = load_stream_state(config)
    reader = JsonlTailReader(config.scored_input)

    if target_reached(config, state):
        finalize_dpo_output(config=config, state=state, model_version=model_version)
        write_heartbeat(config=config, state=state, status="target_already_reached")
        return EXIT_TARGET_REACHED

    while True:
        records = reader.read_available()
        state.seen_scored += len(records)
        candidates: list[dict[str, Any]] = []
        for record in records:
            selected = select_stream_candidate(record, config=config, state=state)
            if selected is None:
                continue
            candidates.append(selected)
            if len(candidates) >= config.batch_size:
                write_heartbeat(config=config, state=state, status="processing_batch")
                process_candidate_batch(
                    candidates,
                    config=config,
                    state=state,
                    bayes_model=bayes_model,
                    generator=generator,
                    instructions=instructions,
                    model_version=model_version,
                )
                candidates = []
                if target_reached(config, state):
                    finalize_dpo_output(
                        config=config,
                        state=state,
                        model_version=model_version,
                    )
                    write_heartbeat(config=config, state=state, status="target_reached")
                    return EXIT_TARGET_REACHED

        if candidates:
            write_heartbeat(config=config, state=state, status="processing_batch")
            process_candidate_batch(
                candidates,
                config=config,
                state=state,
                bayes_model=bayes_model,
                generator=generator,
                instructions=instructions,
                model_version=model_version,
            )
            if target_reached(config, state):
                finalize_dpo_output(config=config, state=state, model_version=model_version)
                write_heartbeat(config=config, state=state, status="target_reached")
                return EXIT_TARGET_REACHED

        if once:
            write_heartbeat(config=config, state=state, status="once_complete")
            return EXIT_TARGET_REACHED if target_reached(config, state) else 0

        if config.done_file is not None and config.done_file.exists():
            finalize_dpo_output(config=config, state=state, model_version=model_version)
            write_heartbeat(config=config, state=state, status="source_exhausted")
            print(
                "[STREAM] scoring finished before target "
                f"accepted={len(state.dpo_records)} target={config.target_records}",
                flush=True,
            )
            return EXIT_SOURCE_EXHAUSTED

        write_heartbeat(config=config, state=state, status="waiting_for_scored_rows")
        time.sleep(config.poll_seconds)


def run_stream_once(config: StreamDpoConfig, *, generator: Any) -> int:
    """テスト用に、現在読める入力だけを処理する。"""
    return stream_dpo_from_scored(config, generator=generator, once=True)


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    config = config_from_args(args)
    return stream_dpo_from_scored(
        config,
        generator=OpenAIResponsesGenerator(),
        once=args.once,
    )


if __name__ == "__main__":
    raise SystemExit(main())
