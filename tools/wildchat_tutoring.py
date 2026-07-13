"""WildChat-1Mから教育・個別指導候補をstreaming抽出する。"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from core.dialogue_schema import build_assistant_samples, validate_conversation
from core.mathdial_basis import canonical_hash, load_yaml


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


def simhash64(tokens: Iterable[str]) -> int:
    values = [0] * 64
    for token in sorted(set(tokens)):
        number = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
        for index in range(64):
            values[index] += 1 if number & (1 << index) else -1
    return sum((1 << index) for index, value in enumerate(values) if value >= 0)


class NearDuplicateIndex:
    """候補だけを保持する小さなSimHash band index。"""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.bands: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.token_sets: list[set[str]] = []

    def is_duplicate(self, tokens: set[str]) -> bool:
        fingerprint = simhash64(tokens)
        candidates: set[int] = set()
        for band in range(4):
            candidates.update(self.bands.get((band, (fingerprint >> (band * 16)) & 0xFFFF), []))
        for index in candidates:
            existing = self.token_sets[index]
            union = tokens | existing
            if union and len(tokens & existing) / len(union) >= self.threshold:
                return True
        index = len(self.token_sets)
        self.token_sets.append(tokens)
        for band in range(4):
            self.bands[(band, (fingerprint >> (band * 16)) & 0xFFFF)].append(index)
        return False


def normalize_wildchat_row(row: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """WildChat rowをPIIを含まない共通会話形式へ変換する。"""
    if str(row.get("language", "")).lower() != "english":
        return None, "non_english"
    if config.get("exclude_toxic", True) and bool(row.get("toxic")):
        return None, "toxic"
    if config.get("exclude_redacted", True) and bool(row.get("redacted")):
        return None, "redacted"
    raw_turns = row.get("conversation")
    if not isinstance(raw_turns, list) or not raw_turns:
        return None, "broken_conversation"
    turns = []
    expected = "user"
    for raw in raw_turns:
        if not isinstance(raw, dict):
            return None, "broken_turn"
        role = str(raw.get("role", "")).strip().lower()
        text = str(raw.get("content", "")).strip()
        turn_language = str(raw.get("language", "")).strip().lower()
        if turn_language and turn_language != "english":
            return None, "non_english_turn"
        if role != expected or not text:
            return None, "invalid_roles_or_empty"
        if bool(raw.get("toxic")) or (config.get("exclude_redacted", True) and bool(raw.get("redacted"))):
            return None, "unsafe_turn"
        turns.append({"role": role, "text": text})
        expected = "assistant" if role == "user" else "user"
    if turns[-1]["role"] != "assistant":
        return None, "incomplete_exchange"
    exchanges = sum(turn["role"] == "assistant" for turn in turns)
    if exchanges < int(config.get("minimum_exchanges", 3)):
        return None, "too_short"
    content_hash = canonical_hash(turns)
    record = {
        "conversation_id": f"wildchat_{content_hash[:20]}",
        "source_dataset": "wildchat",
        "split": "candidate",
        "turns": turns,
        "num_messages": len(turns),
        "num_user_turns": sum(turn["role"] == "user" for turn in turns),
        "num_assistant_turns": exchanges,
        "language": "English",
        "metadata": {
            "conversation_hash": content_hash,
            "source_model": str(row.get("model", "unknown")),
            "eligible_for_training": True,
        },
    }
    return validate_conversation(record), None


def domain_flags(record: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    text = " ".join(turn["text"] for turn in record["turns"]).lower()
    general = any(term.lower() in text for term in config["general_tutoring_terms"])
    math = general and any(term.lower() in text for term in config["math_terms"])
    confusion = any(term.lower() in text for term in config["confusion_terms"])
    followup = any(
        record["turns"][index]["role"] == "assistant" and record["turns"][index + 1]["role"] == "user"
        for index in range(len(record["turns"]) - 1)
    )
    return {"general": general, "math": math, "confusion": confusion, "followup": followup}


def sample_to_scoring_record(sample: dict[str, Any]) -> dict[str, Any]:
    """共通assistant sampleを既存ESConv scorer入力へ変換する。"""
    prompt = "\n".join(f"{'User' if turn['role'] == 'user' else 'AI'}: {turn['text']}" for turn in sample["history"])
    return {
        "sample_id": sample["sample_id"],
        "conversation_id": sample["conversation_id"],
        "turn_index": int(sample["metadata"]["assistant_turn_index"]),
        "prompt": prompt,
        "response": sample["response"],
        "history": sample["history"],
        "next_user_turn": sample.get("next_user_turn"),
        "metadata": {**sample["metadata"], "context_turns": len(sample["history"])},
    }


def extract_candidates(
    rows: Iterable[dict[str, Any]],
    config: dict[str, Any],
    limit: int | None = None,
    *,
    target_candidate_records: int | None = None,
    progress_every: int = 10_000,
    checkpoint_every: int = 100_000,
    initial_general: list[dict[str, Any]] | None = None,
    initial_math: list[dict[str, Any]] | None = None,
    initial_counts: dict[str, Any] | None = None,
    on_checkpoint: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """stream row列からgeneral/math候補と走査統計を返す。"""
    general, math = list(initial_general or []), list(initial_math or [])
    counts: Counter[str] = Counter(initial_counts or {})
    exact_seen: set[str] = {
        str(row.get("metadata", {}).get("conversation_hash", "")) for row in general
    }
    near = NearDuplicateIndex(float(config.get("near_duplicate_jaccard", 0.9)))
    for existing in general:
        near.is_duplicate(tokenize(" ".join(turn["text"] for turn in existing["turns"])))
    if target_candidate_records is not None and counts["general_candidate_records"] >= target_candidate_records:
        counts["stopped_by_candidate_target"] = 1
        return general, math, dict(counts)
    for row in rows:
        if (
            on_checkpoint
            and checkpoint_every > 0
            and counts["stream_rows"] > 0
            and counts["stream_rows"] % checkpoint_every == 0
        ):
            on_checkpoint(general, math, dict(counts), False)
        if limit is not None and counts["stream_rows"] >= limit:
            counts["stopped_by_row_limit"] = 1
            break
        counts["stream_rows"] += 1
        if progress_every > 0 and counts["stream_rows"] % progress_every == 0:
            print(
                "[extract_wildchat] "
                f"stream_rows={counts['stream_rows']} "
                f"general_conversations={len(general)} "
                f"candidate_records={counts['general_candidate_records']}",
                flush=True,
            )
        record, reason = normalize_wildchat_row(row, config)
        if record is None:
            counts[f"excluded_{reason}"] += 1
            continue
        exchanges = record["num_assistant_turns"]
        for threshold in range(2, 6):
            counts[f"at_least_{threshold}_exchanges"] += int(exchanges >= threshold)
        flags = domain_flags(record, config)
        counts["confusion"] += int(flags["confusion"])
        counts["followup"] += int(flags["followup"])
        if not flags["general"]:
            counts["excluded_non_tutoring"] += 1
            continue
        digest = record["metadata"]["conversation_hash"]
        if digest in exact_seen:
            counts["excluded_exact_duplicate"] += 1
            continue
        tokens = tokenize(" ".join(turn["text"] for turn in record["turns"]))
        if near.is_duplicate(tokens):
            counts["excluded_near_duplicate"] += 1
            continue
        exact_seen.add(digest)
        record["metadata"].update({"domain": "general_tutoring", "contains_confusion_signal": flags["confusion"], "has_followup_user": flags["followup"]})
        general.append(record)
        eligible_samples = [
            sample
            for sample in build_assistant_samples(record)
            if sample["metadata"]["dpo_eligible"]
        ]
        counts["general_candidate_records"] += len(eligible_samples)
        if flags["math"]:
            math_record = json.loads(json.dumps(record))
            math_record["metadata"]["domain"] = "math_tutoring"
            math.append(math_record)
            counts["math_candidate_records"] += len(eligible_samples)
        if (
            target_candidate_records is not None
            and counts["general_candidate_records"] >= target_candidate_records
        ):
            counts["stopped_by_candidate_target"] = 1
            break
    counts["general_candidates"] = len(general)
    counts["math_candidates"] = len(math)
    counts["available_assistant_responses"] = sum(len(build_assistant_samples(row)) for row in general)
    counts["target_candidate_records"] = target_candidate_records or 0
    counts.setdefault("stopped_by_candidate_target", 0)
    counts.setdefault("stopped_by_row_limit", 0)
    if on_checkpoint:
        on_checkpoint(general, math, dict(counts), True)
    return general, math, dict(counts)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """中断時に既存成果物を壊さないようJSONLを原子的に置換する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="WildChat tutoring候補をstreaming抽出")
    parser.add_argument("--config", default="configs/datasets/wildchat_tutoring.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fixture", help="テスト用WildChat row JSONL")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--target-candidate-records",
        type=int,
        help="学習可能なassistant候補がこの件数に達したら、会話境界でstreaming走査を終了します。",
    )
    parser.add_argument("--seed", type=int, default=42, help="streaming shuffleの乱数seed。")
    parser.add_argument("--progress-every", type=int, default=10_000, help="走査進捗を表示するraw row間隔。")
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--heartbeat-file")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    output = Path(args.output_dir)
    general_path = output / "general_tutoring_conversations.jsonl"
    math_path = output / "math_tutoring_conversations.jsonl"
    checkpoint_path = output / "stream_checkpoint.json"
    initial_general: list[dict[str, Any]] = []
    initial_math: list[dict[str, Any]] = []
    initial_counts: dict[str, Any] = {}
    processed = 0
    if not args.no_resume and checkpoint_path.exists() and general_path.exists() and math_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if int(checkpoint.get("seed", -1)) != args.seed:
            raise ValueError("WildChat checkpointのseedが現在値と一致しません。")
        initial_general = [json.loads(line) for line in general_path.open(encoding="utf-8") if line.strip()]
        initial_math = [json.loads(line) for line in math_path.open(encoding="utf-8") if line.strip()]
        initial_counts = dict(checkpoint.get("statistics", {}))
        processed = int(initial_counts.get("stream_rows", 0))
        print(f"[extract_wildchat] resume stream_rows={processed} candidates={initial_counts.get('general_candidate_records', 0)}", flush=True)
    if args.fixture:
        source_rows = (json.loads(line) for line in Path(args.fixture).open(encoding="utf-8") if line.strip())
        rows = islice(source_rows, processed, None)
    else:
        from datasets import load_dataset
        rows = load_dataset(config["dataset_name"], split=config["split"], revision=config["revision"], streaming=True)
        shuffle_buffer = int(config.get("stream_shuffle_buffer_size", 0))
        if shuffle_buffer > 0:
            rows = rows.shuffle(seed=args.seed, buffer_size=shuffle_buffer)
        if processed:
            rows = rows.skip(processed)

    def checkpoint_callback(
        general_rows: list[dict[str, Any]],
        math_rows: list[dict[str, Any]],
        counts: dict[str, Any],
        completed: bool,
    ) -> None:
        write_jsonl(general_rows, general_path)
        write_jsonl(math_rows, math_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".json.tmp")
        payload = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seed": args.seed,
            "completed": completed,
            "statistics": counts,
        }
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(checkpoint_path)
        if args.heartbeat_file:
            heartbeat = Path(args.heartbeat_file)
            heartbeat.parent.mkdir(parents=True, exist_ok=True)
            heartbeat.write_text(json.dumps({"timestamp": payload["timestamp"], "state": "running", "stage": "extract_wildchat", "stream_rows": counts.get("stream_rows", 0), "candidate_records": counts.get("general_candidate_records", 0)}, ensure_ascii=False) + "\n", encoding="utf-8")
    general, math, stats = extract_candidates(
        rows,
        config,
        args.limit,
        target_candidate_records=args.target_candidate_records,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
        initial_general=initial_general,
        initial_math=initial_math,
        initial_counts=initial_counts,
        on_checkpoint=checkpoint_callback,
    )
    write_jsonl(general, general_path)
    write_jsonl(math, math_path)
    general_samples = [sample_to_scoring_record(sample) for record in general for sample in build_assistant_samples(record) if sample["metadata"]["dpo_eligible"]]
    math_ids = {row["conversation_id"] for row in math}
    write_jsonl(general_samples, output / "general_tutoring_candidates.jsonl")
    write_jsonl([row for row in general_samples if row["conversation_id"] in math_ids], output / "math_tutoring_candidates.jsonl")
    (output / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "dataset": config["dataset_name"],
        "revision": config["revision"],
        "config": config,
        "stream_shuffle_seed": args.seed,
        "target_candidate_records": args.target_candidate_records,
        "statistics": stats,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
