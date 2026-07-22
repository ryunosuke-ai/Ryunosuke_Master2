"""WildChat-1Mから健康相談の高再現率マルチターン候補をstreaming抽出する。"""

from __future__ import annotations

import argparse
import datetime
import json
import re
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from core.dialogue_schema import build_assistant_samples
from core.mathdial_basis import load_yaml
from tools.wildchat_tutoring import (
    NearDuplicateIndex,
    normalize_wildchat_row,
    sample_to_scoring_record,
    tokenize,
    write_jsonl,
)


KNOWLEDGE_PREFIXES = (
    "what is ",
    "what are ",
    "explain ",
    "define ",
    "tell me about ",
    "how does ",
)
MEDICATION_PATTERN = re.compile(
    r"\b(?:aspirin|ibuprofen|acetaminophen|paracetamol|amoxicillin|azithromycin|"
    r"metformin|insulin|warfarin|lisinopril|prednisone|albuterol|omeprazole|"
    r"[a-z]{4,}(?:cillin|mycin|cycline|pril|sartan|olol|statin|prazole|formin))\b",
    re.IGNORECASE,
)
UNSAFE_MEDICAL_ADVICE_PATTERNS = (
    re.compile(r"\b(?:stop|discontinue|double|triple)\s+(?:taking\s+)?(?:your\s+)?(?:medication|medicine|dose|insulin|warfarin)\b", re.I),
    re.compile(r"\btake\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml)\b", re.I),
    re.compile(r"\b(?:definitely|certainly)\s+(?:have|is)\b", re.I),
    re.compile(r"\b(?:no need|do not need|don't need)\s+to\s+(?:see|contact|visit)\s+(?:a\s+)?(?:doctor|clinic|hospital)\b", re.I),
    re.compile(r"(?:薬|服用|投薬).{0,12}(?:中止|倍量|2倍|３倍|3倍)", re.I),
    re.compile(r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml).{0,10}(?:飲んで|服用して|投与して)", re.I),
    re.compile(r"(?:間違いなく|確実に).{0,8}(?:です|病気|疾患)", re.I),
    re.compile(r"(?:受診|医師|病院).{0,10}(?:必要ありません|不要です|行かなくて)", re.I),
)


def _contains_any(text: str, values: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(str(value).lower() in lowered for value in values)


def health_domain_flags(record: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    user_turns = [turn["text"] for turn in record["turns"] if turn["role"] == "user"]
    all_text = " ".join(turn["text"] for turn in record["turns"])
    user_text = " ".join(user_turns)
    health = _contains_any(user_text, config["domain_keywords"])
    respiratory = health and _contains_any(user_text, config["respiratory_keywords"])
    personal = _contains_any(user_text, config.get("personal_consultation_markers", []))
    followups = user_turns[1:]
    followup_information = any(
        len(tokenize(text)) >= 3
        and (
            _contains_any(text, config.get("patient_information_markers", []))
            or _contains_any(text, config["domain_keywords"])
            or bool(re.search(r"\b(?:i|my|me|mine)\b", text, re.IGNORECASE))
        )
        for text in followups
    )
    first = user_turns[0].strip().lower() if user_turns else ""
    single_knowledge = any(first.startswith(prefix) for prefix in KNOWLEDGE_PREFIXES) and not personal
    explicit_pii = any(re.search(pattern, all_text, flags=re.IGNORECASE) for pattern in config.get("pii_patterns", []))
    toxic_text = _contains_any(all_text, config.get("toxic_keywords", []))
    return {
        "health": health,
        "respiratory": respiratory,
        "personal": personal,
        "followup_information": followup_information,
        "single_knowledge": single_knowledge,
        "explicit_pii": explicit_pii,
        "toxic_text": toxic_text,
    }


def protected_medical_terms(sample: dict[str, Any]) -> list[str]:
    """翻訳時に原語保持を要求する薬剤名だけを抽出する。"""
    text = " ".join(
        [str(turn.get("text", "")) for turn in sample.get("history", [])]
        + [str(sample.get("response", ""))]
    )
    return list(dict.fromkeys(match.group(0) for match in MEDICATION_PATTERN.finditer(text)))


def has_explicit_unsafe_medical_advice(text: str) -> bool:
    """明白な危険投薬・受診抑制・根拠のない断定だけを保守的に検知する。"""
    return any(pattern.search(text) for pattern in UNSAFE_MEDICAL_ADVICE_PATTERNS)


def sample_with_medical_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    """共通scoring recordへ翻訳保持対象の薬剤名だけを追加する。"""
    record = sample_to_scoring_record(sample)
    record["metadata"] = {
        **record["metadata"],
        "protected_medical_terms": protected_medical_terms(sample),
    }
    return record


def extract_candidates(
    rows: Iterable[dict[str, Any]],
    config: dict[str, Any],
    limit: int | None = None,
    *,
    target_candidate_records: int | None = None,
    progress_every: int = 10_000,
    checkpoint_every: int = 100_000,
    initial_general: list[dict[str, Any]] | None = None,
    initial_respiratory: list[dict[str, Any]] | None = None,
    initial_counts: dict[str, Any] | None = None,
    on_checkpoint: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """health domainとマルチターン性だけで粗候補を抽出する。"""
    general = list(initial_general or [])
    respiratory = list(initial_respiratory or [])
    counts: Counter[str] = Counter(initial_counts or {})
    counts["stopped_by_candidate_target"] = 0
    counts["stopped_by_row_limit"] = 0
    exact_seen = {row["metadata"]["conversation_hash"] for row in general}
    near = NearDuplicateIndex(float(config.get("near_duplicate_jaccard", 0.9)))
    for existing in general:
        near.is_duplicate(tokenize(" ".join(turn["text"] for turn in existing["turns"])))
    for row in rows:
        if on_checkpoint and checkpoint_every > 0 and counts["stream_rows"] and counts["stream_rows"] % checkpoint_every == 0:
            on_checkpoint(general, respiratory, dict(counts), False)
        if limit is not None and counts["stream_rows"] >= limit:
            counts["stopped_by_row_limit"] = 1
            break
        counts["stream_rows"] += 1
        if progress_every > 0 and counts["stream_rows"] % progress_every == 0:
            print(
                "[extract_wildchat_health] "
                f"stream_rows={counts['stream_rows']} "
                f"general_conversations={len(general)} "
                f"candidate_records={counts['general_candidate_records']}",
                flush=True,
            )
        record, reason = normalize_wildchat_row(row, config)
        if record is None:
            counts[f"excluded_{reason}"] += 1
            continue
        user_turns = record["num_user_turns"]
        for threshold in range(2, 6):
            counts[f"at_least_{threshold}_user_turns"] += int(user_turns >= threshold)
        if user_turns < int(config.get("minimum_user_turns", 4)):
            counts["excluded_too_few_user_turns"] += 1
            continue
        flags = health_domain_flags(record, config)
        counts["health_domain"] += int(flags["health"])
        counts["personal_symptom_consultation"] += int(flags["personal"])
        counts["followup_information"] += int(flags["followup_information"])
        if not flags["health"]:
            counts["excluded_non_health"] += 1
            continue
        if flags["single_knowledge"] and config.get("exclude_single_turn_knowledge_questions", True):
            counts["excluded_medical_knowledge_only"] += 1
            continue
        if config.get("require_followup_information", True) and not flags["followup_information"]:
            counts["excluded_no_followup_information"] += 1
            continue
        if flags["explicit_pii"]:
            counts["excluded_explicit_pii"] += 1
            continue
        if flags["toxic_text"]:
            counts["excluded_toxic_text"] += 1
            continue
        digest = record["metadata"]["conversation_hash"]
        if digest in exact_seen:
            counts["excluded_exact_duplicate"] += 1
            continue
        token_set = tokenize(" ".join(turn["text"] for turn in record["turns"]))
        if near.is_duplicate(token_set):
            counts["excluded_near_duplicate"] += 1
            continue
        exact_seen.add(digest)
        record["metadata"].update(
            {
                "domain": "general_health_consultation",
                "has_followup_patient_information": True,
                "personal_symptom_consultation": flags["personal"],
                "pii_metadata_retained": False,
            }
        )
        general.append(record)
        eligible = [
            sample for sample in build_assistant_samples(record)
            if sample["metadata"]["dpo_eligible"] and sample.get("next_user_turn") is not None
        ]
        counts["general_candidate_records"] += len(eligible)
        if flags["respiratory"]:
            copied = json.loads(json.dumps(record))
            copied["metadata"]["domain"] = "respiratory_health"
            respiratory.append(copied)
            counts["respiratory_candidate_records"] += len(eligible)
        if target_candidate_records is not None and counts["general_candidate_records"] >= target_candidate_records:
            counts["stopped_by_candidate_target"] = 1
            break
    counts["general_conversations"] = len(general)
    counts["respiratory_conversations"] = len(respiratory)
    counts["target_candidate_records"] = target_candidate_records or 0
    counts["stream_exhausted"] = int(not counts["stopped_by_candidate_target"] and not counts["stopped_by_row_limit"])
    if on_checkpoint:
        on_checkpoint(general, respiratory, dict(counts), True)
    return general, respiratory, dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description="WildChat health候補をstreaming抽出")
    parser.add_argument("--config", default="configs/datasets/wildchat_health.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fixture")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target-candidate-records", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--heartbeat-file")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    output = Path(args.output_dir)
    general_path = output / "general_health_consultation_conversations.jsonl"
    respiratory_path = output / "respiratory_health_conversations.jsonl"
    checkpoint_path = output / "stream_checkpoint.json"
    initial_general: list[dict[str, Any]] = []
    initial_respiratory: list[dict[str, Any]] = []
    initial_counts: dict[str, Any] = {}
    processed = 0
    if not args.no_resume and checkpoint_path.exists() and general_path.exists() and respiratory_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if int(checkpoint.get("seed", -1)) != args.seed:
            raise ValueError("WildChat health checkpointのseedが一致しません。")
        outputs = (
            output / "general_health_consultation_candidates.jsonl",
            output / "respiratory_health_candidates.jsonl",
            output / "statistics.json",
            output / "manifest.json",
        )
        if checkpoint.get("completed") is True and all(path.is_file() for path in outputs):
            print("[extract_wildchat_health] completed checkpointを再利用", flush=True)
            return 0
        initial_general = [json.loads(line) for line in general_path.open(encoding="utf-8") if line.strip()]
        initial_respiratory = [json.loads(line) for line in respiratory_path.open(encoding="utf-8") if line.strip()]
        initial_counts = dict(checkpoint.get("statistics", {}))
        processed = int(initial_counts.get("stream_rows", 0))
    if args.fixture:
        source = (json.loads(line) for line in Path(args.fixture).open(encoding="utf-8") if line.strip())
        rows = islice(source, processed, None)
    else:
        from datasets import load_dataset

        rows = load_dataset(
            config["dataset_name"], split=config["split"], revision=config["revision"], streaming=True
        )
        buffer_size = int(config.get("stream_shuffle_buffer_size", 0))
        if buffer_size:
            rows = rows.shuffle(seed=args.seed, buffer_size=buffer_size)
        if processed:
            rows = rows.skip(processed)

    def checkpoint_callback(general: list[dict[str, Any]], respiratory: list[dict[str, Any]], counts: dict[str, Any], completed: bool) -> None:
        write_jsonl(general, general_path)
        write_jsonl(respiratory, respiratory_path)
        payload = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seed": args.seed,
            "completed": completed,
            "statistics": counts,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(checkpoint_path)
        if args.heartbeat_file:
            Path(args.heartbeat_file).write_text(json.dumps({"timestamp": payload["timestamp"], "stage": "extract_wildchat", "stream_rows": counts.get("stream_rows", 0)}, ensure_ascii=False) + "\n", encoding="utf-8")

    general, respiratory, stats = extract_candidates(
        rows,
        config,
        args.limit,
        target_candidate_records=args.target_candidate_records,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
        initial_general=initial_general,
        initial_respiratory=initial_respiratory,
        initial_counts=initial_counts,
        on_checkpoint=checkpoint_callback,
    )
    write_jsonl(general, general_path)
    write_jsonl(respiratory, respiratory_path)
    general_samples = [
        sample_with_medical_metadata(sample)
        for record in general
        for sample in build_assistant_samples(record)
        if sample["metadata"]["dpo_eligible"] and sample.get("next_user_turn") is not None
    ]
    respiratory_ids = {row["conversation_id"] for row in respiratory}
    write_jsonl(general_samples, output / "general_health_consultation_candidates.jsonl")
    write_jsonl([row for row in general_samples if row["conversation_id"] in respiratory_ids], output / "respiratory_health_candidates.jsonl")
    (output / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "dataset": config["dataset_name"],
        "revision": config["revision"],
        "config": config,
        "stream_shuffle_seed": args.seed,
        "target_candidate_records": args.target_candidate_records,
        "statistics": stats,
        "pii_policy": "retain conversation_hash and source_model only; discard source metadata",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
