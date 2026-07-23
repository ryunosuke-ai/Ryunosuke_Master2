"""MediTOD実験のdomain/topic/BASiS 3群選別。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from core.mathdial_basis import load_yaml
from core.transition_bayes_model import load_transition_bayes_model
from tools.extract_high_posterior_dialogues import (
    derive_selection_label_diagnostics,
    derive_selection_labels_from_model,
    select_high_posterior_records,
)
from tools.mathdial_selection import length_summary, mmr_select, source_text_characters
from tools.wildchat_health import (
    has_explicit_unsafe_medical_advice,
    health_conversation_diagnostic_category,
)
from tools.wildchat_tutoring import tokenize


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def topic_vector(text: str, dimensions: int = 2048) -> dict[int, float]:
    counts: Counter[int] = Counter()
    for token in tokenize(text):
        counts[int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % dimensions] += 1
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    return {key: value / norm for key, value in counts.items()}


def cosine(left: dict[int, float], right: dict[int, float]) -> float:
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def build_topic_reference(conversations: list[dict[str, Any]]) -> dict[int, float]:
    text = " ".join(
        turn["text"]
        for row in conversations
        if row.get("split") == "train" and not row.get("metadata", {}).get("ood")
        for turn in row["turns"]
    )
    return topic_vector(text)


def select_groups(
    scored: list[dict[str, Any]],
    conversations: list[dict[str, Any]],
    *,
    count: int,
    random_count: int,
    seed: int,
    bayes_model_path: Path | str,
    selection_margin: float,
    max_source_characters: int | None,
    allowed_record_keys: set[tuple[str, int]] | None = None,
    domain_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    deduped = {}
    for row in scored:
        key = (str(row["conversation_id"]), int(row["turn_index"]))
        if key not in deduped or float(row.get("posterior", 0)) > float(deduped[key].get("posterior", 0)):
            deduped[key] = row
    if allowed_record_keys is not None:
        deduped = {
            key: row for key, row in deduped.items() if key in allowed_record_keys
        }
    length_filtered = [
        row for row in deduped.values()
        if max_source_characters is None or source_text_characters(row) <= max_source_characters
    ]
    pool = [
        row for row in length_filtered
        if not has_explicit_unsafe_medical_advice(str(row.get("response", "")))
    ]
    fallback_conversations = {
        str(row["conversation_id"]) for row in scored if row.get("llm_error")
    }
    rng = random.Random(seed)
    domain_source = domain_candidates if domain_candidates is not None else pool
    randomized = sorted(
        [
            row
            for row in domain_source
            if (
                max_source_characters is None
                or source_text_characters(row) <= max_source_characters
            )
            and not has_explicit_unsafe_medical_advice(str(row.get("response", "")))
        ],
        key=lambda row: (str(row["conversation_id"]), int(row["turn_index"])),
    )
    rng.shuffle(randomized)
    reference = build_topic_reference(conversations)
    for row in pool:
        row["topic_similarity_score"] = cosine(
            topic_vector(f"{row.get('prompt', '')} {row.get('response', '')}"), reference
        )
    labels = derive_selection_labels_from_model(
        load_transition_bayes_model(bayes_model_path),
        method="state_specific_margin",
        minimum_margin=selection_margin,
    )
    clean_pool = [row for row in pool if str(row["conversation_id"]) not in fallback_conversations]
    basis_candidates = select_high_posterior_records(
        clean_pool,
        min_posterior=0.0,
        max_records=None,
        target_records=None,
        sort_by_posterior=False,
        sort_by_selection=True,
        per_dialogue_limit=3,
        prefer_states=labels["prefer_states"],
        prefer_observations=labels["prefer_observations"],
        low_priority_states=labels["low_priority_states"],
        exclude_states=labels["exclude_states"],
        exclude_observations=labels["exclude_observations"],
        require_preferred=True,
    )
    return {
        "domain_random": randomized[:random_count],
        "topic_similarity_top": sorted(pool, key=lambda row: row["topic_similarity_score"], reverse=True)[:count],
        "basis_top": mmr_select(basis_candidates, count, lambda_relevance=0.8),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MediTOD WildChat 3群選別")
    parser.add_argument("--scored", required=True)
    parser.add_argument("--meditod-conversations", required=True)
    parser.add_argument("--wildchat-conversations")
    parser.add_argument(
        "--health-config",
        default="configs/datasets/wildchat_health.yaml",
    )
    parser.add_argument("--bayes-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--random-count", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-margin", type=float, default=0.05)
    parser.add_argument("--max-source-characters", type=int)
    parser.add_argument("--allowed-records")
    parser.add_argument(
        "--domain-candidates",
        help="domain_randomをscoring済み集合ではなく、この個人健康相談候補から作ります。",
    )
    args = parser.parse_args()
    scored = read_jsonl(args.scored)
    allowed_rows = read_jsonl(args.allowed_records) if args.allowed_records else None
    allowed_record_keys = (
        {
            (str(row["conversation_id"]), int(row["turn_index"]))
            for row in allowed_rows
        }
        if allowed_rows is not None
        else None
    )
    groups = select_groups(
        scored,
        read_jsonl(args.meditod_conversations),
        count=args.count,
        random_count=args.random_count,
        seed=args.seed,
        bayes_model_path=args.bayes_model,
        selection_margin=args.selection_margin,
        max_source_characters=args.max_source_characters,
        allowed_record_keys=allowed_record_keys,
        domain_candidates=(
            read_jsonl(args.domain_candidates) if args.domain_candidates else None
        ),
    )
    shortages = {
        name: (len(rows), args.random_count if name == "domain_random" else args.count)
        for name, rows in groups.items()
        if len(rows) < (args.random_count if name == "domain_random" else args.count)
    }
    if shortages:
        raise RuntimeError(f"MediTOD WildChat選別候補が不足しています: {shortages}")
    output = Path(args.output_dir)
    for name, rows in groups.items():
        write_jsonl(rows, output / f"{name}.jsonl")
    model = load_transition_bayes_model(args.bayes_model)
    report = {
        name: {
            "records": len(rows),
            "conversations": len({row["conversation_id"] for row in rows}),
            "mean_posterior": sum(float(row.get("posterior", 0)) for row in rows) / len(rows),
            "state_coverage": dict(Counter(str(row.get("most_likely_state", "")) for row in rows)),
            "strategy_coverage": dict(Counter(str(row.get("observation", "")) for row in rows)),
            "source_length_characters": length_summary(rows),
        }
        for name, rows in groups.items()
    }
    report["label_derivation"] = derive_selection_label_diagnostics(
        model, method="state_specific_margin", minimum_margin=args.selection_margin
    )
    report["selection_policy"] = {
        "fallback_conversations_excluded_from_basis": True,
        "per_dialogue_limit": 3,
        "mmr_lambda": 0.8,
        "topic_reference": "MediTOD train complete dialogue text",
        "common_explicit_unsafe_medical_advice_filter": True,
        "explicit_unsafe_source_responses_excluded": sum(
            has_explicit_unsafe_medical_advice(str(row.get("response", "")))
            for row in scored
        ),
    }
    if args.wildchat_conversations:
        health_config = load_yaml(args.health_config)
        categories = {
            str(row["conversation_id"]): health_conversation_diagnostic_category(
                row,
                health_config,
            )
            for row in read_jsonl(args.wildchat_conversations)
        }
        report["diagnostic_category_coverage"] = {
            name: dict(
                Counter(
                    categories.get(
                        str(row["conversation_id"]),
                        "unknown",
                    )
                    for row in rows
                )
            )
            for name, rows in groups.items()
        }
        report["diagnostic_category_policy"] = (
            "diagnostic_only; not used for selection eligibility"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
