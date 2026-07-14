"""MathDial実験のdomain/topic/BASiS選別と診断を行う。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from core.transition_bayes_model import load_transition_bayes_model
from tools.extract_high_posterior_dialogues import (
    derive_selection_label_diagnostics,
    derive_selection_labels_from_model,
    select_high_posterior_records,
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
    vector: Counter[int] = Counter()
    for token in tokenize(text):
        index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % dimensions
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
    return {key: value / norm for key, value in vector.items()}


def cosine(left: dict[int, float], right: dict[int, float]) -> float:
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def build_topic_reference(mathdial_conversations: list[dict[str, Any]]) -> dict[int, float]:
    text = " ".join(str(row.get("metadata", {}).get("question", "")) for row in mathdial_conversations if row.get("split") == "train")
    return topic_vector(text)


def mmr_select(rows: list[dict[str, Any]], count: int, *, lambda_relevance: float = 0.8) -> list[dict[str, Any]]:
    """posterior選別候補へ決定論的MMRを適用する。"""
    if len(rows) <= count:
        return rows
    vectors = {str(row.get("sample_id")): tokenize(f"{row.get('prompt', '')} {row.get('response', '')}") for row in rows}
    selected: list[dict[str, Any]] = []
    remaining = list(rows)
    while remaining and len(selected) < count:
        best = None
        best_score = -float("inf")
        for row in remaining:
            tokens = vectors[str(row.get("sample_id"))]
            redundancy = 0.0
            for chosen in selected:
                other = vectors[str(chosen.get("sample_id"))]
                union = tokens | other
                redundancy = max(redundancy, len(tokens & other) / len(union) if union else 0.0)
            relevance = float(row.get("selection_score", row.get("posterior", 0.0)))
            score = lambda_relevance * relevance - (1.0 - lambda_relevance) * redundancy
            if score > best_score:
                best, best_score = row, score
        assert best is not None
        enriched = dict(best)
        enriched.setdefault("selection_metadata", {})["mmr_score"] = best_score
        enriched["selection_metadata"]["mmr_lambda"] = lambda_relevance
        selected.append(enriched)
        remaining.remove(best)
    return selected


def select_groups(
    scored: list[dict[str, Any]],
    mathdial: list[dict[str, Any]],
    *,
    count: int,
    random_count: int,
    seed: int,
    bayes_model_path: Path | str | None = None,
    label_derivation_method: str = "state_specific_margin",
    selection_margin: float = 0.05,
    exclude_fallback_conversations: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """既存posterior抽出を用いて3比較群を作る。"""
    deduped = {}
    for row in scored:
        key = (row["conversation_id"], row["turn_index"])
        if key not in deduped or float(row.get("posterior", 0.0)) > float(deduped[key].get("posterior", 0.0)):
            deduped[key] = row
    pool = sorted(
        deduped.values(),
        key=lambda row: (
            str(row.get("conversation_id", "")),
            int(row.get("turn_index", 0)),
            str(row.get("sample_id", "")),
        ),
    )
    fallback_conversations = {
        str(row.get("conversation_id", ""))
        for row in scored
        if row.get("llm_error")
    }
    rng = random.Random(seed)
    randomized = list(pool)
    rng.shuffle(randomized)
    domain_random = randomized[:random_count]
    reference = build_topic_reference(mathdial)
    for row in pool:
        row["topic_similarity_score"] = cosine(topic_vector(f"{row.get('prompt', '')} {row.get('response', '')}"), reference)
    topic = sorted(pool, key=lambda row: row["topic_similarity_score"], reverse=True)[:count]
    if bayes_model_path:
        labels = derive_selection_labels_from_model(
            load_transition_bayes_model(bayes_model_path),
            method=label_derivation_method,
            minimum_margin=selection_margin,
        )
    else:
        labels = {
            "prefer_states": "diagnosing,scaffolding,explaining,verifying",
            "prefer_observations": "open_diagnosis,probing_question,focusing_question,minimal_hint,scaffolded_hint,misconception_correction,comprehension_check",
            "low_priority_states": "explaining",
            "exclude_states": "premature_telling,generic_ungrounded",
            "exclude_observations": "",
        }
    basis_pool = pool
    if exclude_fallback_conversations:
        basis_pool = [
            row
            for row in pool
            if str(row.get("conversation_id", "")) not in fallback_conversations
        ]
    basis_candidates = select_high_posterior_records(
        basis_pool, min_posterior=0.0, max_records=None, target_records=None,
        sort_by_posterior=False, sort_by_selection=True, per_dialogue_limit=3,
        prefer_states=labels["prefer_states"],
        prefer_observations=labels["prefer_observations"],
        low_priority_states=labels["low_priority_states"],
        exclude_states=labels["exclude_states"],
        exclude_observations=labels["exclude_observations"], require_preferred=True,
    )
    basis = mmr_select(basis_candidates, count, lambda_relevance=0.8)
    return {"domain_random": domain_random, "topic_similarity_top": topic, "basis_top": basis}


def diagnostics(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result = {}
    for name, rows in groups.items():
        result[name] = {
            "records": len(rows),
            "conversations": len({row["conversation_id"] for row in rows}),
            "mean_posterior": sum(float(row.get("posterior", 0.0)) for row in rows) / len(rows) if rows else 0.0,
            "state_coverage": dict(Counter(str(row.get("most_likely_state", "")) for row in rows)),
            "strategy_coverage": dict(Counter(str(row.get("observation", "")) for row in rows)),
            "source_model_coverage": dict(Counter(str(row.get("metadata", {}).get("source_model", "unknown")) for row in rows)),
        }
    ids = {name: {row.get("sample_id") for row in rows} for name, rows in groups.items()}
    result["overlap"] = {f"{left}_vs_{right}": len(ids[left] & ids[right]) for left in ids for right in ids if left < right}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="MathDial WildChat 3群選別")
    parser.add_argument("--scored", required=True)
    parser.add_argument("--mathdial-conversations", required=True)
    parser.add_argument("--bayes-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--random-count", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-derivation-method",
        choices=("mean_difference", "state_specific_margin"),
        default="state_specific_margin",
    )
    parser.add_argument("--selection-margin", type=float, default=0.05)
    parser.add_argument(
        "--exclude-fallback-conversations",
        action="store_true",
        help="fallbackを含む会話全体をBASiS選別から除外します。",
    )
    args = parser.parse_args()
    scored = read_jsonl(args.scored)
    groups = select_groups(
        scored, read_jsonl(args.mathdial_conversations),
        count=args.count, random_count=args.random_count, seed=args.seed,
        bayes_model_path=args.bayes_model,
        label_derivation_method=args.label_derivation_method,
        selection_margin=args.selection_margin,
        exclude_fallback_conversations=args.exclude_fallback_conversations,
    )
    shortages = {
        name: (len(rows), args.random_count if name == "domain_random" else args.count)
        for name, rows in groups.items()
        if len(rows) < (args.random_count if name == "domain_random" else args.count)
    }
    if shortages:
        detail = ", ".join(f"{name}={actual}/{required}" for name, (actual, required) in shortages.items())
        raise RuntimeError(f"WildChat選別候補が不足しています: {detail}")
    output = Path(args.output_dir)
    for name, rows in groups.items():
        write_jsonl(rows, output / f"{name}.jsonl")
    report = diagnostics(groups)
    fallback_conversations = {
        str(row.get("conversation_id", ""))
        for row in scored
        if row.get("llm_error")
    }
    report["basis_quality_filter"] = {
        "exclude_fallback_conversations": args.exclude_fallback_conversations,
        "fallback_conversations": len(fallback_conversations),
        "selected_basis_from_fallback_conversations": sum(
            str(row.get("conversation_id", "")) in fallback_conversations
            for row in groups["basis_top"]
        ),
    }
    report["label_derivation"] = derive_selection_label_diagnostics(
        load_transition_bayes_model(args.bayes_model),
        method=args.label_derivation_method,
        minimum_margin=args.selection_margin,
    )
    (output / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
