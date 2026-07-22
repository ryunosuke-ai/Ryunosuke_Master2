"""3モデル応答とOracle結果からblindなA/B人手評価itemを作る。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from core.three_model_likert_survey import FINAL_CHOICES, load_definition, read_jsonl


MODEL_RESPONSE_KEYS = {
    "base": "base_response",
    "basis": "basis_response",
    "random_dpo": "random_dpo_response",
}
MODEL_ALIASES = {
    "base": "base",
    "basis": "basis",
    "bayes_dpo": "basis",
    "random_dpo": "random_dpo",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_history(row: dict[str, Any]) -> str:
    history = row.get("history_ja") or row.get("history") or []
    parts = []
    prompt = str(row.get("problem_ja") or row.get("prompt_ja") or row.get("prompt") or "").strip()
    if prompt and not history:
        parts.append(f"患者/学習者: {prompt}")
    for turn in history:
        role = str(turn.get("role") or turn.get("speaker") or "").lower()
        label = "AI" if role in {"assistant", "doctor", "ai"} else "User"
        parts.append(f"{label}: {str(turn.get('text') or turn.get('content') or '').strip()}")
    return "\n\n".join(part for part in parts if part.split(":", 1)[-1].strip())


def load_oracle_scores(path: Path) -> tuple[dict[str, dict[str, dict[str, float]]], tuple[str, ...]]:
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    axes: tuple[str, ...] = ()
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or row.get("prompt_id") or "")
        model = MODEL_ALIASES.get(str(row.get("model_name") or ""))
        scores = row.get("scores")
        if not sample_id or model is None or not isinstance(scores, dict) or not scores:
            continue
        current_axes = tuple(str(key) for key in scores)
        if axes and current_axes != axes:
            raise ValueError("Oracle raw内で評価軸が一致しません。")
        axes = current_axes
        if model in result[sample_id]:
            raise ValueError(f"Oracle scoreが重複しています: {sample_id}/{model}")
        result[sample_id][model] = {key: float(value) for key, value in scores.items()}
    return dict(result), axes


def readable_and_distinct(responses: dict[str, str]) -> tuple[bool, str]:
    normalized = {model: " ".join(value.split()) for model, value in responses.items()}
    if any(len(value) < 4 or len(value) > 1500 for value in normalized.values()):
        return False, "response_length"
    if len({sha256_text(value) for value in normalized.values()}) < 3:
        return False, "identical_response"
    if any("<script" in value.lower() or "```" in value for value in normalized.values()):
        return False, "display_risk"
    return True, "passed"


def candidate_rows(responses_path: Path, oracle_path: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    scores, axes = load_oracle_scores(oracle_path)
    output = []
    for row in read_jsonl(responses_path):
        sample_id = str(row.get("sample_id") or row.get("prompt_id") or "")
        if sample_id not in scores or set(scores[sample_id]) != set(MODEL_RESPONSE_KEYS):
            continue
        responses = {model: str(row.get(key) or "").strip() for model, key in MODEL_RESPONSE_KEYS.items()}
        passed, reason = readable_and_distinct(responses)
        means = {model: mean(scores[sample_id][model].values()) for model in MODEL_RESPONSE_KEYS}
        metadata = row.get("metadata") or {}
        stratum = str(
            row.get("selection_stratum")
            or metadata.get("selection_stratum")
            or (
                f"{row.get('selection_teacher_move')}:{row.get('selection_stage')}"
                if row.get("selection_teacher_move") or row.get("selection_stage")
                else "unspecified"
            )
        )
        output.append(
            {
                "sample_id": sample_id,
                "conversation": format_history(row),
                "responses": responses,
                "oracle_axis_scores": scores[sample_id],
                "oracle_means": means,
                "basis_advantage": means["basis"] - max(means["base"], means["random_dpo"]),
                "readability_passed": passed,
                "readability_reason": reason,
                "stratum": stratum,
                "conversation_id": str(row.get("conversation_id") or metadata.get("conversation_id") or ""),
            }
        )
    return output, axes


def select_outcome_enriched(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    eligible = [row for row in candidates if row["readability_passed"] and row["conversation"]]
    ranked = sorted(eligible, key=lambda row: (-row["basis_advantage"], -row["oracle_means"]["basis"], row["sample_id"]))
    if len(ranked) < count:
        raise ValueError(f"人手評価候補が不足しています: {len(ranked)}/{count}")
    selected = []
    per_stratum: Counter[str] = Counter()
    per_conversation: Counter[str] = Counter()
    remaining = ranked.copy()
    # Oracle差で富化しつつ、評価対象スタイルを単一段階へ偏らせない。
    strata = sorted({row["stratum"] for row in remaining})
    if len(strata) <= count:
        for stratum in strata:
            row = next(candidate for candidate in remaining if candidate["stratum"] == stratum)
            remaining.remove(row)
            selected.append(row)
            per_stratum[stratum] += 1
            per_conversation[row["conversation_id"]] += 1
    while remaining and len(selected) < count:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                remaining[index]["basis_advantage"] - 0.25 * per_stratum[remaining[index]["stratum"]] - 0.5 * per_conversation[remaining[index]["conversation_id"]],
                remaining[index]["oracle_means"]["basis"],
                remaining[index]["sample_id"],
            ),
        )
        row = remaining.pop(best_index)
        selected.append(row)
        per_stratum[row["stratum"]] += 1
        per_conversation[row["conversation_id"]] += 1
    return selected


def position_orders(count: int, seed: int) -> list[tuple[str, str, str]]:
    permutations = [
        ("base", "basis", "random_dpo"), ("basis", "random_dpo", "base"),
        ("random_dpo", "base", "basis"), ("base", "random_dpo", "basis"),
        ("basis", "base", "random_dpo"), ("random_dpo", "basis", "base"),
    ]
    orders = [permutations[index % len(permutations)] for index in range(count)]
    random.Random(seed).shuffle(orders)
    return orders


def build_records(selected: list[dict[str, Any]], definition: dict[str, Any], seed: int) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    orders = position_orders(len(selected), seed)
    public = {"A": [], "B": []}
    private = []
    half = len(selected) // 2
    for index, (row, order) in enumerate(zip(selected, orders), start=1):
        experiment = "A" if index <= half else "B"
        number = index if experiment == "A" else index - half
        item_id = f"{definition['dataset']}_{experiment.lower()}_{number:02d}"
        item = {
            "item_id": item_id,
            "item_number": number,
            "conversation": row["conversation"],
            "response_a": row["responses"][order[0]],
            "response_b": row["responses"][order[1]],
            "response_c": row["responses"][order[2]],
            "likert_statements": definition["axes"],
            "likert_columns": [str(value) for value in range(1, 8)],
            "likert_anchors": {"1": "全く当てはまらない", "4": "どちらともいえない", "7": "非常によく当てはまる"},
            "final_choice_question": definition["final_choice_question"],
            "final_choice_options": list(FINAL_CHOICES),
        }
        public[experiment].append(item)
        private.append(
            {
                "item_id": item_id,
                "sample_id": row["sample_id"],
                "conversation_id": row["conversation_id"],
                "stratum": row["stratum"],
                "selection_type": "outcome_enriched_secondary_human_eval",
                "position_to_model": {position: model for position, model in zip(("A", "B", "C"), order)},
                "oracle_axis_scores": row["oracle_axis_scores"],
                "oracle_means": row["oracle_means"],
                "basis_advantage": row["basis_advantage"],
                "response_sha256": {model: sha256_text(value) for model, value in row["responses"].items()},
            }
        )
    return public, private


def main() -> int:
    parser = argparse.ArgumentParser(description="3モデルLikert人手評価item作成")
    parser.add_argument("--dataset", choices=("mathdial", "meditod"), required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--oracle-raw", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--definition", type=Path)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 2 or args.count % 2:
        raise ValueError("--countは2以上の偶数にしてください。")
    definition_path = args.definition or Path(f"configs/user_evaluations/{args.dataset}_likert_v1.yaml")
    definition = load_definition(definition_path)
    if definition["dataset"] != args.dataset:
        raise ValueError("--datasetとdefinitionが一致しません。")
    candidates, oracle_axes = candidate_rows(args.responses, args.oracle_raw)
    selected = select_outcome_enriched(candidates, args.count)
    public, private = build_records(selected, definition, args.seed)
    for experiment, rows in public.items():
        write_jsonl(args.output_root / f"experiment_{experiment.lower()}" / "form_items_public.jsonl", rows)
    write_jsonl(args.output_root / "private_answer_key.jsonl", private)
    write_jsonl(args.output_root / "blind_review_sheet.jsonl", [
        {"item_id": item["item_id"], "conversation": item["conversation"], "response_a": item["response_a"], "response_b": item["response_b"], "response_c": item["response_c"]}
        for experiment in ("A", "B") for item in public[experiment]
    ])
    with (args.output_root / "candidate_audit.csv").open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["sample_id", "conversation_id", "stratum", "readability_passed", "readability_reason", "basis_advantage", "basis_mean", "base_mean", "random_dpo_mean", "selected"]
        writer = csv.DictWriter(file, fieldnames=fieldnames); writer.writeheader()
        selected_ids = {row["sample_id"] for row in selected}
        for row in sorted(candidates, key=lambda item: item["sample_id"]):
            writer.writerow({"sample_id": row["sample_id"], "conversation_id": row["conversation_id"], "stratum": row["stratum"], "readability_passed": row["readability_passed"], "readability_reason": row["readability_reason"], "basis_advantage": row["basis_advantage"], "basis_mean": row["oracle_means"]["basis"], "base_mean": row["oracle_means"]["base"], "random_dpo_mean": row["oracle_means"]["random_dpo"], "selected": row["sample_id"] in selected_ids})
    manifest = {
        "dataset": args.dataset,
        "survey_version": definition["survey_version"],
        "selection_type": "outcome_enriched_secondary_human_eval",
        "interpretation": "Oracle上でBASiS優位な項目へ富化した副次的人手評価であり、test全体の無条件な主結果ではない。",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": args.count,
        "items_per_experiment": args.count // 2,
        "seed": args.seed,
        "oracle_axes": list(oracle_axes),
        "stratum_counts": dict(Counter(row["stratum"] for row in selected)),
        "inputs": {"responses": str(args.responses), "responses_sha256": hashlib.sha256(args.responses.read_bytes()).hexdigest(), "oracle_raw": str(args.oracle_raw), "oracle_raw_sha256": hashlib.sha256(args.oracle_raw.read_bytes()).hexdigest(), "definition": str(definition_path), "definition_sha256": hashlib.sha256(definition_path.read_bytes()).hexdigest()},
        "public_information_excludes": ["model identity", "Oracle score", "answer position"],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.dataset}人手評価itemを書き出しました: {args.output_root} ({args.count}件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
