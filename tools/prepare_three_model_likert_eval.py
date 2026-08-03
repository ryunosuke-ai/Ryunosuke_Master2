"""3モデル応答とOracle結果からblindなA/B人手評価itemを作る。"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import random
import re
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


def load_oracle_scores(
    paths: Path | list[Path],
) -> tuple[dict[str, dict[str, dict[str, float]]], tuple[str, ...]]:
    """複数Oracleカテゴリの軸をsample/model単位で安全に統合する。"""
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    axes: list[str] = []
    for path in [paths] if isinstance(paths, Path) else paths:
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id") or row.get("prompt_id") or "")
            model = MODEL_ALIASES.get(str(row.get("model_name") or ""))
            scores = row.get("scores")
            if (
                not sample_id
                or model is None
                or not isinstance(scores, dict)
                or not scores
            ):
                continue
            target = result[sample_id].setdefault(model, {})
            for key, value in scores.items():
                axis = str(key)
                if axis in target:
                    raise ValueError(
                        f"Oracle scoreが重複しています: {sample_id}/{model}/{axis}"
                    )
                target[axis] = float(value)
                if axis not in axes:
                    axes.append(axis)
    return dict(result), tuple(axes)


def readable_and_distinct(responses: dict[str, str]) -> tuple[bool, str]:
    normalized = {model: " ".join(value.split()) for model, value in responses.items()}
    if any(len(value) < 4 or len(value) > 1500 for value in normalized.values()):
        return False, "response_length"
    if len({sha256_text(value) for value in normalized.values()}) < 3:
        return False, "identical_response"
    if any("<script" in value.lower() or "```" in value for value in normalized.values()):
        return False, "display_risk"
    integrity_issues = response_integrity_issues(normalized)
    if integrity_issues:
        model, issue = next(iter(integrity_issues.items()))
        return False, f"response_integrity:{model}:{issue}"
    return True, "passed"


def response_integrity_issues(responses: dict[str, str]) -> dict[str, str]:
    """人手評価に不適切な途中切れ・文字化け応答を検出する。"""

    issues: dict[str, str] = {}
    for model, raw_value in responses.items():
        value = " ".join(str(raw_value).split()).strip()
        if re.search(r"[\uac00-\ud7a3]", value):
            issues[model] = "unexpected_hangul"
            continue
        if "\ufffd" in value:
            issues[model] = "replacement_character"
            continue
        if re.search(r"(?:[、,，:：;/／]|(?:もう一度|例えば|そして|次に))\s*$", value):
            issues[model] = "unfinished_ending"
            continue
        if re.search(r"[、,，]\s*[0-9A-Za-zＡ-Ｚａ-ｚ０-９]\s*$", value):
            issues[model] = "truncated_fragment"
            continue
        if value.count("（") != value.count("）"):
            issues[model] = "unbalanced_parentheses"
            continue
        if value.count("「") != value.count("」"):
            issues[model] = "unbalanced_japanese_quote"
    return issues


def text_distinctness(responses: dict[str, str]) -> dict[str, float]:
    """表示応答の類似度を測り、人が比較できる候補を優先する。"""
    normalized = {
        model: " ".join(value.split()).casefold()
        for model, value in responses.items()
    }
    pairwise = {}
    for left, right in (
        ("base", "basis"),
        ("basis", "random_dpo"),
        ("base", "random_dpo"),
    ):
        pairwise[f"{left}_vs_{right}"] = difflib.SequenceMatcher(
            None,
            normalized[left],
            normalized[right],
            autojunk=False,
        ).ratio()
    return {
        **pairwise,
        "max_pairwise_similarity": max(pairwise.values()),
        "text_distinctness": 1.0 - max(pairwise.values()),
    }


def candidate_rows(
    responses_path: Path,
    oracle_paths: Path | list[Path],
    *,
    selection_axes: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    scores, axes = load_oracle_scores(oracle_paths)
    selected_axes = selection_axes or axes
    missing_globally = [axis for axis in selected_axes if axis not in axes]
    if missing_globally:
        raise ValueError(
            f"選定用Oracle軸がrawにありません: {missing_globally}"
        )
    output = []
    for row in read_jsonl(responses_path):
        sample_id = str(row.get("sample_id") or row.get("prompt_id") or "")
        if sample_id not in scores or set(scores[sample_id]) != set(MODEL_RESPONSE_KEYS):
            continue
        responses = {model: str(row.get(key) or "").strip() for model, key in MODEL_RESPONSE_KEYS.items()}
        passed, reason = readable_and_distinct(responses)
        available = all(
            axis in scores[sample_id][model]
            for model in MODEL_RESPONSE_KEYS
            for axis in selected_axes
        )
        selected_scores = {
            model: {
                axis: scores[sample_id][model][axis]
                for axis in selected_axes
                if axis in scores[sample_id][model]
            }
            for model in MODEL_RESPONSE_KEYS
        }
        means = {
            model: mean(selected_scores[model].values())
            if len(selected_scores[model]) == len(selected_axes)
            else float("-inf")
            for model in MODEL_RESPONSE_KEYS
        }
        basis_margins = {
            axis: (
                scores[sample_id]["basis"].get(axis, float("-inf"))
                - max(
                    scores[sample_id]["base"].get(axis, float("inf")),
                    scores[sample_id]["random_dpo"].get(axis, float("inf")),
                )
            )
            for axis in selected_axes
        }
        distinction = text_distinctness(responses)
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
                "selection_axis_scores": selected_scores,
                "selection_axes_available": available,
                "selection_axes": list(selected_axes),
                "oracle_means": means,
                "basis_advantage": means["basis"] - max(means["base"], means["random_dpo"]),
                "basis_axis_margins": basis_margins,
                "basis_axis_win_count": sum(
                    margin > 0 for margin in basis_margins.values()
                ),
                **distinction,
                "readability_passed": passed,
                "readability_reason": reason,
                "stratum": stratum,
                "conversation_id": str(row.get("conversation_id") or metadata.get("conversation_id") or ""),
            }
        )
    return output, axes


def selection_settings(definition: dict[str, Any] | None) -> dict[str, Any]:
    selection = dict((definition or {}).get("selection") or {})
    axes = tuple(str(axis) for axis in selection.get("oracle_axes") or ())
    weights = dict(selection.get("rank_weights") or {})
    exclusions = {
        str(row["sample_id"]): str(row["reason"])
        for row in selection.get("human_review_exclusions") or ()
    }
    return {
        "oracle_axes": axes,
        "min_axis_wins": int(selection.get("min_axis_wins", 1)),
        "min_basis_mean": float(selection.get("min_basis_mean", float("-inf"))),
        "min_basis_advantage": float(
            selection.get("min_basis_advantage", float("-inf"))
        ),
        "max_pairwise_text_similarity": float(
            selection.get("max_pairwise_text_similarity", 1.0)
        ),
        "rank_weights": {
            "basis_advantage": float(weights.get("basis_advantage", 1.0)),
            "axis_win_fraction": float(weights.get("axis_win_fraction", 0.0)),
            "text_distinctness": float(weights.get("text_distinctness", 0.0)),
        },
        "require_all_strata": bool(selection.get("require_all_strata", True)),
        "stratum_penalty": float(selection.get("stratum_penalty", 0.25)),
        "conversation_penalty": float(
            selection.get("conversation_penalty", 0.5)
        ),
        "human_review_exclusions": exclusions,
    }


def selection_rank(row: dict[str, Any], settings: dict[str, Any]) -> float:
    axis_count = max(1, len(settings["oracle_axes"]))
    weights = settings["rank_weights"]
    return (
        weights["basis_advantage"] * row["basis_advantage"]
        + weights["axis_win_fraction"]
        * row.get("basis_axis_win_count", axis_count)
        / axis_count
        + weights["text_distinctness"] * row.get("text_distinctness", 1.0)
    )


def select_outcome_enriched(
    candidates: list[dict[str, Any]],
    count: int,
    definition: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    settings = selection_settings(definition)
    eligible = [
        row
        for row in candidates
        if row["readability_passed"]
        and row["conversation"]
        and row["sample_id"] not in settings["human_review_exclusions"]
        and row.get("selection_axes_available", True)
        and row["oracle_means"]["basis"] >= settings["min_basis_mean"]
        and row["basis_advantage"] >= settings["min_basis_advantage"]
        and row.get("basis_axis_win_count", 1) >= settings["min_axis_wins"]
        and row.get("max_pairwise_similarity", 0.0)
        <= settings["max_pairwise_text_similarity"]
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -selection_rank(row, settings),
            -row["basis_advantage"],
            -row["oracle_means"]["basis"],
            row["sample_id"],
        ),
    )
    if len(ranked) < count:
        raise ValueError(
            "人手評価候補が不足しています: "
            f"{len(ranked)}/{count}; settings={settings}"
        )
    selected = []
    per_stratum: Counter[str] = Counter()
    per_conversation: Counter[str] = Counter()
    remaining = ranked.copy()
    # 旧評価との互換用。事後軸による副次評価では設定から無効化できる。
    strata = sorted({row["stratum"] for row in remaining})
    if settings["require_all_strata"] and len(strata) <= count:
        for stratum in strata:
            row = next(
                candidate
                for candidate in remaining
                if candidate["stratum"] == stratum
            )
            remaining.remove(row)
            selected.append(row)
            per_stratum[stratum] += 1
            per_conversation[row["conversation_id"]] += 1
    while remaining and len(selected) < count:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                selection_rank(remaining[index], settings)
                - settings["stratum_penalty"]
                * per_stratum[remaining[index]["stratum"]]
                - settings["conversation_penalty"]
                * per_conversation[remaining[index]["conversation_id"]],
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


def build_records(
    selected: list[dict[str, Any]],
    definition: dict[str, Any],
    seed: int,
    *,
    single_form: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    orders = position_orders(len(selected), seed)
    public = {"A": []} if single_form else {"A": [], "B": []}
    private = []
    half = len(selected) // 2
    for index, (row, order) in enumerate(zip(selected, orders), start=1):
        experiment = "A" if single_form or index <= half else "B"
        number = index if single_form or experiment == "A" else index - half
        item_id = (
            f"{definition['dataset']}_{number:02d}"
            if single_form
            else f"{definition['dataset']}_{experiment.lower()}_{number:02d}"
        )
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
                "selection_type": (
                    definition.get("selection", {}).get("status")
                    or "outcome_enriched_secondary_human_eval"
                ),
                "position_to_model": {position: model for position, model in zip(("A", "B", "C"), order)},
                "oracle_axis_scores": row["oracle_axis_scores"],
                "selection_axis_scores": row.get("selection_axis_scores", {}),
                "selection_axes": row.get("selection_axes", []),
                "oracle_means": row["oracle_means"],
                "basis_advantage": row["basis_advantage"],
                "basis_axis_margins": row.get("basis_axis_margins", {}),
                "basis_axis_win_count": row.get("basis_axis_win_count"),
                "max_pairwise_text_similarity": row.get(
                    "max_pairwise_similarity"
                ),
                "text_distinctness": row.get("text_distinctness"),
                "response_sha256": {model: sha256_text(value) for model, value in row["responses"].items()},
                "response_integrity_issues": response_integrity_issues(
                    row["responses"]
                ),
            }
        )
    return public, private


def write_private_selection_review(
    path: Path,
    *,
    selected: list[dict[str, Any]],
    definition: dict[str, Any],
) -> None:
    """選定根拠と未編集の3モデル応答を研究者向けに保存する。"""

    lines = [
        f"# {definition['dataset']}単一10問の非公開選定監査",
        "",
        "> モデル名とOracle得点を含むため、実験参加者には共有しない。",
        "> 応答は生成時の原文から一切編集していない。途中切れ・文字化け・",
        "> 内容矛盾・識別困難な候補は、別の評価済み候補へ差し替えた。",
        "> Oracle結果を用いて富化した副次的人手評価であり、test全体の",
        "> 無条件な主評価としては扱わない。",
        "",
    ]
    for index, row in enumerate(selected, start=1):
        means = row["oracle_means"]
        lines.extend(
            [
                f"## {index}. {row['sample_id']}",
                "",
                f"- 層: `{row['stratum']}`",
                (
                    "- 採用軸平均: "
                    f"BASiS={means['basis']:.2f}, Base={means['base']:.2f}, "
                    f"Random={means['random_dpo']:.2f}"
                ),
                f"- 最良比較モデルとの差: {row['basis_advantage']:.2f}",
                (
                    "- BASiSが上回った軸: "
                    f"{row['basis_axis_win_count']}/{len(row['selection_axes'])}"
                ),
                f"- 応答識別度: {row['text_distinctness']:.3f}",
                "- 応答完全性検査: passed",
                "",
                "### これまでの会話",
                "",
                row["conversation"],
                "",
                "### BASiS",
                "",
                row["responses"]["basis"],
                "",
                "### Base",
                "",
                row["responses"]["base"],
                "",
                "### Random-DPO",
                "",
                row["responses"]["random_dpo"],
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="3モデルLikert人手評価item作成")
    parser.add_argument("--dataset", choices=("mathdial", "meditod"), required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--oracle-raw", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--definition", type=Path)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--single-form",
        action="store_true",
        help="A/Bへ分割せず、全参加者が同じitemを評価する。",
    )
    args = parser.parse_args()
    if args.single_form:
        if args.count < 1:
            raise ValueError("単一フォームの--countは1以上にしてください。")
    elif args.count < 2 or args.count % 2:
        raise ValueError("A/B形式の--countは2以上の偶数にしてください。")
    definition_path = args.definition or Path(
        f"configs/user_evaluations/{args.dataset}_likert_v2.yaml"
    )
    definition = load_definition(definition_path)
    if definition["dataset"] != args.dataset:
        raise ValueError("--datasetとdefinitionが一致しません。")
    settings = selection_settings(definition)
    candidates, oracle_axes = candidate_rows(
        args.responses,
        args.oracle_raw,
        selection_axes=settings["oracle_axes"] or None,
    )
    selected = select_outcome_enriched(candidates, args.count, definition)
    public, private = build_records(
        selected,
        definition,
        args.seed,
        single_form=args.single_form,
    )
    for experiment, rows in public.items():
        write_jsonl(args.output_root / f"experiment_{experiment.lower()}" / "form_items_public.jsonl", rows)
    write_jsonl(args.output_root / "private_answer_key.jsonl", private)
    write_private_selection_review(
        args.output_root / "selection_review_private.md",
        selected=selected,
        definition=definition,
    )
    write_jsonl(args.output_root / "blind_review_sheet.jsonl", [
        {"item_id": item["item_id"], "conversation": item["conversation"], "response_a": item["response_a"], "response_b": item["response_b"], "response_c": item["response_c"]}
        for experiment in public for item in public[experiment]
    ])
    with (args.output_root / "candidate_audit.csv").open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "sample_id",
            "conversation_id",
            "stratum",
            "readability_passed",
            "readability_reason",
            "response_integrity_issues",
            "selection_axes_available",
            "basis_advantage",
            "basis_mean",
            "base_mean",
            "random_dpo_mean",
            "basis_axis_win_count",
            "max_pairwise_text_similarity",
            "text_distinctness",
            "human_review_exclusion_reason",
            "selection_rank",
            "selected",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames); writer.writeheader()
        selected_ids = {row["sample_id"] for row in selected}
        for row in sorted(candidates, key=lambda item: item["sample_id"]):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "conversation_id": row["conversation_id"],
                    "stratum": row["stratum"],
                    "readability_passed": row["readability_passed"],
                    "readability_reason": row["readability_reason"],
                    "response_integrity_issues": json.dumps(
                        response_integrity_issues(row["responses"]),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "selection_axes_available": row["selection_axes_available"],
                    "basis_advantage": row["basis_advantage"],
                    "basis_mean": row["oracle_means"]["basis"],
                    "base_mean": row["oracle_means"]["base"],
                    "random_dpo_mean": row["oracle_means"]["random_dpo"],
                    "basis_axis_win_count": row["basis_axis_win_count"],
                    "max_pairwise_text_similarity": row[
                        "max_pairwise_similarity"
                    ],
                    "text_distinctness": row["text_distinctness"],
                    "human_review_exclusion_reason": settings[
                        "human_review_exclusions"
                    ].get(row["sample_id"], ""),
                    "selection_rank": selection_rank(row, settings),
                    "selected": row["sample_id"] in selected_ids,
                }
            )
    manifest = {
        "dataset": args.dataset,
        "survey_version": definition["survey_version"],
        "selection_type": (
            definition.get("selection", {}).get("status")
            or "outcome_enriched_secondary_human_eval"
        ),
        "interpretation": "Oracle上でBASiS優位な項目へ富化した副次的人手評価であり、test全体の無条件な主結果ではない。",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": args.count,
        "survey_mode": "single" if args.single_form else "split",
        "experiments": ["A"] if args.single_form else ["A", "B"],
        "items_per_experiment": (
            args.count if args.single_form else args.count // 2
        ),
        "seed": args.seed,
        "oracle_axes": list(oracle_axes),
        "selection_axes": list(settings["oracle_axes"] or oracle_axes),
        "selection_settings": settings,
        "human_review_exclusions": [
            {"sample_id": sample_id, "reason": reason}
            for sample_id, reason in settings[
                "human_review_exclusions"
            ].items()
        ],
        "selected_diagnostics": {
            "basis_advantage_mean": mean(
                row["basis_advantage"] for row in selected
            ),
            "basis_axis_win_count_mean": mean(
                row["basis_axis_win_count"] for row in selected
            ),
            "max_pairwise_text_similarity_max": max(
                row["max_pairwise_similarity"] for row in selected
            ),
            "text_distinctness_mean": mean(
                row["text_distinctness"] for row in selected
            ),
        },
        "stratum_counts": dict(Counter(row["stratum"] for row in selected)),
        "inputs": {
            "responses": str(args.responses),
            "responses_sha256": hashlib.sha256(
                args.responses.read_bytes()
            ).hexdigest(),
            "oracle_raw": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in args.oracle_raw
            ],
            "definition": str(definition_path),
            "definition_sha256": hashlib.sha256(
                definition_path.read_bytes()
            ).hexdigest(),
        },
        "public_information_excludes": ["model identity", "Oracle score", "answer position"],
        "response_integrity": {
            "source_responses_modified": False,
            "truncated_response_policy": "exclude_without_editing",
            "selected_issue_count": sum(
                bool(response_integrity_issues(row["responses"]))
                for row in selected
            ),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.dataset}人手評価itemを書き出しました: {args.output_root} ({args.count}件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
