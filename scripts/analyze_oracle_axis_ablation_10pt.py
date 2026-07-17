#!/usr/bin/env python3
"""10段階Oracle評価結果の評価軸別・再集計分析を行う。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


DEFAULT_ROOT = Path(
    "artifacts/evaluations/oracle_eval_runs/"
    "esconv_topconf_three_model_gpt54_100_10pt_topconf_three_model_10pt"
)

MODEL_KEYS = ("base", "bayes_dpo", "random_dpo")
MODEL_LABELS = {
    "base": "Base",
    "bayes_dpo": "BASiS",
    "random_dpo": "Random-DPO",
}
PAIRWISE = (
    ("BASiS_vs_Base", "bayes_dpo", "base"),
    ("BASiS_vs_Random-DPO", "bayes_dpo", "random_dpo"),
    ("Base_vs_Random-DPO", "base", "random_dpo"),
)
BASIS_PAIRS = (
    ("BASiS_vs_Base", "bayes_dpo", "base"),
    ("BASiS_vs_Random-DPO", "bayes_dpo", "random_dpo"),
)
ALPHA = 0.05


@dataclass(frozen=True)
class CategorySpec:
    name: str
    directory: str
    axes: tuple[str, ...]


@dataclass(frozen=True)
class ScoreSetSpec:
    name: str
    category: str
    axes: tuple[str, ...]


CATEGORY_SPECS = {
    "conversation_style": CategorySpec(
        name="conversation_style",
        directory="oracle_conversation_style_10pt",
        axes=("fluency", "engagingness", "style_consistency", "style_similarity"),
    ),
    "strategy_transition": CategorySpec(
        name="strategy_transition",
        directory="oracle_strategy_transition_10pt",
        axes=("strategy_appropriateness_score", "transition_smoothness_score"),
    ),
    "tst": CategorySpec(
        name="tst",
        directory="oracle_tst_10pt",
        axes=("style_strength", "content_preservation", "naturalness"),
    ),
    "usr_quality": CategorySpec(
        name="usr_quality",
        directory="oracle_usr_quality_10pt",
        axes=(
            "understandable",
            "natural",
            "maintains_context",
            "interesting_or_engaging",
            "overall_quality",
        ),
    ),
}

SCORE_SET_SPECS = (
    ScoreSetSpec(
        "conversation_style_full",
        "conversation_style",
        ("fluency", "engagingness", "style_consistency", "style_similarity"),
    ),
    ScoreSetSpec(
        "conversation_style_core3",
        "conversation_style",
        ("fluency", "style_consistency", "style_similarity"),
    ),
    ScoreSetSpec(
        "conversation_style_core2",
        "conversation_style",
        ("style_consistency", "style_similarity"),
    ),
    ScoreSetSpec(
        "strategy_transition_full",
        "strategy_transition",
        ("strategy_appropriateness_score", "transition_smoothness_score"),
    ),
    ScoreSetSpec(
        "strategy_transition_core",
        "strategy_transition",
        ("strategy_appropriateness_score",),
    ),
    ScoreSetSpec(
        "tst_full",
        "tst",
        ("style_strength", "content_preservation", "naturalness"),
    ),
    ScoreSetSpec("tst_style_only", "tst", ("style_strength",)),
    ScoreSetSpec(
        "tst_quality_constraints",
        "tst",
        ("content_preservation", "naturalness"),
    ),
    ScoreSetSpec(
        "usr_quality_full",
        "usr_quality",
        (
            "understandable",
            "natural",
            "maintains_context",
            "interesting_or_engaging",
            "overall_quality",
        ),
    ),
)

WINNER_FLIP_SPECS = (
    ("conversation_style_full", "conversation_style_core3"),
    ("conversation_style_full", "conversation_style_core2"),
    ("strategy_transition_full", "strategy_transition_core"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="10段階Oracle評価の評価軸別影響分析を行う。"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--tie_threshold", type=float, default=0.25)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_category_scores(root: Path) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """category -> sample_id -> model_key -> axis_scores を返す。"""

    data: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for category, spec in CATEGORY_SPECS.items():
        raw_path = root / spec.directory / "raw.jsonl"
        if not raw_path.exists():
            raise FileNotFoundError(f"raw.jsonl が見つかりません: {raw_path}")

        category_rows: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                sample_id = rec["sample_id"]
                model_name = rec["model_name"]
                scores = rec.get("scores", {})
                missing = [axis for axis in spec.axes if axis not in scores]
                if missing:
                    raise ValueError(
                        f"{raw_path}: {sample_id}/{model_name} に軸がありません: {missing}"
                    )
                category_rows[sample_id][model_name] = {
                    axis: float(scores[axis]) for axis in spec.axes
                }

        incomplete = [
            sample_id
            for sample_id, rows in category_rows.items()
            if not all(model in rows for model in MODEL_KEYS)
        ]
        if incomplete:
            raise ValueError(
                f"{category}: 3モデルが揃っていないsample_idがあります: {incomplete[:10]}"
            )
        data[category] = dict(category_rows)
    return data


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def bootstrap_mean_ci(
    values: list[float],
    *,
    n_bootstrap: int,
    rng: random.Random,
) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if n_bootstrap <= 0:
        m = mean(values)
        se = sample_std(values) / math.sqrt(len(values))
        return m - 1.96 * se, m + 1.96 * se

    n = len(values)
    boot_means = []
    for _ in range(n_bootstrap):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        boot_means.append(total / n)
    boot_means.sort()
    low_idx = int(0.025 * n_bootstrap)
    high_idx = min(n_bootstrap - 1, int(0.975 * n_bootstrap))
    return boot_means[low_idx], boot_means[high_idx]


def average_ranks(values: list[float]) -> tuple[list[float], list[int]]:
    """昇順で順位を付け、同点には平均順位を与える。"""

    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    tie_sizes: list[int] = []
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for pos in range(i, j):
            ranks[indexed[pos][0]] = avg_rank
        if j - i > 1:
            tie_sizes.append(j - i)
        i = j
    return ranks, tie_sizes


def friedman_test(values_by_model: dict[str, list[float]]) -> tuple[float, float, float]:
    n = len(next(iter(values_by_model.values())))
    k = len(MODEL_KEYS)
    rank_sums = [0.0] * k
    tie_correction_terms = 0

    for idx in range(n):
        values = [values_by_model[model][idx] for model in MODEL_KEYS]
        ranks, tie_sizes = average_ranks(values)
        for model_idx, rank in enumerate(ranks):
            rank_sums[model_idx] += rank
        tie_correction_terms += sum(tie**3 - tie for tie in tie_sizes)

    chi2 = (12.0 / (n * k * (k + 1))) * sum(rank**2 for rank in rank_sums)
    chi2 -= 3 * n * (k + 1)

    correction = 1.0 - tie_correction_terms / (n * (k**3 - k))
    if correction > 0:
        chi2 /= correction

    # df = k - 1 = 2 のカイ二乗分布では survival function が exp(-x/2)。
    p_value = math.exp(-max(0.0, chi2) / 2.0)
    kendalls_w = chi2 / (n * (k - 1))
    return chi2, p_value, kendalls_w


def paired_permutation_p(
    diffs: list[float],
    *,
    n_permutations: int,
    rng: random.Random,
) -> float:
    observed = abs(mean(diffs))
    if observed == 0:
        return 1.0

    count = 1
    for _ in range(n_permutations):
        signed_sum = 0.0
        for diff in diffs:
            signed_sum += diff if rng.random() < 0.5 else -diff
        if abs(signed_sum / len(diffs)) >= observed - 1e-12:
            count += 1
    return count / (n_permutations + 1)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda idx: p_values[idx])
    adjusted = [0.0] * len(p_values)
    prev = 0.0
    total = len(p_values)
    for rank, idx in enumerate(order):
        value = min(1.0, p_values[idx] * (total - rank))
        value = max(prev, value)
        adjusted[idx] = value
        prev = value
    return adjusted


def effect_size_dz(diffs: list[float]) -> float:
    sd = sample_std(diffs)
    return mean(diffs) / sd if sd else 0.0


def win_tie_loss(diffs: list[float], tie_threshold: float) -> tuple[int, int, int]:
    wins = sum(1 for diff in diffs if diff >= tie_threshold)
    losses = sum(1 for diff in diffs if diff <= -tie_threshold)
    ties = len(diffs) - wins - losses
    return wins, ties, losses


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def values_for_axis(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    category: str,
    axis: str,
    model: str,
) -> list[float]:
    rows = data[category]
    return [rows[sample_id][model][axis] for sample_id in sorted(rows)]


def values_for_score_set(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    spec: ScoreSetSpec,
    model: str,
) -> list[float]:
    rows = data[spec.category]
    return [
        mean([rows[sample_id][model][axis] for axis in spec.axes])
        for sample_id in sorted(rows)
    ]


def build_axis_level_summary(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    n_bootstrap: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, spec in CATEGORY_SPECS.items():
        for axis in spec.axes:
            for model in MODEL_KEYS:
                values = values_for_axis(data, category, axis, model)
                ci_low, ci_high = bootstrap_mean_ci(
                    values, n_bootstrap=n_bootstrap, rng=rng
                )
                rows.append(
                    {
                        "category": category,
                        "axis": axis,
                        "model_name": MODEL_LABELS[model],
                        "n": len(values),
                        "mean": mean(values),
                        "std": sample_std(values),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                    }
                )
    return rows


def build_axis_level_pairwise(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    tie_threshold: float,
    n_permutations: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, spec in CATEGORY_SPECS.items():
        sample_ids = sorted(data[category])
        for axis in spec.axes:
            axis_rows = []
            raw_p_values = []
            for comparison, left, right in PAIRWISE:
                diffs = [
                    data[category][sample_id][left][axis]
                    - data[category][sample_id][right][axis]
                    for sample_id in sample_ids
                ]
                wins, ties, losses = win_tie_loss(diffs, tie_threshold)
                p_raw = paired_permutation_p(
                    diffs, n_permutations=n_permutations, rng=rng
                )
                raw_p_values.append(p_raw)
                axis_rows.append(
                    {
                        "category": category,
                        "axis": axis,
                        "comparison": comparison,
                        "n": len(diffs),
                        "mean_diff": mean(diffs),
                        "median_diff": median(diffs),
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                        "p_raw": p_raw,
                        "p_holm": None,
                        "effect_size": effect_size_dz(diffs),
                        "significant": None,
                    }
                )
            for row, p_holm in zip(axis_rows, holm_adjust(raw_p_values)):
                row["p_holm"] = p_holm
                row["significant"] = str(p_holm < ALPHA).lower()
                rows.append(row)
    return rows


def build_core_summary(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    n_bootstrap: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SCORE_SET_SPECS:
        for model in MODEL_KEYS:
            values = values_for_score_set(data, spec, model)
            ci_low, ci_high = bootstrap_mean_ci(
                values, n_bootstrap=n_bootstrap, rng=rng
            )
            rows.append(
                {
                    "score_set": spec.name,
                    "model_name": MODEL_LABELS[model],
                    "n": len(values),
                    "mean": mean(values),
                    "std": sample_std(values),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return rows


def build_core_friedman_and_posthoc(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    tie_threshold: float,
    n_permutations: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    friedman_rows: list[dict[str, Any]] = []
    posthoc_rows: list[dict[str, Any]] = []

    for spec in SCORE_SET_SPECS:
        values_by_model = {
            model: values_for_score_set(data, spec, model) for model in MODEL_KEYS
        }
        chi2, p_value, kendalls_w = friedman_test(values_by_model)
        significant = p_value < ALPHA
        friedman_rows.append(
            {
                "score_set": spec.name,
                "n": len(values_by_model["base"]),
                "friedman_chi2": chi2,
                "p_value": p_value,
                "kendalls_w": kendalls_w,
                "significant": str(significant).lower(),
            }
        )
        if not significant:
            continue

        raw_p_values = []
        score_rows = []
        for comparison, left, right in PAIRWISE:
            diffs = [
                left_value - right_value
                for left_value, right_value in zip(
                    values_by_model[left], values_by_model[right]
                )
            ]
            wins, ties, losses = win_tie_loss(diffs, tie_threshold)
            p_raw = paired_permutation_p(diffs, n_permutations=n_permutations, rng=rng)
            raw_p_values.append(p_raw)
            score_rows.append(
                {
                    "score_set": spec.name,
                    "comparison": comparison,
                    "n": len(diffs),
                    "mean_diff": mean(diffs),
                    "median_diff": median(diffs),
                    "p_raw": p_raw,
                    "p_holm": None,
                    "effect_size": effect_size_dz(diffs),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "significant": None,
                }
            )
        for row, p_holm in zip(score_rows, holm_adjust(raw_p_values)):
            row["p_holm"] = p_holm
            row["significant"] = str(p_holm < ALPHA).lower()
            posthoc_rows.append(row)

    return friedman_rows, posthoc_rows


def build_axis_drag_analysis(
    data: dict[str, dict[str, dict[str, dict[str, float]]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, spec in CATEGORY_SPECS.items():
        for comparison, left, right in BASIS_PAIRS:
            for axis in spec.axes:
                left_values = values_for_axis(data, category, axis, left)
                right_values = values_for_axis(data, category, axis, right)
                axis_mean_diff = mean(left_values) - mean(right_values)
                if axis_mean_diff > 0:
                    direction = "helps_BASiS"
                elif axis_mean_diff < 0:
                    direction = "hurts_BASiS"
                else:
                    direction = "neutral"
                rows.append(
                    {
                        "category": category,
                        "axis": axis,
                        "comparison": comparison,
                        "axis_mean_diff": axis_mean_diff,
                        "contribution_to_overall_diff": axis_mean_diff
                        / len(spec.axes),
                        "direction": direction,
                    }
                )
    return rows


def classify_basis(diff: float, tie_threshold: float) -> str:
    if diff >= tie_threshold:
        return "basis_win"
    if diff <= -tie_threshold:
        return "basis_loss"
    return "tie"


def score_set_values_by_sample(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    spec: ScoreSetSpec,
) -> dict[str, dict[str, float]]:
    rows = data[spec.category]
    output: dict[str, dict[str, float]] = {}
    for sample_id in sorted(rows):
        output[sample_id] = {
            model: mean([rows[sample_id][model][axis] for axis in spec.axes])
            for model in MODEL_KEYS
        }
    return output


def build_winner_flip_analysis(
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    tie_threshold: float,
) -> list[dict[str, Any]]:
    spec_by_name = {spec.name: spec for spec in SCORE_SET_SPECS}
    rows: list[dict[str, Any]] = []

    for before_name, after_name in WINNER_FLIP_SPECS:
        before = score_set_values_by_sample(data, spec_by_name[before_name])
        after = score_set_values_by_sample(data, spec_by_name[after_name])
        for comparison, left, right in BASIS_PAIRS:
            before_counts = {"basis_win": 0, "tie": 0, "basis_loss": 0}
            after_counts = {"basis_win": 0, "tie": 0, "basis_loss": 0}
            losses_to_wins = 0
            losses_to_ties = 0
            ties_to_wins = 0

            for sample_id in sorted(before):
                before_status = classify_basis(
                    before[sample_id][left] - before[sample_id][right],
                    tie_threshold,
                )
                after_status = classify_basis(
                    after[sample_id][left] - after[sample_id][right],
                    tie_threshold,
                )
                before_counts[before_status] += 1
                after_counts[after_status] += 1
                if before_status == "basis_loss" and after_status == "basis_win":
                    losses_to_wins += 1
                if before_status == "basis_loss" and after_status == "tie":
                    losses_to_ties += 1
                if before_status == "tie" and after_status == "basis_win":
                    ties_to_wins += 1

            rows.append(
                {
                    "score_set_before": before_name,
                    "score_set_after": after_name,
                    "comparison": comparison,
                    "n": len(before),
                    "before_basis_wins": before_counts["basis_win"],
                    "before_ties": before_counts["tie"],
                    "before_basis_losses": before_counts["basis_loss"],
                    "after_basis_wins": after_counts["basis_win"],
                    "after_ties": after_counts["tie"],
                    "after_basis_losses": after_counts["basis_loss"],
                    "losses_to_wins": losses_to_wins,
                    "losses_to_ties": losses_to_ties,
                    "ties_to_wins": ties_to_wins,
                }
            )
    return rows


def read_strategy_existing_metrics(root: Path) -> list[dict[str, Any]]:
    summary_path = root / "oracle_strategy_transition_10pt" / "summary.csv"
    if not summary_path.exists():
        return []

    fields = [
        "model_name",
        "count",
        "mean_strategy_appropriateness",
        "mean_transition_smoothness",
        "strategy_accuracy",
        "strategy_macro_f1",
        "strategy_weighted_f1",
        "strategy_jsd_to_esconv",
        "strategy_tvd_to_esconv",
        "strategy_entropy",
        "most_frequent_strategy",
        "most_frequent_strategy_ratio",
        "transition_jsd_to_esconv",
        "transition_tvd_to_esconv",
        "transition_entropy",
    ]
    rows: list[dict[str, Any]] = []
    with summary_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({field: row.get(field, "") for field in fields})
    return rows


def rows_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(grouped)


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def build_report(
    output_dir: Path,
    axis_summary: list[dict[str, Any]],
    axis_pairwise: list[dict[str, Any]],
    core_summary: list[dict[str, Any]],
    core_friedman: list[dict[str, Any]],
    core_posthoc: list[dict[str, Any]],
    axis_drag: list[dict[str, Any]],
    winner_flip: list[dict[str, Any]],
    strategy_existing_metrics: list[dict[str, Any]],
) -> str:
    summary_by_category_axis_model = {
        (row["category"], row["axis"], row["model_name"]): row
        for row in axis_summary
    }
    core_by_score_model = {
        (row["score_set"], row["model_name"]): row for row in core_summary
    }

    def means_for(category: str, axis: str) -> list[str]:
        return [
            fmt(summary_by_category_axis_model[(category, axis, label)]["mean"])
            for label in ("Base", "BASiS", "Random-DPO")
        ]

    conv_axis_rows = [
        [axis, *means_for("conversation_style", axis)]
        for axis in CATEGORY_SPECS["conversation_style"].axes
    ]
    strategy_axis_rows = [
        [axis, *means_for("strategy_transition", axis)]
        for axis in CATEGORY_SPECS["strategy_transition"].axes
    ]
    tst_axis_rows = [
        [axis, *means_for("tst", axis)] for axis in CATEGORY_SPECS["tst"].axes
    ]
    usr_axis_rows = [
        [axis, *means_for("usr_quality", axis)]
        for axis in CATEGORY_SPECS["usr_quality"].axes
    ]

    drag_sorted = sorted(
        [
            row
            for row in axis_drag
            if row["comparison"] in {"BASiS_vs_Base", "BASiS_vs_Random-DPO"}
        ],
        key=lambda row: row["axis_mean_diff"],
    )
    worst_drag_rows = [
        [
            row["category"],
            row["axis"],
            row["comparison"],
            fmt(row["axis_mean_diff"]),
            fmt(row["contribution_to_overall_diff"]),
        ]
        for row in drag_sorted[:8]
    ]

    core_rows = []
    for score_set in [
        "conversation_style_full",
        "conversation_style_core3",
        "conversation_style_core2",
        "strategy_transition_full",
        "strategy_transition_core",
        "tst_full",
        "tst_style_only",
        "tst_quality_constraints",
        "usr_quality_full",
    ]:
        means = [
            core_by_score_model[(score_set, label)]["mean"]
            for label in ("Base", "BASiS", "Random-DPO")
        ]
        winner = ("Base", "BASiS", "Random-DPO")[max(range(3), key=lambda idx: means[idx])]
        friedman = next(row for row in core_friedman if row["score_set"] == score_set)
        core_rows.append(
            [
                score_set,
                fmt(means[0]),
                fmt(means[1]),
                fmt(means[2]),
                winner,
                fmt(friedman["p_value"], 4),
                friedman["significant"],
            ]
        )

    conv_posthoc = [
        row
        for row in core_posthoc
        if row["score_set"]
        in {
            "conversation_style_full",
            "conversation_style_core3",
            "conversation_style_core2",
        }
        and row["comparison"].startswith("BASiS")
    ]
    strategy_posthoc = [
        row
        for row in core_posthoc
        if row["score_set"] in {"strategy_transition_full", "strategy_transition_core"}
        and row["comparison"].startswith("BASiS")
    ]
    tst_posthoc = [
        row
        for row in core_posthoc
        if row["score_set"]
        in {"tst_full", "tst_style_only", "tst_quality_constraints"}
        and row["comparison"].startswith("BASiS")
    ]

    flip_rows = [
        [
            row["score_set_before"],
            row["score_set_after"],
            row["comparison"],
            f"{row['before_basis_wins']}/{row['before_ties']}/{row['before_basis_losses']}",
            f"{row['after_basis_wins']}/{row['after_ties']}/{row['after_basis_losses']}",
            row["losses_to_wins"],
            row["losses_to_ties"],
            row["ties_to_wins"],
        ]
        for row in winner_flip
    ]
    strategy_metric_rows = [
        [
            row["model_name"],
            row["strategy_accuracy"],
            row["strategy_macro_f1"],
            row["strategy_weighted_f1"],
            row["strategy_jsd_to_esconv"],
            row["strategy_tvd_to_esconv"],
            row["strategy_entropy"],
            row["transition_jsd_to_esconv"],
            row["transition_tvd_to_esconv"],
            row["transition_entropy"],
        ]
        for row in strategy_existing_metrics
    ]

    report = f"""# 10段階Oracle評価: 評価軸別影響分析

## 位置づけ

この分析は、既存の10段階Oracle評価結果に対する診断的な再集計です。元の4カテゴリ評価結果を置き換えるものではありません。目的は、`tst`、`strategy_transition`、`conversation_style` に一般対話品質寄りの軸が混ざっていないかを確認し、「目的スタイル評価」と「一般対話品質評価」を分離することです。

検定は同一promptに対する Base / BASiS(Bayes-DPO) / Random-DPO の対応ありデータとして行いました。3モデル全体差にはFriedman検定、効果量にはKendall's W、有意な再集計スコアの事後比較には対応あり符号反転permutation testとカテゴリ内Holm補正を使っています。

## 評価軸別平均

### conversation_style

{markdown_table(conv_axis_rows, ["axis", "Base", "BASiS", "Random-DPO"])}

`engagingness` がBASiSを大きく下げています。BASiSは `fluency` ではBase/Random-DPOより高い一方、`style_consistency` と `style_similarity` ではBaseやRandom-DPOと近いかやや低い程度です。したがって、conversation_style全体の低下は主に `engagingness` に集中しています。

### strategy_transition

{markdown_table(strategy_axis_rows, ["axis", "Base", "BASiS", "Random-DPO"])}

両軸ともBASiSが平均では最も高いです。`transition_smoothness_score` は一般的な会話進行の自然さを含む可能性がありますが、この結果ではBASiSを下げる軸ではありません。ただし、既存summaryの `strategy_accuracy`、macro_f1、weighted_f1、strategy/transition JSD・TVD・entropy は、LLM Oracleが推定した理想戦略やESConv参照分布との一致・分布差であり、真の人手ラベルとの一致ではありません。

{markdown_table(strategy_metric_rows, ["model", "accuracy", "macro_f1", "weighted_f1", "strategy_jsd", "strategy_tvd", "strategy_entropy", "transition_jsd", "transition_tvd", "transition_entropy"])}

### tst

{markdown_table(tst_axis_rows, ["axis", "Base", "BASiS", "Random-DPO"])}

BASiSは `style_strength` で最も強く、`content_preservation` と `naturalness` でもBase/Random-DPO以上です。TSTでのBASiS優位は、スタイル強度だけでなく品質制約軸にも支えられています。

### usr_quality

{markdown_table(usr_axis_rows, ["axis", "Base", "BASiS", "Random-DPO"])}

BASiSは `understandable`、`natural`、`maintains_context` では高いか同等ですが、`interesting_or_engaging` が大きく低く、`overall_quality` もBase/Random-DPOを下回ります。USR品質でのBASiS低下は、主に会話の広がり・返信しやすさ・全体品質評価に由来します。

## 再集計スコア

{markdown_table(core_rows, ["score_set", "Base", "BASiS", "Random-DPO", "winner", "Friedman p", "significant"])}

`conversation_style` では、4軸すべてのfullではBASiSが低いですが、`engagingness` を除いたcore3ではBaseとの差がほぼ消えます。さらに `fluency` も除いたcore2ではRandom-DPOがわずかに高く、BASiSはBaseに近い水準です。つまり、BASiSのconversation_style低下は、目的スタイル模倣というより会話継続性を含む `engagingness` の影響が大きいです。

`strategy_transition` はfullでもcoreでも3モデル間の有意差は出ていません。`transition_smoothness_score` を除いて `strategy_appropriateness_score` のみにしても、BASiSが平均では最も高いものの、有意差までは確認できません。

`tst` はfull、style_only、quality_constraintsのいずれでもBASiSが最も高いです。これは、BASiSがESConvらしい支援スタイルを強めつつ、内容保持や自然さを大きく犠牲にしていないことを示します。

## BASiSを下げていた軸

{markdown_table(worst_drag_rows, ["category", "axis", "comparison", "axis_mean_diff", "overall_contribution"])}

最も大きくBASiSを下げていたのは、`conversation_style.engagingness` と `usr_quality.interesting_or_engaging` です。どちらも「次に話しやすい」「会話が広がる」「返信したくなる」といった一般対話品質・会話継続性に近い性質を持ちます。これらは本研究の目的コーパスらしさ評価から完全に不要という意味ではありませんが、USR系の一般対話品質評価と重複するため、主たるスタイル模倣評価からは分離して扱うのが妥当です。

## 勝敗変化

{markdown_table(flip_rows, ["before", "after", "comparison", "before W/T/L", "after W/T/L", "losses_to_wins", "losses_to_ties", "ties_to_wins"])}

`conversation_style_full` から `conversation_style_core3/core2` にすると、BASiSのlossがtieへ移るpromptが増えます。これは、`engagingness` がBASiS不利の勝敗を作っていたことを示します。ただし、除外後にBASiSが一貫して大きく勝つわけではないため、主張は「評価軸の重複を整理すると、conversation_styleでのBASiS不利は大きく弱まる」に留めるべきです。

## 推奨する評価軸セット

- `tst`: `style_strength`, `content_preservation`, `naturalness` を維持する。BASiSの強みを最も直接的に測れており、品質制約も含むため妥当です。
- `conversation_style`: 目的スタイル評価としては `style_consistency`, `style_similarity` を中核にし、必要に応じて `fluency` を補助軸として残す。`engagingness` はUSR系の一般対話品質へ移す候補です。
- `strategy_transition`: 目的戦略評価としては `strategy_appropriateness_score` を中核にし、`transition_smoothness_score` は会話進行品質として補助的に扱うか、USR系に近い軸として分離する候補です。ただし今回の結果では、この軸はBASiSを下げていません。
- `usr_quality`: `understandable`, `natural`, `maintains_context`, `interesting_or_engaging`, `overall_quality` を一般対話品質カテゴリとして維持する。BASiSの弱点分析では特に `interesting_or_engaging` と `overall_quality` を重視します。

## 注意点

この分析は、都合の悪い指標を消して主結果を良くするためのものではありません。元の4カテゴリ評価結果はそのまま残し、追加分析として評価軸の影響を調べています。`engagingness` や `transition_smoothness_score` を除外・移動する場合は、BASiSを有利にするためではなく、USR系の一般対話品質評価と重複する軸を分離するため、という研究上の理由を明記する必要があります。

詳細な数値は同じディレクトリのCSVを参照してください。

- `axis_level_summary.csv`
- `axis_level_pairwise.csv`
- `core_reaggregation_summary.csv`
- `core_reaggregation_friedman.csv`
- `core_reaggregation_posthoc.csv`
- `axis_drag_analysis.csv`
- `winner_flip_analysis.csv`
"""
    report_path = output_dir / "axis_ablation_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.root / "axis_ablation_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_category_scores(args.root)
    rng = random.Random(args.seed)

    axis_summary = build_axis_level_summary(
        data, n_bootstrap=args.n_bootstrap, rng=rng
    )
    axis_pairwise = build_axis_level_pairwise(
        data,
        tie_threshold=args.tie_threshold,
        n_permutations=args.n_bootstrap,
        rng=rng,
    )
    core_summary = build_core_summary(data, n_bootstrap=args.n_bootstrap, rng=rng)
    core_friedman, core_posthoc = build_core_friedman_and_posthoc(
        data,
        tie_threshold=args.tie_threshold,
        n_permutations=args.n_bootstrap,
        rng=rng,
    )
    axis_drag = build_axis_drag_analysis(data)
    winner_flip = build_winner_flip_analysis(data, tie_threshold=args.tie_threshold)
    strategy_existing_metrics = read_strategy_existing_metrics(args.root)

    write_csv(
        output_dir / "axis_level_summary.csv",
        axis_summary,
        ["category", "axis", "model_name", "n", "mean", "std", "ci95_low", "ci95_high"],
    )
    write_csv(
        output_dir / "axis_level_pairwise.csv",
        axis_pairwise,
        [
            "category",
            "axis",
            "comparison",
            "n",
            "mean_diff",
            "median_diff",
            "wins",
            "ties",
            "losses",
            "p_raw",
            "p_holm",
            "effect_size",
            "significant",
        ],
    )
    write_csv(
        output_dir / "core_reaggregation_summary.csv",
        core_summary,
        ["score_set", "model_name", "n", "mean", "std", "ci95_low", "ci95_high"],
    )
    write_csv(
        output_dir / "core_reaggregation_friedman.csv",
        core_friedman,
        ["score_set", "n", "friedman_chi2", "p_value", "kendalls_w", "significant"],
    )
    write_csv(
        output_dir / "core_reaggregation_posthoc.csv",
        core_posthoc,
        [
            "score_set",
            "comparison",
            "n",
            "mean_diff",
            "median_diff",
            "p_raw",
            "p_holm",
            "effect_size",
            "wins",
            "ties",
            "losses",
            "significant",
        ],
    )
    write_csv(
        output_dir / "axis_drag_analysis.csv",
        axis_drag,
        [
            "category",
            "axis",
            "comparison",
            "axis_mean_diff",
            "contribution_to_overall_diff",
            "direction",
        ],
    )
    write_csv(
        output_dir / "winner_flip_analysis.csv",
        winner_flip,
        [
            "score_set_before",
            "score_set_after",
            "comparison",
            "n",
            "before_basis_wins",
            "before_ties",
            "before_basis_losses",
            "after_basis_wins",
            "after_ties",
            "after_basis_losses",
            "losses_to_wins",
            "losses_to_ties",
            "ties_to_wins",
        ],
    )
    write_csv(
        output_dir / "strategy_transition_existing_metrics.csv",
        strategy_existing_metrics,
        [
            "model_name",
            "count",
            "mean_strategy_appropriateness",
            "mean_transition_smoothness",
            "strategy_accuracy",
            "strategy_macro_f1",
            "strategy_weighted_f1",
            "strategy_jsd_to_esconv",
            "strategy_tvd_to_esconv",
            "strategy_entropy",
            "most_frequent_strategy",
            "most_frequent_strategy_ratio",
            "transition_jsd_to_esconv",
            "transition_tvd_to_esconv",
            "transition_entropy",
        ],
    )
    build_report(
        output_dir,
        axis_summary,
        axis_pairwise,
        core_summary,
        core_friedman,
        core_posthoc,
        axis_drag,
        winner_flip,
        strategy_existing_metrics,
    )

    metadata = {
        "root": str(args.root),
        "output_dir": str(output_dir),
        "tie_threshold": args.tie_threshold,
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_bootstrap,
        "seed": args.seed,
        "model_mapping": {
            "Base": "base",
            "BASiS/Bayes-DPO": "bayes_dpo",
            "Random-DPO": "random_dpo",
        },
        "note": "既存10段階Oracle評価raw.jsonlに対する診断的な評価軸再集計。",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote analysis files to {output_dir}")


if __name__ == "__main__":
    main()
