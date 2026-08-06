#!/usr/bin/env python3
"""Base/BASiS/Random/Gold-onlyの対応あり4モデル統計。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from scripts.analyze_oracle_three_model_significance import (
    effect_size,
    holm_adjust,
    paired_permutation_p,
)
from tools.gold_only_dpo import FOUR_MODELS, normalize_model


PAIRS = (
    ("BASiS_vs_Base", "basis", "base"),
    ("BASiS_vs_Random-DPO", "basis", "random_dpo"),
    ("BASiS_vs_Gold-only", "basis", "gold_only"),
    ("Gold-only_vs_Base", "gold_only", "base"),
    ("Gold-only_vs_Random-DPO", "gold_only", "random_dpo"),
    ("Base_vs_Random-DPO", "base", "random_dpo"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_raw_specs(values: list[str]) -> list[tuple[str, Path]]:
    output = []
    for value in values:
        if "=" not in value:
            raise ValueError("--rawはcategory=path形式で指定してください。")
        category, path = value.split("=", 1)
        if not category.strip() or not path.strip():
            raise ValueError("--rawのcategory/pathは空にできません。")
        output.append((category.strip(), Path(path)))
    return output


def load_axis_scores(
    specs: list[tuple[str, Path]],
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for category, path in specs:
        for row in read_jsonl(path):
            sample = str(row["sample_id"])
            model = normalize_model(str(row["model_name"]))
            if model not in FOUR_MODELS:
                raise ValueError(f"未知のモデル名です: {model}")
            scores = {key: float(value) for key, value in row.get("scores", {}).items()}
            scores["category_overall"] = float(row["overall_score"])
            for axis, value in scores.items():
                key = f"{category}.{axis}"
                if model in result[key][sample]:
                    raise ValueError(f"重複スコアです: {key}/{sample}/{model}")
                result[key][sample][model] = value
    return result


def average_ranks(values: list[float]) -> tuple[list[float], list[int]]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    ties: list[int] = []
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = rank
        if end - start > 1:
            ties.append(end - start)
        start = end
    return ranks, ties


def regularized_gamma_q(shape: float, value: float) -> float:
    """正規化上側不完全ガンマQ(a,x)を標準ライブラリだけで計算する。"""

    if shape <= 0 or value < 0:
        raise ValueError("gamma引数が不正です。")
    if value == 0:
        return 1.0
    epsilon = 3e-14
    tiny = 1e-300
    max_iterations = 1000
    log_term = -value + shape * math.log(value) - math.lgamma(shape)
    if value < shape + 1.0:
        total = term = 1.0 / shape
        ap = shape
        for _ in range(max_iterations):
            ap += 1.0
            term *= value / ap
            total += term
            if abs(term) < abs(total) * epsilon:
                return max(0.0, min(1.0, 1.0 - total * math.exp(log_term)))
        raise RuntimeError("gamma seriesが収束しませんでした。")
    b = value + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    result = d
    for index in range(1, max_iterations + 1):
        coefficient = -index * (index - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return max(0.0, min(1.0, result * math.exp(log_term)))
    raise RuntimeError("gamma continued fractionが収束しませんでした。")


def friedman_test(values: dict[str, list[float]]) -> tuple[float, float, float]:
    models = list(FOUR_MODELS)
    lengths = {len(values[model]) for model in models}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
        raise ValueError("Friedman検定には長さが等しい2標本以上が必要です。")
    sample_count = next(iter(lengths))
    model_count = len(models)
    rank_sums = [0.0] * model_count
    tie_terms = 0
    for index in range(sample_count):
        ranks, ties = average_ranks([values[model][index] for model in models])
        for model_index, rank in enumerate(ranks):
            rank_sums[model_index] += rank
        tie_terms += sum(size**3 - size for size in ties)
    chi2 = (
        12.0
        / (sample_count * model_count * (model_count + 1))
        * sum(rank**2 for rank in rank_sums)
        - 3 * sample_count * (model_count + 1)
    )
    correction = 1.0 - tie_terms / (
        sample_count * (model_count**3 - model_count)
    )
    if correction <= 0:
        chi2 = 0.0
    else:
        chi2 /= correction
    chi2 = max(0.0, chi2)
    p_value = regularized_gamma_q((model_count - 1) / 2.0, chi2 / 2.0)
    kendalls_w = chi2 / (sample_count * (model_count - 1))
    return chi2, p_value, kendalls_w


def bootstrap_ci(
    values: list[float], *, rng: random.Random, draws: int
) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    sampled = sorted(
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(draws)
    )
    return sampled[int(0.025 * (draws - 1))], sampled[int(0.975 * (draws - 1))]


def rank_biserial(differences: list[float]) -> float:
    nonzero = [value for value in differences if value]
    if not nonzero:
        return 0.0
    ranks, _ = average_ranks([abs(value) for value in nonzero])
    positive = sum(ranks[index] for index, value in enumerate(nonzero) if value > 0)
    negative = sum(ranks[index] for index, value in enumerate(nonzero) if value < 0)
    return (positive - negative) / (positive + negative)


def analyze(
    data: dict[str, dict[str, dict[str, float]]],
    *,
    permutations: int,
    bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    summaries: list[dict[str, Any]] = []
    omnibus: list[dict[str, Any]] = []
    posthoc: list[dict[str, Any]] = []
    for axis in sorted(data):
        incomplete = {
            sample: sorted(set(FOUR_MODELS) - set(scores))
            for sample, scores in data[axis].items()
            if set(scores) != set(FOUR_MODELS)
        }
        if incomplete:
            raise ValueError(
                f"4モデルが揃わない評価sampleがあります: {axis} "
                f"{dict(list(incomplete.items())[:10])}"
            )
        sample_ids = sorted(data[axis])
        if len(sample_ids) < 2:
            raise ValueError(f"統計対象sampleが不足しています: {axis}")
        values = {
            model: [data[axis][sample][model] for sample in sample_ids]
            for model in FOUR_MODELS
        }
        model_means = {model: mean(values[model]) for model in FOUR_MODELS}
        highest = max(model_means, key=model_means.get)
        for model in FOUR_MODELS:
            low, high = bootstrap_ci(values[model], rng=rng, draws=bootstrap)
            summaries.append(
                {
                    "axis": axis,
                    "model_name": model,
                    "n": len(sample_ids),
                    "mean": model_means[model],
                    "std": stdev(values[model]),
                    "ci95_low": low,
                    "ci95_high": high,
                    "is_highest": model == highest,
                }
            )
        chi2, p_value, kendalls_w = friedman_test(values)
        significant = p_value < 0.05
        omnibus.append(
            {
                "axis": axis,
                "n": len(sample_ids),
                "models": 4,
                "friedman_chi2": chi2,
                "degrees_of_freedom": 3,
                "p_value": p_value,
                "kendalls_w": kendalls_w,
                "significant": significant,
                "highest_model": highest,
            }
        )
        if not significant:
            continue
        pending: list[dict[str, Any]] = []
        raw_p_values: list[float] = []
        for comparison, left, right in PAIRS:
            differences = [
                data[axis][sample][left] - data[axis][sample][right]
                for sample in sample_ids
            ]
            low, high = bootstrap_ci(differences, rng=rng, draws=bootstrap)
            p_raw = paired_permutation_p(
                differences,
                n_permutations=permutations,
                rng=rng,
            )
            raw_p_values.append(p_raw)
            pending.append(
                {
                    "axis": axis,
                    "comparison": comparison,
                    "n": len(differences),
                    "mean_diff": mean(differences),
                    "median_diff": median(differences),
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_raw": p_raw,
                    "p_holm": None,
                    "cohens_dz": effect_size(differences),
                    "rank_biserial": rank_biserial(differences),
                    "wins": sum(value > 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                    "losses": sum(value < 0 for value in differences),
                    "left_win_rate": sum(value > 0 for value in differences) / len(differences),
                    "tie_rate": sum(value == 0 for value in differences) / len(differences),
                    "right_win_rate": sum(value < 0 for value in differences) / len(differences),
                    "significant": None,
                }
            )
        for row, adjusted in zip(pending, holm_adjust(raw_p_values)):
            row["p_holm"] = adjusted
            row["significant"] = adjusted < 0.05
            posthoc.append(row)
    return summaries, omnibus, posthoc


def consultation_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in read_jsonl(path):
        sample = str(row["sample_id"])
        conversation = str((row.get("metadata") or {}).get("conversation_id") or "")
        if not conversation:
            raise ValueError(f"診療IDがありません: {sample}")
        if sample in result and result[sample] != conversation:
            raise ValueError(f"sampleの診療IDが不一致です: {sample}")
        result[sample] = conversation
    return result


def aggregate_clusters(
    data: dict[str, dict[str, dict[str, float]]], mapping: dict[str, str]
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for axis, samples in data.items():
        buckets: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for sample, scores in samples.items():
            if sample not in mapping:
                raise ValueError(f"cluster mapにsampleがありません: {sample}")
            for model, value in scores.items():
                buckets[mapping[sample]][model].append(value)
        for cluster, model_values in buckets.items():
            result[axis][cluster] = {
                model: mean(values) for model, values in model_values.items()
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Gold-only 4モデル対応あり統計")
    parser.add_argument("--raw", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cluster-map", type=Path)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-status", default="confirmatory_fixed_prompt_reuse")
    args = parser.parse_args()
    specs = parse_raw_specs(args.raw)
    data = load_axis_scores(specs)
    summary, omnibus, posthoc = analyze(
        data,
        permutations=args.permutations,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    write_csv(args.output_dir / "model_summary.csv", summary)
    write_csv(args.output_dir / "omnibus_friedman.csv", omnibus)
    write_csv(args.output_dir / "posthoc_pairwise.csv", posthoc)
    if args.cluster_map:
        cluster_data = aggregate_clusters(data, consultation_map(args.cluster_map))
        cluster_summary, cluster_omnibus, cluster_posthoc = analyze(
            cluster_data,
            permutations=args.permutations,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
        write_csv(args.output_dir / "cluster_model_summary.csv", cluster_summary)
        write_csv(args.output_dir / "cluster_omnibus_friedman.csv", cluster_omnibus)
        write_csv(args.output_dir / "cluster_posthoc_pairwise.csv", cluster_posthoc)
    metadata = {
        "models": list(FOUR_MODELS),
        "raw": [{"category": category, "path": path.as_posix()} for category, path in specs],
        "permutations": args.permutations,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "inference_status": args.inference_status,
        "posthoc_policy": "only_after_significant_friedman_holm_within_axis_six_pairs",
        "existing_three_model_scores_reused": True,
        "oracle_temporal_limitation": "Gold-only was judged later than the existing three models.",
        "cluster_map": args.cluster_map.as_posix() if args.cluster_map else None,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
