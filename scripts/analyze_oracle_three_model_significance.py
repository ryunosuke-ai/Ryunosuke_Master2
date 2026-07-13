#!/usr/bin/env python3
"""3モデルOracle評価の対応あり有意差検定を行う。"""

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
ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle評価raw.jsonlから3モデル対応あり検定を行う。")
    parser.add_argument("--root", type=Path, required=True, help="評価runのrootディレクトリ。")
    parser.add_argument("--output_dir", type=Path, default=None, help="検定結果の出力先。未指定ならroot。")
    parser.add_argument(
        "--category",
        action="append",
        required=True,
        help="category_name=relative_output_dir の形式。例: conversation_style_v2=oracle_conversation_style_esconv_v2_10pt",
    )
    parser.add_argument("--tie_threshold", type=float, default=0.25)
    parser.add_argument("--n_permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_categories(values: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--category は name=dir 形式で指定してください: {value}")
        name, directory = value.split("=", 1)
        name = name.strip()
        directory = directory.strip()
        if not name or not directory:
            raise ValueError(f"--category は name=dir 形式で指定してください: {value}")
        parsed.append((name, directory))
    return parsed


def record_score(record: dict[str, Any]) -> float:
    if record.get("overall_score") not in (None, ""):
        return float(record["overall_score"])
    scores = record.get("scores")
    if not isinstance(scores, dict) or not scores:
        raise ValueError(f"スコアがありません: {record.get('sample_id')}")
    return mean([float(value) for value in scores.values()])


def load_wide_scores(raw_path: Path) -> dict[str, dict[str, float]]:
    by_sample: dict[str, dict[str, float]] = defaultdict(dict)
    with raw_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id") or record.get("prompt_id") or "").strip()
            model_name = str(record.get("model_name") or "").strip()
            if not sample_id or model_name not in MODEL_KEYS:
                raise ValueError(f"{raw_path}:{line_number} sample_id/model_nameが不正です。")
            if model_name in by_sample[sample_id]:
                raise ValueError(f"{raw_path}:{line_number} {sample_id}/{model_name} が重複しています。")
            by_sample[sample_id][model_name] = record_score(record)
    incomplete = [
        sample_id for sample_id, scores in by_sample.items() if not all(model in scores for model in MODEL_KEYS)
    ]
    if incomplete:
        raise ValueError(f"{raw_path}: 3モデルが揃っていないsample_idがあります: {incomplete[:10]}")
    return dict(by_sample)


def average_ranks(values: list[float]) -> tuple[list[float], list[int]]:
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
    p_value = math.exp(-max(0.0, chi2) / 2.0)
    kendalls_w = chi2 / (n * (k - 1))
    return chi2, p_value, kendalls_w


def paired_permutation_p(diffs: list[float], *, n_permutations: int, rng: random.Random) -> float:
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
    previous = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, p_values[idx] * (len(p_values) - rank))
        value = max(previous, value)
        adjusted[idx] = value
        previous = value
    return adjusted


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def effect_size(diffs: list[float]) -> float:
    sd = sample_std(diffs)
    return mean(diffs) / sd if sd else 0.0


def win_tie_loss(diffs: list[float], threshold: float) -> tuple[int, int, int]:
    wins = sum(1 for diff in diffs if diff >= threshold)
    losses = sum(1 for diff in diffs if diff <= -threshold)
    ties = len(diffs) - wins - losses
    return wins, ties, losses


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    categories = parse_categories(args.category)
    output_dir = args.output_dir or args.root
    rng = random.Random(args.seed)

    omnibus_rows: list[dict[str, Any]] = []
    posthoc_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for category, directory in categories:
        raw_path = args.root / directory / "raw.jsonl"
        if not raw_path.exists():
            raise FileNotFoundError(f"raw.jsonlが見つかりません: {raw_path}")
        by_sample = load_wide_scores(raw_path)
        sample_ids = sorted(by_sample)
        values_by_model = {
            model: [by_sample[sample_id][model] for sample_id in sample_ids]
            for model in MODEL_KEYS
        }

        for model in MODEL_KEYS:
            values = values_by_model[model]
            summary_rows.append(
                {
                    "category": category,
                    "model_name": MODEL_LABELS[model],
                    "n": len(values),
                    "mean": mean(values),
                    "std": sample_std(values),
                }
            )

        chi2, p_value, kendalls_w = friedman_test(values_by_model)
        significant = p_value < ALPHA
        omnibus_rows.append(
            {
                "category": category,
                "n": len(sample_ids),
                "friedman_chi2": chi2,
                "p_value": p_value,
                "kendalls_w": kendalls_w,
                "significant": str(significant).lower(),
            }
        )
        if not significant:
            continue

        raw_p_values: list[float] = []
        rows_for_category: list[dict[str, Any]] = []
        for comparison, left, right in PAIRWISE:
            diffs = [
                values_by_model[left][idx] - values_by_model[right][idx]
                for idx in range(len(sample_ids))
            ]
            wins, ties, losses = win_tie_loss(diffs, args.tie_threshold)
            p_raw = paired_permutation_p(diffs, n_permutations=args.n_permutations, rng=rng)
            raw_p_values.append(p_raw)
            rows_for_category.append(
                {
                    "category": category,
                    "comparison": comparison,
                    "n": len(diffs),
                    "mean_diff": mean(diffs),
                    "median_diff": median(diffs),
                    "p_raw": p_raw,
                    "p_holm": None,
                    "effect_size": effect_size(diffs),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "significant": None,
                }
            )
        for row, p_holm in zip(rows_for_category, holm_adjust(raw_p_values)):
            row["p_holm"] = p_holm
            row["significant"] = str(p_holm < ALPHA).lower()
            posthoc_rows.append(row)

    write_csv(
        output_dir / "summary_three_model_overall.csv",
        summary_rows,
        ["category", "model_name", "n", "mean", "std"],
    )
    write_csv(
        output_dir / "omnibus_friedman.csv",
        omnibus_rows,
        ["category", "n", "friedman_chi2", "p_value", "kendalls_w", "significant"],
    )
    write_csv(
        output_dir / "posthoc_pairwise.csv",
        posthoc_rows,
        [
            "category",
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
    print(f"3モデル有意差検定を書き出しました: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
