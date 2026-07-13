#!/usr/bin/env python3
"""MathDial 3モデルOracle評価の軸別対応あり検定。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_oracle_three_model_significance import friedman_test, holm_adjust, paired_permutation_p


MODELS = ("base", "basis", "random_dpo")
ALIASES = {"bayes_dpo": "basis", "BASiS": "basis", "random": "random_dpo", "Base": "base"}
PAIRS = (("BASiS_vs_Base", "basis", "base"), ("BASiS_vs_Random-DPO", "basis", "random_dpo"), ("Base_vs_Random-DPO", "base", "random_dpo"))


def normalize_model(value: str) -> str:
    return ALIASES.get(value, value)


def load_axis_scores(paths: list[Path]) -> dict[str, dict[str, dict[str, float]]]:
    """category/axis/sample/modelの対応ありスコアを読む。"""
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        category = path.parent.name
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            sample = str(row["sample_id"])
            model = normalize_model(str(row["model_name"]))
            if model not in MODELS:
                continue
            scores = dict(row.get("scores", {}))
            scores["category_overall"] = float(row["overall_score"])
            for axis, value in scores.items():
                result[f"{category}.{axis}"][sample][model] = float(value)
    return result


def bootstrap_mean_ci(values: list[float], *, rng: random.Random, draws: int) -> tuple[float, float]:
    if len(values) < 2:
        value = values[0] if values else 0.0
        return value, value
    samples = sorted(mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(draws))
    return samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]


def rank_biserial(diffs: list[float]) -> float:
    nonzero = [value for value in diffs if value != 0]
    if not nonzero:
        return 0.0
    ranked = sorted(enumerate(nonzero), key=lambda item: abs(item[1]))
    rank_by_index = {index: rank + 1 for rank, (index, _) in enumerate(ranked)}
    positive = sum(rank_by_index[index] for index, value in enumerate(nonzero) if value > 0)
    negative = sum(rank_by_index[index] for index, value in enumerate(nonzero) if value < 0)
    return (positive - negative) / (positive + negative)


def analyze(data: dict[str, dict[str, dict[str, float]]], *, permutations: int, bootstrap: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    summaries, omnibus, posthoc = [], [], []
    for axis in sorted(data):
        complete = {sample: scores for sample, scores in data[axis].items() if all(model in scores for model in MODELS)}
        if not complete:
            continue
        sample_ids = sorted(complete)
        values = {model: [complete[sample][model] for sample in sample_ids] for model in MODELS}
        model_means = {model: mean(values[model]) for model in MODELS}
        highest = max(model_means, key=model_means.get)
        for model in MODELS:
            low, high = bootstrap_mean_ci(values[model], rng=rng, draws=bootstrap)
            summaries.append({"axis": axis, "model_name": model, "n": len(sample_ids), "mean": model_means[model], "std": stdev(values[model]) if len(sample_ids) > 1 else 0.0, "ci95_low": low, "ci95_high": high, "is_highest": str(model == highest).lower()})
        chi2, p_value, kendall = friedman_test({"base": values["base"], "bayes_dpo": values["basis"], "random_dpo": values["random_dpo"]})
        significant = p_value < 0.05
        omnibus.append({"axis": axis, "n": len(sample_ids), "friedman_chi2": chi2, "p_value": p_value, "kendalls_w": kendall, "significant": str(significant).lower(), "basis_highest": str(highest == "basis").lower()})
        if not significant:
            continue
        pending, raw_ps = [], []
        for name, left, right in PAIRS:
            diffs = [values[left][index] - values[right][index] for index in range(len(sample_ids))]
            p_raw = paired_permutation_p(diffs, n_permutations=permutations, rng=rng)
            low, high = bootstrap_mean_ci(diffs, rng=rng, draws=bootstrap)
            sd = stdev(diffs) if len(diffs) > 1 else 0.0
            pending.append({"axis": axis, "comparison": name, "n": len(diffs), "mean_diff": mean(diffs), "median_diff": median(diffs), "ci95_low": low, "ci95_high": high, "p_raw": p_raw, "p_holm": None, "cohens_dz": mean(diffs) / sd if sd else 0.0, "rank_biserial": rank_biserial(diffs), "wins": sum(value > 0 for value in diffs), "ties": sum(value == 0 for value in diffs), "losses": sum(value < 0 for value in diffs), "significant": None})
            raw_ps.append(p_raw)
        for row, adjusted in zip(pending, holm_adjust(raw_ps)):
            row["p_holm"] = adjusted
            row["significant"] = str(adjusted < 0.05).lower()
            posthoc.append(row)
    return summaries, omnibus, posthoc


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as file:
        if fields:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="MathDial軸別有意差検定")
    parser.add_argument("--raw", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary, omnibus, posthoc = analyze(load_axis_scores(args.raw), permutations=args.permutations, bootstrap=args.bootstrap, seed=args.seed)
    write_csv(args.output_dir / "model_summary.csv", summary)
    write_csv(args.output_dir / "omnibus_friedman.csv", omnibus)
    write_csv(args.output_dir / "posthoc_pairwise.csv", posthoc)
    (args.output_dir / "metadata.json").write_text(json.dumps({"raw": [str(path) for path in args.raw], "permutations": args.permutations, "bootstrap": args.bootstrap, "seed": args.seed}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
