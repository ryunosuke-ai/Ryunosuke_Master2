#!/usr/bin/env python3
"""MediTOD prompt対応統計と診療単位cluster感度分析。"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from scripts.run_mathdial_statistics import analyze, load_axis_scores, write_csv


def consultation_map(oracle_input: Path) -> dict[str, str]:
    result = {}
    for line in oracle_input.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        conversation_id = str(row.get("metadata", {}).get("conversation_id", ""))
        if not conversation_id:
            raise ValueError(f"Oracle inputにconversation_idがありません: {sample_id}")
        if sample_id in result and result[sample_id] != conversation_id:
            raise ValueError(f"sample_idの診療IDが不一致です: {sample_id}")
        result[sample_id] = conversation_id
    return result


def aggregate_by_consultation(
    data: dict[str, dict[str, dict[str, float]]], mapping: dict[str, str]
) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for axis, samples in data.items():
        buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for sample_id, scores in samples.items():
            if sample_id not in mapping:
                continue
            for model, value in scores.items():
                buckets[mapping[sample_id]][model].append(float(value))
        for consultation_id, model_values in buckets.items():
            output[axis][consultation_id] = {
                model: mean(values) for model, values in model_values.items()
            }
    return output


def bootstrap_consultation_difference(
    data: dict[str, dict[str, dict[str, float]]], *, draws: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for axis, consultations in sorted(data.items()):
        complete = [scores for scores in consultations.values() if all(name in scores for name in ("base", "basis", "random_dpo"))]
        if not complete:
            continue
        for comparison, left, right in (
            ("BASiS_vs_Base", "basis", "base"),
            ("BASiS_vs_Random-DPO", "basis", "random_dpo"),
            ("Base_vs_Random-DPO", "base", "random_dpo"),
        ):
            differences = [scores[left] - scores[right] for scores in complete]
            samples = sorted(mean(differences[rng.randrange(len(differences))] for _ in differences) for _ in range(draws))
            rows.append(
                {
                    "axis": axis,
                    "comparison": comparison,
                    "consultations": len(differences),
                    "mean_diff": mean(differences),
                    "ci95_low": samples[int(0.025 * (draws - 1))],
                    "ci95_high": samples[int(0.975 * (draws - 1))],
                    "analysis_unit": "consultation",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="MediTOD有意差検定")
    parser.add_argument("--raw", action="append", type=Path, required=True)
    parser.add_argument("--oracle-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prompt_data = load_axis_scores(args.raw)
    summary, omnibus, posthoc = analyze(
        prompt_data,
        permutations=args.permutations,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    write_csv(args.output_dir / "model_summary.csv", summary)
    write_csv(args.output_dir / "omnibus_friedman.csv", omnibus)
    write_csv(args.output_dir / "posthoc_pairwise.csv", posthoc)
    clustered = aggregate_by_consultation(prompt_data, consultation_map(args.oracle_input))
    cluster_summary, cluster_omnibus, cluster_posthoc = analyze(
        clustered,
        permutations=args.permutations,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    write_csv(args.output_dir / "cluster_model_summary.csv", cluster_summary)
    write_csv(args.output_dir / "cluster_omnibus_friedman.csv", cluster_omnibus)
    write_csv(args.output_dir / "cluster_posthoc_pairwise.csv", cluster_posthoc)
    write_csv(
        args.output_dir / "cluster_bootstrap_pairwise.csv",
        bootstrap_consultation_difference(clustered, draws=args.bootstrap, seed=args.seed),
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "raw": [str(path) for path in args.raw],
                "oracle_input": str(args.oracle_input),
                "permutations": args.permutations,
                "bootstrap": args.bootstrap,
                "seed": args.seed,
                "primary_unit": "prompt",
                "sensitivity_unit": "consultation",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
