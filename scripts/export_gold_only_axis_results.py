#!/usr/bin/env python3
"""Gold-only比較の各評価軸スコアをテキストとJSONへまとめる。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODELS = ("base", "gold_only", "basis", "random_dpo")
MODEL_LABELS = {
    "base": "Base",
    "gold_only": "Gold-only DPO",
    "basis": "BASiS-DPO",
    "random_dpo": "Random-DPO",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_scores(path: Path, *, prefix: str = "") -> list[dict[str, Any]]:
    filename = f"{prefix}model_summary.csv"
    rows = read_csv(path / filename)
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["axis"], {})[row["model_name"]] = row
    scores: list[dict[str, Any]] = []
    for axis_key in sorted(grouped):
        model_rows = grouped[axis_key]
        missing = set(MODELS) - set(model_rows)
        if missing:
            raise ValueError(f"{path}/{axis_key}: モデル不足 {sorted(missing)}")
        category, _, axis = axis_key.partition(".")
        scores.append(
            {
                "category": category,
                "axis": axis or category,
                "axis_key": axis_key,
                "n": int(model_rows["base"]["n"]),
                "scores": {
                    MODEL_LABELS[model]: round(float(model_rows[model]["mean"]), 3)
                    for model in MODELS
                },
            }
        )
    return scores


def write_scores(
    *, dataset: str, evaluation_set: str, scores: list[dict[str, Any]], output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{dataset} - {evaluation_set}", ""]
    current_category = None
    for row in scores:
        if row["category"] != current_category:
            current_category = row["category"]
            lines.extend([f"[{current_category}]", ""])
        lines.append(f"{row['axis']} (n={row['n']})")
        for model in MODELS:
            label = MODEL_LABELS[model]
            lines.append(f"  {label}: {row['scores'][label]:.3f}")
        lines.append("")
    output.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "evaluation_set": evaluation_set,
                "models": [MODEL_LABELS[model] for model in MODELS],
                "axes": scores,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Gold-only軸別4モデルスコア出力")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/gold_only_dpo/runs/gold_only_dpo500_v1"),
    )
    args = parser.parse_args()
    combined: list[dict[str, Any]] = []
    for dataset in ("esconv", "mathdial", "meditod"):
        dataset_root = args.root / dataset
        specifications = [("main", dataset_root / "statistics", "")]
        if dataset == "meditod":
            specifications.extend(
                [
                    ("consultation_cluster", dataset_root / "statistics", "cluster_"),
                    ("ood", dataset_root / "statistics_ood", ""),
                    (
                        "ood_consultation_cluster",
                        dataset_root / "statistics_ood",
                        "cluster_",
                    ),
                ]
            )
        for evaluation_set, statistics_dir, prefix in specifications:
            scores = load_scores(statistics_dir, prefix=prefix)
            write_scores(
                dataset=dataset,
                evaluation_set=evaluation_set,
                scores=scores,
                output=dataset_root / "reports" / f"axis_scores_{evaluation_set}",
            )
            if evaluation_set == "main":
                combined.append({"dataset": dataset, "axes": scores})

    combined_txt: list[str] = []
    for item in combined:
        combined_txt.extend([item["dataset"], ""])
        for row in item["axes"]:
            combined_txt.append(f"{row['axis_key']} (n={row['n']})")
            combined_txt.extend(
                f"  {label}: {score:.3f}" for label, score in row["scores"].items()
            )
            combined_txt.append("")
    (args.root / "all_datasets_axis_scores.txt").write_text(
        "\n".join(combined_txt), encoding="utf-8"
    )
    (args.root / "all_datasets_axis_scores.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
