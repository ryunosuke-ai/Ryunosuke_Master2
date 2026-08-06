"""Gold-only DPO 4モデル比較の監査可能なMarkdownレポートを作る。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_statistics(title: str, directory: Path) -> list[str]:
    summary = read_csv(directory / "model_summary.csv")
    omnibus = read_csv(directory / "omnibus_friedman.csv")
    posthoc = read_csv(directory / "posthoc_pairwise.csv")
    lines = [f"## {title}", "", "| Axis | Base | Gold-only | BASiS | Random | Friedman p | Kendall's W |", "|---|---:|---:|---:|---:|---:|---:|"]
    means: dict[str, dict[str, float]] = {}
    for row in summary:
        means.setdefault(row["axis"], {})[row["model_name"]] = float(row["mean"])
    omnibus_by_axis = {row["axis"]: row for row in omnibus}
    for axis in sorted(means):
        values = means[axis]
        test = omnibus_by_axis[axis]
        lines.append(
            f"| {axis} | {values['base']:.3f} | {values['gold_only']:.3f} | "
            f"{values['basis']:.3f} | {values['random_dpo']:.3f} | "
            f"{float(test['p_value']):.4g} | {float(test['kendalls_w']):.3f} |"
        )
    lines.extend(["", "### Significant post-hoc comparisons", ""])
    significant = [row for row in posthoc if row.get("significant", "").lower() == "true"]
    if not significant:
        lines.append("Holm補正後に有意な比較はありません。")
    else:
        lines.extend(["| Axis | Comparison | Mean difference | Holm p |", "|---|---|---:|---:|"])
        for row in significant:
            lines.append(
                f"| {row['axis']} | {row['comparison']} | {float(row['mean_diff']):.3f} | "
                f"{float(row['p_holm']):.4g} |"
            )
    lines.append("")
    return lines


def build_report(dataset: str, root: Path, output: Path) -> None:
    manifest = read_json(root / "data" / "gold_only_manifest.json")
    generation = read_json(root / "evaluation" / "generation_manifest.json")
    lines = [
        f"# {dataset} Gold-only DPO 500 Comparison",
        "",
        "## Experimental condition",
        "",
        "- Baseline: Gold-only DPO (not SFT)",
        f"- Training records: {manifest['audit']['records']}",
        f"- Estimated optimizer steps: {manifest['estimated_optimizer_steps']}",
        f"- Source SHA-256: `{manifest['source_sha256']}`",
        f"- Evaluation prompt-set SHA-256: `{generation['model_prompt_set_sha256']}`",
        f"- Evaluation source SHA-256: `{generation['evaluation_source_sha256']}`",
        "- Existing Base/BASiS/Random responses and Oracle judgments were reused read-only.",
        "- Only Gold-only responses and judgments were newly generated.",
        "- Oracle timing limitation: the Gold-only judgments were produced later than the existing three-model judgments.",
        "",
    ]
    lines.extend(render_statistics("Main evaluation", root / "statistics"))
    if (root / "statistics_ood" / "model_summary.csv").is_file():
        lines.extend(render_statistics("OOD secondary evaluation", root / "statistics_ood"))
    if dataset == "mathdial":
        lines.extend(
            [
                "## Interpretation constraint",
                "",
                "The outcome-selected 100-prompt MathDial set is exploratory and is not treated as a confirmatory result.",
                "",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gold-only DPO比較レポート")
    parser.add_argument("--dataset", choices=("esconv", "mathdial", "meditod"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.dataset, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
