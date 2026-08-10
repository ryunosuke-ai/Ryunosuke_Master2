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
COMPARISONS = (
    "BASiS_vs_Base",
    "BASiS_vs_Gold-only",
    "BASiS_vs_Random-DPO",
    "Gold-only_vs_Base",
    "Gold-only_vs_Random-DPO",
    "Base_vs_Random-DPO",
)

REPRESENTATIVE_AXES = {
    "esconv": (
        "text_style_transfer.style_strength",
        "conversation_style.esconv_tone_similarity",
        "conversation_style.supporter_role_consistency",
        "conversation_style.non_directive_support_style",
        "strategy_transition.strategy_stage_alignment",
        "strategy_transition.premature_advice_avoidance",
        "text_style_transfer.naturalness",
    ),
    "mathdial": (
        "pedagogical_v2.equitable_tutoring",
        "pedagogical_v2.learner_reasoning_diagnosis",
        "pedagogical_v2.mistake_location_and_targeting",
        "pedagogical_v2.guidance_quality",
        "pedagogical_v2.feedback_actionability",
        "pedagogical_v2.answer_revealing_calibration",
        "pedagogical_v2.teacher_move_stage_alignment",
    ),
    "meditod": (
        "general.response_relevance",
        "general.overall_quality",
        "history.premature_assessment_avoidance",
        "safety.appropriate_uncertainty",
        "general.understandable",
        "safety.unsafe_medical_advice",
        "safety.unsupported_diagnosis",
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def load_scores(path: Path, *, prefix: str = "") -> list[dict[str, Any]]:
    filename = f"{prefix}model_summary.csv"
    rows = read_csv(path / filename)
    omnibus = {
        row["axis"]: row
        for row in read_csv(path / f"{prefix}omnibus_friedman.csv")
    }
    posthoc = {
        (row["axis"], row["comparison"]): row
        for row in read_csv(path / f"{prefix}posthoc_pairwise.csv")
    }
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
        model_statistics = {}
        for model in MODELS:
            model_row = model_rows[model]
            mean = float(model_row["mean"])
            ci_low = float(model_row["ci95_low"])
            ci_high = float(model_row["ci95_high"])
            model_statistics[MODEL_LABELS[model]] = {
                "mean": round(mean, 6),
                "std": round(float(model_row["std"]), 6),
                "ci95_low": round(ci_low, 6),
                "ci95_high": round(ci_high, 6),
                "errorbar_lower": round(mean - ci_low, 6),
                "errorbar_upper": round(ci_high - mean, 6),
            }
        omnibus_row = omnibus[axis_key]
        pairwise = []
        for comparison in COMPARISONS:
            comparison_row = posthoc.get((axis_key, comparison))
            if comparison_row is None:
                pairwise.append(
                    {
                        "comparison": comparison,
                        "status": "not_tested_omnibus_not_significant",
                        "p_holm": None,
                        "stars": "",
                    }
                )
                continue
            p_holm = float(comparison_row["p_holm"])
            pairwise.append(
                {
                    "comparison": comparison,
                    "status": "tested",
                    "mean_difference": round(float(comparison_row["mean_diff"]), 6),
                    "ci95_low": round(float(comparison_row["ci95_low"]), 6),
                    "ci95_high": round(float(comparison_row["ci95_high"]), 6),
                    "p_raw": float(comparison_row["p_raw"]),
                    "p_holm": p_holm,
                    "stars": significance_stars(p_holm),
                    "significant": comparison_row["significant"].lower() == "true",
                }
            )
        scores.append(
            {
                "category": category,
                "axis": axis or category,
                "axis_key": axis_key,
                "n": int(model_rows["base"]["n"]),
                "models": model_statistics,
                "omnibus": {
                    "friedman_chi2": float(omnibus_row["friedman_chi2"]),
                    "degrees_of_freedom": int(omnibus_row["degrees_of_freedom"]),
                    "p_value": float(omnibus_row["p_value"]),
                    "kendalls_w": float(omnibus_row["kendalls_w"]),
                    "significant": omnibus_row["significant"].lower() == "true",
                },
                "pairwise_holm": pairwise,
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
            values = row["models"][label]
            lines.append(
                f"  {label}: mean={values['mean']:.3f}, std={values['std']:.3f}, "
                f"bootstrap_ci95=[{values['ci95_low']:.3f}, {values['ci95_high']:.3f}], "
                f"errorbar=[-{values['errorbar_lower']:.3f}, +{values['errorbar_upper']:.3f}]"
            )
        omnibus = row["omnibus"]
        lines.append(
            f"  omnibus: Friedman_chi2={omnibus['friedman_chi2']:.4f}, "
            f"df={omnibus['degrees_of_freedom']}, p={omnibus['p_value']:.8g}, "
            f"Kendalls_W={omnibus['kendalls_w']:.4f}, significant={omnibus['significant']}"
        )
        lines.append("  pairwise_holm:")
        for comparison in row["pairwise_holm"]:
            if comparison["status"] != "tested":
                lines.append(
                    f"    {comparison['comparison']}: not_tested "
                    "(Friedman omnibus was not significant)"
                )
            else:
                lines.append(
                    f"    {comparison['comparison']}: mean_diff={comparison['mean_difference']:.3f}, "
                    f"p_holm={comparison['p_holm']:.8g}, stars={comparison['stars']}"
                )
        lines.append("")
    output.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "evaluation_set": evaluation_set,
                "model_order": [MODEL_LABELS[model] for model in MODELS],
                "error_bar": "bootstrap_95_percent_confidence_interval",
                "significance": "Holm-adjusted paired post-hoc; *=p<.05, **=p<.01, ***=p<.001",
                "axes": scores,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def select_representative_scores(
    dataset: str, scores: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """論文図で固定した代表7軸を指定順に抽出する。"""
    by_key = {row["axis_key"]: row for row in scores}
    expected = REPRESENTATIVE_AXES[dataset]
    missing = [axis for axis in expected if axis not in by_key]
    if missing:
        raise ValueError(f"{dataset}: 代表軸の評価が不足しています: {missing}")
    return [by_key[axis] for axis in expected]


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
            if evaluation_set == "main":
                scores = select_representative_scores(dataset, scores)
            write_scores(
                dataset=dataset,
                evaluation_set=evaluation_set,
                scores=scores,
                output=dataset_root / "reports" / f"axis_scores_{evaluation_set}",
            )
            if evaluation_set == "main":
                combined.append({"dataset": dataset, "axes": scores})

    combined_txt: list[str] = [
        "GOLD-ONLY DPO 4-MODEL AXIS SCORES FOR FIGURE GENERATION",
        "",
        "SCORE_SCALE: 1-10",
        "ERROR_BAR: bootstrap 95% confidence interval",
        "ERRORBAR_FORMAT: [mean-ci95_low, ci95_high-mean]",
        "OMNIBUS_TEST: Friedman test for four paired models",
        "EFFECT_SIZE: Kendall's W",
        "POSTHOC: paired permutation test, Holm correction within each axis (6 pairs)",
        "STAR_RULE: * p_holm<0.05; ** p_holm<0.01; *** p_holm<0.001; ns otherwise",
        "IMPORTANT: post-hoc was not run when Friedman omnibus was not significant",
        "MODEL_ORDER: Base, Gold-only DPO, BASiS-DPO, Random-DPO",
        "EVALUATION_SET: main evaluation only (MediTOD OOD/cluster are separate files)",
        "AXIS_SET: pre-specified representative seven axes per dataset",
        "",
    ]
    for item in combined:
        combined_txt.extend([item["dataset"], ""])
        for row in item["axes"]:
            combined_txt.append(f"AXIS: {row['axis_key']} (n={row['n']})")
            for label, values in row["models"].items():
                combined_txt.append(
                    f"  {label}: mean={values['mean']:.3f}, std={values['std']:.3f}, "
                    f"bootstrap_ci95=[{values['ci95_low']:.3f}, {values['ci95_high']:.3f}], "
                    f"errorbar=[-{values['errorbar_lower']:.3f}, +{values['errorbar_upper']:.3f}]"
                )
            omnibus = row["omnibus"]
            combined_txt.append(
                f"  omnibus: Friedman_chi2={omnibus['friedman_chi2']:.4f}, "
                f"df={omnibus['degrees_of_freedom']}, p={omnibus['p_value']:.8g}, "
                f"Kendalls_W={omnibus['kendalls_w']:.4f}, significant={omnibus['significant']}"
            )
            combined_txt.append("  pairwise_holm:")
            for comparison in row["pairwise_holm"]:
                if comparison["status"] == "tested":
                    combined_txt.append(
                        f"    {comparison['comparison']}: "
                        f"mean_diff={comparison['mean_difference']:.3f}, "
                        f"p_holm={comparison['p_holm']:.8g}, stars={comparison['stars']}"
                    )
                else:
                    combined_txt.append(
                        f"    {comparison['comparison']}: not_tested "
                        "(Friedman omnibus was not significant)"
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
