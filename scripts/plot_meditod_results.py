#!/usr/bin/env python3
"""MediTOD × WildChat-1Mの事後選択7軸を描画する。"""

from pathlib import Path

from tools.plot_oracle_grouped_bars import AxisSpec, FigureSpec, run_plot_cli


ROOT = Path("artifacts/meditod_wildchat/runs/meditod_wildchat_gpt56_v2")
HISTORY_RAW = ROOT / "evaluation/oracle/history/raw.jsonl"
GENERAL_RAW = ROOT / "evaluation/oracle/general/raw.jsonl"
SAFETY_RAW = ROOT / "evaluation/oracle/safety/raw.jsonl"

SPEC = FigureSpec(
    slug="meditod_wildchat",
    title="MediTOD × WildChat-1M",
    axes=(
        AxisSpec(
            "response_relevance",
            "Response Relevance",
            {"base": 4.46, "basis": 5.09, "random_dpo": 4.91},
            GENERAL_RAW,
        ),
        AxisSpec(
            "overall_quality",
            "Overall Quality",
            {"base": 5.16, "basis": 5.69, "random_dpo": 5.46},
            GENERAL_RAW,
        ),
        AxisSpec(
            "premature_assessment_avoidance",
            "Premature Assessment Avoidance",
            {"base": 8.43, "basis": 8.90, "random_dpo": 8.63},
            HISTORY_RAW,
        ),
        AxisSpec(
            "appropriate_uncertainty",
            "Appropriate Uncertainty",
            {"base": 7.84, "basis": 8.26, "random_dpo": 7.92},
            SAFETY_RAW,
        ),
        AxisSpec(
            "understandable",
            "Understandability",
            {"base": 8.56, "basis": 8.77, "random_dpo": 8.55},
            GENERAL_RAW,
        ),
        AxisSpec(
            "unsafe_medical_advice",
            "Unsafe Medical Advice Avoidance",
            {"base": 9.52, "basis": 9.71, "random_dpo": 9.36},
            SAFETY_RAW,
        ),
        AxisSpec(
            "unsupported_diagnosis",
            "Unsupported Diagnosis Avoidance",
            {"base": 9.62, "basis": 9.82, "random_dpo": 9.61},
            SAFETY_RAW,
        ),
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_plot_cli(SPEC))
