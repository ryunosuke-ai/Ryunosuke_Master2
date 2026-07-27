#!/usr/bin/env python3
"""MathDial × WildChat-1Mの事後選択7軸を描画する。"""

from pathlib import Path

from tools.plot_oracle_grouped_bars import AxisSpec, FigureSpec, run_plot_cli


RAW = Path(
    "artifacts/mathdial_wildchat/evaluation_rechecks/"
    "mathdial_v6_instruction_outcome_selected_top100_v1/"
    "evaluation/oracle/pedagogical_v2/raw.jsonl"
)

SPEC = FigureSpec(
    slug="mathdial_wildchat",
    title="MathDial × WildChat-1M",
    axes=(
        AxisSpec(
            "equitable_tutoring",
            "Equitable Tutoring",
            {"base": 6.770, "basis": 7.750, "random_dpo": 6.000},
            RAW,
        ),
        AxisSpec(
            "learner_reasoning_diagnosis",
            "Reasoning Diagnosis",
            {"base": 7.380, "basis": 8.700, "random_dpo": 6.970},
            RAW,
        ),
        AxisSpec(
            "mistake_location_and_targeting",
            "Mistake Targeting",
            {"base": 7.530, "basis": 8.850, "random_dpo": 7.270},
            RAW,
        ),
        AxisSpec(
            "guidance_quality",
            "Guidance Quality",
            {"base": 6.880, "basis": 8.020, "random_dpo": 6.570},
            RAW,
        ),
        AxisSpec(
            "feedback_actionability",
            "Feedback Actionability",
            {"base": 7.010, "basis": 8.030, "random_dpo": 6.180},
            RAW,
        ),
        AxisSpec(
            "answer_revealing_calibration",
            "Answer Calibration",
            {"base": 7.880, "basis": 8.860, "random_dpo": 7.500},
            RAW,
        ),
        AxisSpec(
            "teacher_move_stage_alignment",
            "Move/Stage Alignment",
            {"base": 7.440, "basis": 8.620, "random_dpo": 7.060},
            RAW,
        ),
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_plot_cli(SPEC))
