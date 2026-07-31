#!/usr/bin/env python3
"""ESConv × DailyDialogの代表7軸を描画する。"""

from pathlib import Path

from tools.plot_oracle_grouped_bars import AxisSpec, FigureSpec, run_plot_cli


ROOT = Path(
    "artifacts/evaluations/oracle_eval_runs/"
    "esconv_topconf_three_model_esconv_v2_100_gpt54_v1_"
    "topconf_three_model_esconv_v2_10pt"
)
TST_RAW = Path(
    "artifacts/evaluations/oracle_eval_runs/"
    "esconv_topconf_three_model_gpt54_100_10pt_"
    "topconf_three_model_10pt/oracle_tst_10pt/raw.jsonl"
)
STYLE_RAW = ROOT / "oracle_conversation_style_esconv_v2_10pt/raw.jsonl"
TRANSITION_RAW = ROOT / "oracle_strategy_transition_esconv_v2_10pt/raw.jsonl"

SPEC = FigureSpec(
    slug="esconv_dailydialog",
    title="ESConv × DailyDialog",
    axes=(
        AxisSpec(
            "style_strength",
            "Style Strength",
            {"base": 8.26, "basis": 8.65, "random_dpo": 8.21},
            TST_RAW,
        ),
        AxisSpec(
            "esconv_tone_similarity",
            "ESConv Tone Similarity",
            {"base": 8.21, "basis": 8.52, "random_dpo": 8.13},
            STYLE_RAW,
        ),
        AxisSpec(
            "supporter_role_consistency",
            "Supporter Role Consistency",
            {"base": 8.34, "basis": 8.60, "random_dpo": 8.41},
            STYLE_RAW,
        ),
        AxisSpec(
            "non_directive_support_style",
            "Non-directive Support Style",
            {"base": 7.84, "basis": 8.34, "random_dpo": 7.99},
            STYLE_RAW,
        ),
        AxisSpec(
            "strategy_stage_alignment",
            "Strategy/Stage Alignment",
            {"base": 7.84, "basis": 8.32, "random_dpo": 7.98},
            TRANSITION_RAW,
        ),
        AxisSpec(
            "premature_advice_avoidance",
            "Premature Advice Avoidance",
            {"base": 8.57, "basis": 9.44, "random_dpo": 8.92},
            TRANSITION_RAW,
        ),
        AxisSpec(
            "naturalness",
            "Naturalness",
            {"base": 8.35, "basis": 8.53, "random_dpo": 8.17},
            TST_RAW,
        ),
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_plot_cli(SPEC))
