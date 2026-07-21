from __future__ import annotations

from pathlib import Path

from scripts.prepare_esconv_google_form_eval import MODEL_KEYS
from scripts.prepare_esconv_google_form_likert_eval import (
    FINAL_CHOICE_OPTIONS,
    FORM_DESCRIPTION,
    LIKERT_COLUMNS,
    LIKERT_STATEMENTS,
    public_record,
    selection_diagnostics,
    write_apps_script,
)


def candidate(index: int, category: str, advantage: float) -> dict:
    axes = (
        "style_strength",
        "esconv_tone_similarity",
        "supporter_role_consistency",
        "non_directive_support_style",
        "premature_advice_avoidance",
    )
    return {
        "prompt_id": f"p{index:03d}",
        "category": category,
        "axis_focus": [],
        "history": [{"speaker": "User", "text": f"履歴{index}"}],
        "prompt": f"相談{index}",
        "responses": {
            "base": f"base {index}",
            "basis": f"basis {index}",
            "random": f"random {index}",
        },
        "oracle_axis_scores": {
            "base": {axis: 6.0 for axis in axes},
            "bayes_dpo": {axis: 8.0 + advantage for axis in axes},
            "random_dpo": {axis: 6.5 for axis in axes},
        },
        "representative_means": {
            "base": 6.0,
            "basis": 8.0 + advantage,
            "random": 6.5,
        },
        "basis_advantage_over_best_control": 1.5 + advantage,
    }


def test_public_record_has_three_seven_axis_ratings_and_final_choice():
    row = candidate(1, "category", 0.5)
    public = public_record(
        row,
        item_number=1,
        order=("basis", "base", "random"),
    )
    assert len(public["likert_statements"]) == 7
    assert public["likert_columns"] == list(LIKERT_COLUMNS)
    assert public["response_a"] == "basis 1"
    assert public["final_choice_options"] == list(FINAL_CHOICE_OPTIONS)
    assert "モデル" not in public["final_choice_question"]


def test_selection_diagnostics_marks_result_as_posthoc():
    rows = [candidate(index, "category", index / 100) for index in range(20)]
    diagnostics = selection_diagnostics(rows, permutations=200, seed=42)
    assert diagnostics["selection_conditioned_posthoc"] is True
    assert diagnostics["n"] == 20
    assert (
        diagnostics["representative_five_axis_means"]["basis"]
        > diagnostics["representative_five_axis_means"]["base"]
    )
    assert all(
        row["mean_difference"] > 0
        for row in diagnostics["pairwise_representative_mean"][:2]
    )


def test_apps_script_uses_readable_scales_name_and_consent_branch(tmp_path: Path):
    row = candidate(1, "category", 0.5)
    public = public_record(
        row,
        item_number=1,
        order=tuple(MODEL_KEYS),
    )
    output = tmp_path / "form.gs"
    write_apps_script(output, [public], "テスト")
    script = output.read_text(encoding="utf-8")
    assert "addScaleItem()" in script
    assert ".setBounds(1, 7)" in script
    assert "氏名を入力してください。" in script
    assert "参加者ID" not in script
    assert "addGridItem()" not in script
    assert "PageNavigationType.SUBMIT" in script
    assert "createEsconvLikertForm" in script
    assert "実験指示" in script
    assert "助言や解決策を急がず" in script
    assert "参加者情報" in script
    assert "参加者情報と評価 1" not in script
    assert script.index("参加者情報") < script.index("評価 1 /")
    assert "当てはまる程度を選んでください。" in script
    assert "質問 ${statementIndex + 1}" in script
    assert "ESConv" not in FORM_DESCRIPTION
    assert len(LIKERT_STATEMENTS) == 7
