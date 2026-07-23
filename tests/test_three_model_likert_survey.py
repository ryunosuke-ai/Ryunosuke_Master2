from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from streamlit.testing.v1 import AppTest

from core.three_model_likert_survey import (
    FINAL_CHOICES,
    RESPONSE_POSITIONS,
    assign_participant,
    axis_keys,
    export_responses_csv,
    load_definition,
    load_participant_responses,
    load_public_experiments,
    save_response,
)
from tools.analyze_three_model_likert_responses import analyze, load_ratings
from tools.prepare_three_model_likert_eval import build_records, select_outcome_enriched, write_jsonl


def response_row(index: int) -> dict:
    return {
        "sample_id": f"sample_{index}",
        "conversation": f"User: 症状についての会話{index}",
        "conversation_id": f"conversation_{index}",
        "stratum": "symptom_attributes" if index % 2 else "medical_history",
        "responses": {"base": f"一般応答{index}", "basis": f"具体的な質問応答{index}", "random_dpo": f"ランダム応答{index}"},
        "oracle_axis_scores": {"base": {"axis": 5.0}, "basis": {"axis": 8.0}, "random_dpo": {"axis": 4.0}},
        "oracle_means": {"base": 5.0, "basis": 8.0, "random_dpo": 4.0},
        "basis_advantage": 3.0,
        "readability_passed": True,
        "readability_reason": "passed",
    }


def prepare_public(root: Path, definition: dict, count: int = 4) -> list[dict]:
    selected = select_outcome_enriched([response_row(index) for index in range(count)], count)
    public, private = build_records(selected, definition, seed=42)
    for experiment, rows in public.items():
        write_jsonl(root / f"experiment_{experiment.lower()}" / "form_items_public.jsonl", rows)
    write_jsonl(root / "private_answer_key.jsonl", private)
    (root / "manifest.json").write_text(json.dumps({"dataset": definition["dataset"], "survey_version": definition["survey_version"], "items_per_experiment": count // 2}), encoding="utf-8")
    return private


def complete_ratings(definition: dict, value: int) -> dict[str, dict[str, int]]:
    return {axis: {position: value for position in RESPONSE_POSITIONS} for axis in axis_keys(definition)}


def test_public_bundle_has_no_model_identity_and_is_balanced(tmp_path: Path):
    definition = load_definition(Path("configs/user_evaluations/meditod_likert_v1.yaml"))
    private = prepare_public(tmp_path, definition)
    experiments = load_public_experiments(tmp_path, definition)
    assert len(experiments["A"]) == len(experiments["B"]) == 2
    public_text = (tmp_path / "experiment_a" / "form_items_public.jsonl").read_text()
    assert "position_to_model" not in public_text
    assert "basis_advantage" not in public_text
    assert "oracle" not in public_text.lower()
    positions = [next(position for position, model in row["position_to_model"].items() if model == "basis") for row in private]
    assert len(set(positions)) >= 2


def test_sqlite_assignment_resume_export_and_statistics(tmp_path: Path):
    definition = load_definition(Path("configs/user_evaluations/meditod_likert_v1.yaml"))
    private = prepare_public(tmp_path / "forms", definition)
    database = tmp_path / "responses.sqlite3"
    participant, created = assign_participant(database, definition, "研究 太郎", requested_experiment="A")
    assert created
    item_id = private[0]["item_id"]
    save_response(
        database,
        definition,
        participant=participant,
        item_id=item_id,
        ratings=complete_ratings(definition, 6),
        final_choice=FINAL_CHOICES[0],
        final_choice_reason="不足情報を具体的に聞いているため",
        comment="確認",
    )
    resumed, created_again = assign_participant(database, definition, " 研究  太郎 ", requested_experiment="A")
    assert not created_again and resumed.participant_id == participant.participant_id
    assert item_id in load_participant_responses(database, definition, participant.participant_id)
    output = tmp_path / "responses.csv"
    written = export_responses_csv(database, definition, output)
    assert written == len(axis_keys(definition)) * 3
    with output.open(encoding="utf-8-sig") as file:
        first_row = next(csv.DictReader(file))
    assert first_row["full_name"] == "研究 太郎"
    assert first_row["final_choice_reason"].startswith("不足情報")
    mapping = {row["item_id"]: row["position_to_model"] for row in private}
    values, choices = load_ratings(database, mapping)
    assert choices
    # 参加者が1名だけのため検定表は空だが、復号済み構造は保持される。
    assert values
    assert analyze(values, permutations=10, bootstrap=10, seed=42) == ([], [], [])


def test_generic_existing_database_migrates_reason_column(tmp_path: Path):
    definition = load_definition(
        Path("configs/user_evaluations/meditod_likert_v1.yaml")
    )
    database = tmp_path / "legacy.sqlite3"
    participant, _ = assign_participant(database, definition, "既存 医療回答者")
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE responses RENAME TO responses_new")
        connection.execute(
            """
            CREATE TABLE responses (
                participant_id TEXT NOT NULL,
                experiment TEXT NOT NULL,
                item_id TEXT NOT NULL,
                ratings_json TEXT NOT NULL,
                final_choice TEXT NOT NULL,
                comment TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (participant_id, item_id)
            )
            """
        )
        connection.execute(
            "INSERT INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                participant.participant_id,
                participant.experiment,
                "item_01",
                json.dumps(complete_ratings(definition, 6)),
                FINAL_CHOICES[0],
                "",
                "created",
                "updated",
            ),
        )
        connection.execute("DROP TABLE responses_new")
    loaded = load_participant_responses(
        database,
        definition,
        participant.participant_id,
    )
    assert loaded["item_01"]["final_choice_reason"] == "理由なし"


def test_streamlit_generic_layout_and_dataset_specific_text(tmp_path: Path, monkeypatch):
    definition_path = Path("configs/user_evaluations/meditod_likert_v1.yaml")
    definition = load_definition(definition_path)
    forms = tmp_path / "forms"
    prepare_public(forms, definition, count=20)
    monkeypatch.setenv("THREE_MODEL_SURVEY_DEFINITION", str(definition_path))
    monkeypatch.setenv("THREE_MODEL_SURVEY_FORM_ROOT", str(forms))
    monkeypatch.setenv("THREE_MODEL_SURVEY_DATABASE", str(tmp_path / "responses.sqlite3"))
    app = AppTest.from_file("apps/three_model_likert_user_eval.py", default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "医療面接応答の7段階評価"
    app.text_input[0].input("画面 確認")
    app.checkbox[0].check()
    app.button[0].click(); app.run()
    assert not app.exception
    assert app.title[0].value == "医療面接応答の評価"
    assert len(app.radio) == len(axis_keys(definition)) * 3 + 1
    assert any(
        area.label == "そう選んだ理由を教えてください。"
        for area in app.text_area
    )
    assert not any("必須回答です。" in block.value for block in app.markdown)
    assert any(
        "rating-scroll-bottom-spacer" in block.value
        for block in app.markdown
    )
    assert any("reference-panel" in block.value for block in app.markdown)
    assert any("評価の目安" in block.value for block in app.markdown)
    assert any("position: fixed" in block.value and "evaluation_navigation" in block.value for block in app.markdown)
