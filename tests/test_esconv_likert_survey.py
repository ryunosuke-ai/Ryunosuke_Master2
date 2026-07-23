from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from apps.esconv_likert_user_eval import (
    build_reference_html,
    find_missing_evaluation_fields,
    first_unanswered_index,
    readable_text_html,
)
from core.esconv_likert_survey import (
    EXPECTED_AXIS_KEYS,
    RESPONSE_POSITIONS,
    assign_participant,
    export_responses_csv,
    load_participant_responses,
    load_public_experiments,
    save_response,
    validate_public_item,
)
from scripts.prepare_esconv_google_form_likert_eval import (
    GOOD_RESPONSE_EXAMPLE,
    LIKERT_STATEMENTS,
    STYLE_FEATURES,
)


def public_item(index: int) -> dict:
    """Webアンケート用の公開fixtureを作る。"""
    return {
        "item_id": f"item_{index:02d}",
        "item_number": index,
        "conversation": f"相談者: 会話{index}",
        "response_a": f"応答A {index}",
        "response_b": f"応答B {index}",
        "response_c": f"応答C {index}",
        "likert_statements": list(LIKERT_STATEMENTS),
        "likert_columns": [str(value) for value in range(1, 8)],
        "likert_anchors": {},
        "final_choice_question": "最もふさわしい応答はどれですか。",
        "final_choice_options": [
            "応答A",
            "応答B",
            "応答C",
            "ほぼ同じ",
            "判断できない",
        ],
    }


def write_public_experiments(root: Path) -> None:
    """A/B各10件の公開JSONLを書く。"""
    for experiment in ("a", "b"):
        directory = root / f"experiment_{experiment}"
        directory.mkdir(parents=True)
        with (directory / "form_items_public.jsonl").open("w", encoding="utf-8") as file:
            for index in range(1, 11):
                file.write(json.dumps(public_item(index), ensure_ascii=False) + "\n")


def complete_ratings(value: int = 6) -> dict[str, dict[str, int]]:
    """7軸x3応答の完全なratingを返す。"""
    return {
        axis_key: {position: value for position in RESPONSE_POSITIONS}
        for axis_key in EXPECTED_AXIS_KEYS
    }


def test_public_experiments_load_without_private_model_identity(tmp_path: Path):
    write_public_experiments(tmp_path)
    experiments = load_public_experiments(tmp_path)
    assert set(experiments) == {"A", "B"}
    assert len(experiments["A"]) == len(experiments["B"]) == 10
    assert "position_to_model" not in experiments["A"][0]


def test_public_item_rejects_private_mapping():
    item = public_item(1)
    item["position_to_model"] = {"A": "basis"}
    with pytest.raises(ValueError, match="非公開情報"):
        validate_public_item(item, experiment="A")


def test_participant_assignment_is_balanced_stable_and_resumable(tmp_path: Path):
    database = tmp_path / "responses.sqlite3"
    first, created_first = assign_participant(database, "研究 太郎")
    second, created_second = assign_participant(database, "研究 花子")
    resumed, created_resumed = assign_participant(database, "  研究   太郎 ")
    assert created_first is True
    assert created_second is True
    assert created_resumed is False
    assert first.experiment == "A"
    assert second.experiment == "B"
    assert resumed.participant_id == first.participant_id


def test_participant_assignment_can_be_fixed_by_experiment_url(tmp_path: Path):
    database = tmp_path / "responses.sqlite3"
    participant_b, _ = assign_participant(
        database,
        "実験 B参加者",
        requested_experiment="B",
    )
    participant_a, _ = assign_participant(
        database,
        "実験 A参加者",
        requested_experiment="A",
    )
    assert participant_b.experiment == "B"
    assert participant_a.experiment == "A"
    with pytest.raises(ValueError, match="実験Bへ割当済み"):
        assign_participant(
            database,
            "実験 B参加者",
            requested_experiment="A",
        )


def test_response_upsert_resume_and_csv_export(tmp_path: Path):
    database = tmp_path / "responses.sqlite3"
    participant, _ = assign_participant(database, "研究 太郎")
    created = save_response(
        database,
        participant=participant,
        item_id="item_01",
        ratings=complete_ratings(6),
        final_choice="応答B",
        final_choice_reason="会話に合っているため",
        comment="最初の回答",
    )
    updated = save_response(
        database,
        participant=participant,
        item_id="item_01",
        ratings=complete_ratings(7),
        final_choice="応答A",
        final_choice_reason="気持ちを具体的に受け止めているため",
        comment="修正済み",
    )
    responses = load_participant_responses(database, participant.participant_id)
    assert created is True
    assert updated is False
    assert responses["item_01"]["ratings"]["style_strength"]["A"] == 7
    assert responses["item_01"]["final_choice"] == "応答A"
    assert responses["item_01"]["final_choice_reason"].startswith("気持ち")

    output = tmp_path / "responses.csv"
    written = export_responses_csv(database, output)
    with output.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert written == len(EXPECTED_AXIS_KEYS) * len(RESPONSE_POSITIONS)
    assert rows[0]["full_name"] == "研究 太郎"
    assert rows[0]["final_choice_reason"].startswith("気持ち")


def test_existing_database_adds_reason_without_losing_responses(tmp_path: Path):
    database = tmp_path / "legacy.sqlite3"
    participant, _ = assign_participant(database, "既存 回答者")
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
                json.dumps(complete_ratings()),
                "応答A",
                "",
                "created",
                "updated",
            ),
        )
        connection.execute("DROP TABLE responses_new")
    responses = load_participant_responses(database, participant.participant_id)
    assert responses["item_01"]["final_choice"] == "応答A"
    assert responses["item_01"]["final_choice_reason"] == "理由なし"


def test_reference_html_escapes_input_and_contains_all_responses():
    item = public_item(1)
    item["conversation"] = "<script>alert(1)</script>"
    rendered = build_reference_html(item)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert all(f"応答{position}" in rendered for position in RESPONSE_POSITIONS)
    assert "評価の目安" in rendered
    assert "良い例" in rendered
    assert "良くない例2" in rendered


def test_readable_text_html_adds_sentence_breaks_without_changing_text():
    rendered = readable_text_html("つらかったのですね。よく話してくれました。")
    assert rendered == "つらかったのですね。<br>よく話してくれました。"
    assert readable_text_html("<script>危険</script>").startswith("&lt;script&gt;")


def test_first_unanswered_and_style_instruction_contract():
    items = [public_item(1), public_item(2)]
    assert first_unanswered_index(items, {}) == 0
    assert first_unanswered_index(items, {"item_01": {}}) == 1
    assert first_unanswered_index(items, {"item_01": {}, "item_02": {}}) == 2
    assert len(STYLE_FEATURES) == 3
    assert "?" not in GOOD_RESPONSE_EXAMPLE
    assert "？" not in GOOD_RESPONSE_EXAMPLE


def test_missing_evaluation_fields_prevent_incomplete_submission():
    ratings = {
        axis_key: {position: 6 for position in RESPONSE_POSITIONS}
        for axis_key in EXPECTED_AXIS_KEYS
    }
    ratings["style_strength"]["B"] = None
    assert find_missing_evaluation_fields(ratings, None, "") == [
        "質問1・応答B",
        "最後の質問",
        "選んだ理由",
    ]
    ratings["style_strength"]["B"] = 6
    assert find_missing_evaluation_fields(ratings, "応答A", "") == [
        "選んだ理由"
    ]
    assert find_missing_evaluation_fields(
        ratings,
        "応答A",
        "具体的に受け止めているため",
    ) == []


def test_streamlit_start_and_evaluation_screens_render(tmp_path: Path, monkeypatch):
    database = tmp_path / "ui_responses.sqlite3"
    monkeypatch.setenv("ESCONV_SURVEY_DATABASE", database.as_posix())
    app = AppTest.from_file(
        "apps/esconv_likert_user_eval.py",
        default_timeout=30,
    ).run()
    assert not app.exception
    assert app.title[0].value == "相談支援応答の7段階評価"
    app.text_input[0].input("画面 確認")
    app.checkbox[0].check()
    app.button[0].click()
    app.run()
    assert not app.exception
    assert app.title[0].value == "相談支援応答の評価"
    assert len(app.radio) == 22
    assert any(
        area.label == "選んだ理由"
        for area in app.text_area
    )
    assert any(
        "そう選んだ理由を教えてください。" in block.value
        and "必須回答です。" in block.value
        for block in app.markdown
    )
    assert any(
        "rating-scroll-bottom-spacer" in block.value
        for block in app.markdown
    )
    assert [radio.label for radio in app.radio[:3]] == ["応答A", "応答B", "応答C"]
    assert any("reference-panel" in block.value for block in app.markdown)
    assert any(
        'stColumn"]:has(.reference-panel)' in block.value
        for block in app.markdown
    )
    assert any(
        'class*="st-key-rating_scroll_container_"' in block.value
        and "overscroll-behavior: contain" in block.value
        and ".reference-panel" in block.value
        and "overflow: visible" in block.value
        for block in app.markdown
    )
    scroll_containers = [
        node
        for node in app.get("flex_container")
        if node.proto.id.endswith("-rating_scroll_container_item_01")
    ]
    assert len(scroll_containers) == 1
    assert scroll_containers[0].proto.height_config.pixel_height == 500
    assert all(
        child.type != "button"
        for child in scroll_containers[0].children.values()
    )
    navigation_containers = [
        node
        for node in app.get("flex_container")
        if node.proto.id.endswith("-evaluation_navigation")
    ]
    assert len(navigation_containers) == 1
    assert any(
        ".st-key-evaluation_navigation" in block.value
        and "position: fixed" in block.value
        and "bottom: 0" in block.value
        and ".st-key-evaluation_validation" in block.value
        and "top: 72px" in block.value
        and "right: 20px" in block.value
        and "width: min(440px" in block.value
        for block in app.markdown
    )


def test_streamlit_experiment_b_url_fixes_assignment(tmp_path: Path, monkeypatch):
    database = tmp_path / "experiment_b.sqlite3"
    monkeypatch.setenv("ESCONV_SURVEY_DATABASE", database.as_posix())
    app = AppTest.from_file(
        "apps/esconv_likert_user_eval.py",
        default_timeout=30,
    )
    app.query_params["experiment"] = "B"
    app.run()
    assert any("実験B" in caption.value for caption in app.caption)
    app.text_input[0].input("B 専用参加者")
    app.checkbox[0].check()
    app.button[0].click()
    app.run()
    assert not app.exception
    participant, created = assign_participant(
        database,
        "B 専用参加者",
        requested_experiment="B",
    )
    assert created is False
    assert participant.experiment == "B"


def test_streamlit_same_name_resumes_first_unanswered_item(tmp_path: Path, monkeypatch):
    database = tmp_path / "resume.sqlite3"
    participant, _ = assign_participant(database, "再開 確認")
    save_response(
        database,
        participant=participant,
        item_id="item_01",
        ratings=complete_ratings(6),
        final_choice="応答A",
        final_choice_reason="理由を記録",
        comment="",
    )
    monkeypatch.setenv("ESCONV_SURVEY_DATABASE", database.as_posix())
    app = AppTest.from_file(
        "apps/esconv_likert_user_eval.py",
        default_timeout=30,
    ).run()
    app.text_input[0].input("再開 確認")
    app.checkbox[0].check()
    app.button[0].click()
    app.run()
    assert not app.exception
    assert any("評価 2 / 10" in caption.value for caption in app.caption)
