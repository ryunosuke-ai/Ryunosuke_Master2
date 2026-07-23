"""MathDial/MediTOD向け汎用3モデルLikert Streamlitアプリ。"""

from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.esconv_likert_user_eval import (  # noqa: E402
    apply_page_style,
    readable_text_html,
    render_html_panel,
    reset_evaluation_scroll,
)
from core.three_model_likert_survey import (  # noqa: E402
    FINAL_CHOICE_REASON_QUESTION,
    FINAL_CHOICES,
    RESPONSE_POSITIONS,
    Participant,
    assign_participant,
    axis_keys,
    load_definition,
    load_participant_responses,
    load_public_experiments,
    participant_to_dict,
    save_response,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--definition", type=Path, default=Path(os.environ.get("THREE_MODEL_SURVEY_DEFINITION", "configs/user_evaluations/meditod_likert_v1.yaml")))
    parser.add_argument("--form-root", type=Path, default=Path(os.environ.get("THREE_MODEL_SURVEY_FORM_ROOT", "artifacts/user_eval/meditod")))
    parser.add_argument("--database", type=Path, default=Path(os.environ.get("THREE_MODEL_SURVEY_DATABASE", "artifacts/user_eval/web/meditod_likert_responses.sqlite3")))
    args, _ = parser.parse_known_args()
    return args


def requested_experiment() -> str | None:
    raw = st.query_params.get("experiment")
    if raw is None:
        return None
    if isinstance(raw, list):
        raw = raw[-1] if raw else ""
    value = str(raw).strip().upper()
    if value not in {"A", "B"}:
        raise ValueError("URLのexperimentにはAまたはBを指定してください。")
    return value


def build_reference_html(
    item: dict[str, Any],
    definition: dict[str, Any],
) -> str:
    parts = [
        '<div class="reference-panel">',
        '<div class="reference-heading">これまでの会話</div>',
        f'<div class="conversation-text">{readable_text_html(item["conversation"])}</div>',
    ]
    for position in RESPONSE_POSITIONS:
        parts.extend(
            [
                '<div class="response-card">',
                f'<div class="response-label">応答{position}</div>',
                f'<div class="response-text">{readable_text_html(item[f"response_{position.lower()}"])}</div>',
                "</div>",
            ]
        )
    example = definition["example"]
    poor_examples = "".join(
        (
            '<div class="guide-poor">'
            f"<strong>良くない例{index}</strong>"
            f"{readable_text_html(row['response'])}<br>"
            f"特徴: {readable_text_html(row['explanation'])}"
            "</div>"
        )
        for index, row in enumerate(example["poor_responses"], start=1)
    )
    parts.extend(
        [
            '<div class="reference-example-guide">',
            '<div class="guide-heading">評価の目安</div>',
            '<div class="guide-good">',
            "<strong>良い例</strong>",
            readable_text_html(example["good_response"]),
            "<br>特徴: ",
            readable_text_html(example["good_explanation"]),
            "</div>",
            poor_examples,
            "</div>",
        ]
    )
    parts.append("</div>")
    return "".join(parts)


def first_unanswered(items: list[dict[str, Any]], saved: dict[str, Any]) -> int:
    for index, item in enumerate(items):
        if str(item["item_id"]) not in saved:
            return index
    return len(items)


def render_start(database: Path, definition: dict[str, Any], experiments: dict[str, list[dict[str, Any]]], requested: str | None) -> None:
    st.title(definition["page_title"])
    st.caption(f"実験{requested}" if requested else "実験A/B自動割当")
    features = "".join(f"<li>{html.escape(str(value))}</li>" for value in definition["style_features"])
    render_html_panel(
        "survey-intro",
        "<h3>実験指示</h3>"
        f"<p>{html.escape(str(definition['intro']))}</p>"
        f'<ul class="style-list">{features}</ul>'
        f"<p>評価は全部で{len(next(iter(experiments.values())))}件です。各評価では、これまでの会話と匿名の応答A〜Cを示します。各応答を同じ7項目で1〜7点評価し、最後に最もふさわしい応答を選んでください。</p>",
    )
    st.subheader("評価例")
    example = definition["example"]
    st.write(f"**User:** {example['user']}")
    good, poor = st.columns(2, gap="large")
    with good:
        render_html_panel("example-panel good", f'<div class="example-title">良い応答例</div><div class="example-response">{html.escape(str(example["good_response"]))}</div><p>{html.escape(str(example["good_explanation"]))}</p>')
    with poor:
        contents = ['<div class="example-title">良くない応答例</div>']
        for index, row in enumerate(example["poor_responses"], start=1):
            contents.append(f'<strong>例{index}</strong><div class="example-response">{html.escape(str(row["response"]))}</div><p>{html.escape(str(row["explanation"]))}</p>')
        render_html_panel("example-panel poor", "".join(contents))
    st.markdown("### 参加者情報と同意")
    st.write("モデル名は表示されません。氏名と回答は研究目的で保存し、研究担当者だけが取り扱います。")
    with st.form("survey_start_form"):
        full_name = st.text_input("氏名", placeholder="例: 山田 太郎")
        consent = st.checkbox("説明を読み、氏名と回答を研究目的で保存・利用することに同意します。")
        submitted = st.form_submit_button("評価を開始", type="primary")
    if not submitted:
        return
    if not full_name.strip() or not consent:
        st.error("氏名の入力と同意確認が必要です。")
        return
    try:
        participant, _ = assign_participant(database, definition, full_name, requested_experiment=requested)
    except (OSError, ValueError) as exc:
        st.error(f"参加者情報を保存できませんでした: {exc}")
        return
    saved = load_participant_responses(database, definition, participant.participant_id)
    st.session_state.generic_survey_participant = participant_to_dict(participant)
    st.session_state.generic_survey_index = first_unanswered(experiments[participant.experiment], saved)
    st.rerun()


def render_evaluation(database: Path, definition: dict[str, Any], participant: Participant, items: list[dict[str, Any]]) -> None:
    saved_all = load_participant_responses(database, definition, participant.participant_id)
    total = len(items)
    index = max(0, min(int(st.session_state.get("generic_survey_index", first_unanswered(items, saved_all))), total))
    if index >= total:
        st.title("回答ありがとうございました")
        st.success(f"{total}件すべての回答を保存しました。")
        if st.button("最後の評価を確認する"):
            st.session_state.generic_survey_index = total - 1
            st.rerun()
        return
    reset_evaluation_scroll()
    item = items[index]
    item_id = str(item["item_id"])
    saved = saved_all.get(item_id)
    st.title(definition["evaluation_title"])
    st.progress(len(set(saved_all).intersection(str(row["item_id"]) for row in items)) / total)
    st.caption(f"評価 {index + 1} / {total}　保存済み {len(saved_all)} / {total}")
    notice = st.container(key="evaluation_validation", border=False)
    with st.form(f"evaluation_{participant.participant_id}_{item_id}"):
        reference_column, rating_column = st.columns([1.08, 0.92], gap="large")
        with reference_column:
            st.markdown(
                build_reference_html(item, definition),
                unsafe_allow_html=True,
            )
        scroll = rating_column.container(height=500, border=False, key=f"rating_scroll_container_{item_id}", autoscroll=False)
        with scroll:
            render_html_panel("rating-guide", "各質問について応答A〜Cをそれぞれ1〜7で評価してください。1は『全く当てはまらない』、4は『どちらともいえない』、7は『非常によく当てはまる』です。")
            ratings: dict[str, dict[str, int | None]] = {}
            for question, statement in enumerate(item["likert_statements"], start=1):
                axis = str(statement["key"])
                st.markdown(f'<h4 class="axis-heading">質問 {question}</h4>', unsafe_allow_html=True)
                st.write(str(statement["statement"]))
                ratings[axis] = {}
                for position in RESPONSE_POSITIONS:
                    previous = None
                    if saved:
                        previous = saved.get("ratings", {}).get(axis, {}).get(position)
                    ratings[axis][position] = st.radio(f"応答{position}", list(range(1, 8)), index=int(previous) - 1 if previous else None, horizontal=True, key=f"rating_{participant.participant_id}_{item_id}_{axis}_{position}")
                st.divider()
            previous_choice = str(saved.get("final_choice") or "") if saved else ""
            st.markdown("#### 最後の質問")
            final_choice = st.radio(
                str(definition["final_choice_question"]),
                list(FINAL_CHOICES),
                index=(
                    FINAL_CHOICES.index(previous_choice)
                    if previous_choice in FINAL_CHOICES
                    else None
                ),
                horizontal=True,
                key=f"choice_{participant.participant_id}_{item_id}",
            )
            final_choice_reason = st.text_area(
                FINAL_CHOICE_REASON_QUESTION,
                value=(
                    str(saved.get("final_choice_reason") or "")
                    if saved
                    else ""
                ),
                placeholder="選んだ応答のどこが良かったか、他の応答と何が違ったかを書いてください。",
                height=100,
                key=f"choice_reason_{participant.participant_id}_{item_id}",
            )
            comment = st.text_area("この評価についてのコメント（任意）", value=str(saved.get("comment") or "") if saved else "")
        navigation = st.container(key="evaluation_navigation", border=False)
        with navigation:
            previous_column, next_column = st.columns([1, 2])
            with previous_column:
                previous_button = st.form_submit_button("前の評価へ", disabled=index == 0, use_container_width=True)
            with next_column:
                next_button = st.form_submit_button("保存して次の評価へ", type="primary", use_container_width=True)
    if previous_button:
        st.session_state.generic_survey_index = max(0, index - 1)
        st.rerun()
    if not next_button:
        return
    missing = [f"質問{number + 1}・応答{position}" for number, axis in enumerate(axis_keys(definition)) for position in RESPONSE_POSITIONS if ratings[axis][position] is None]
    if final_choice is None:
        missing.append("最後の質問")
    if not final_choice_reason.strip():
        missing.append("選んだ理由")
    if missing:
        notice.warning("未回答があります: " + "、".join(missing[:8]))
        return
    completed = {axis: {position: int(ratings[axis][position]) for position in RESPONSE_POSITIONS} for axis in axis_keys(definition)}
    try:
        save_response(
            database,
            definition,
            participant=participant,
            item_id=item_id,
            ratings=completed,
            final_choice=str(final_choice),
            final_choice_reason=final_choice_reason,
            comment=comment,
        )
    except (OSError, ValueError) as exc:
        notice.error(f"回答を保存できませんでした: {exc}")
        return
    st.session_state.generic_survey_index = index + 1
    st.rerun()


def main() -> None:
    args = parse_args()
    definition = load_definition(args.definition)
    st.set_page_config(page_title=definition["page_title"], page_icon=None, layout="wide", initial_sidebar_state="collapsed")
    apply_page_style()
    try:
        experiments = load_public_experiments(args.form_root, definition)
        requested = requested_experiment()
    except (OSError, ValueError) as exc:
        st.error(f"評価データを読み込めませんでした: {exc}")
        st.stop()
    if "generic_survey_participant" not in st.session_state:
        render_start(args.database, definition, experiments, requested)
        return
    participant = Participant(**st.session_state.generic_survey_participant)
    if requested and participant.experiment != requested:
        st.error(f"このブラウザでは実験{participant.experiment}を回答中です。")
        st.stop()
    render_evaluation(args.database, definition, participant, experiments[participant.experiment])


if __name__ == "__main__":
    main()
