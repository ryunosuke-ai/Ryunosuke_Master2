"""ESConv 3モデルLikertユーザ評価用Streamlitアプリ。"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.esconv_likert_survey import (  # noqa: E402
    EXPECTED_AXIS_KEYS,
    FINAL_CHOICES,
    RESPONSE_POSITIONS,
    Participant,
    assign_participant,
    load_participant_responses,
    load_public_experiments,
    participant_to_dict,
    save_response,
)
from scripts.prepare_esconv_google_form_likert_eval import (  # noqa: E402
    EXAMPLE_USER_UTTERANCE,
    GOOD_RESPONSE_EXAMPLE,
    GOOD_RESPONSE_EXPLANATION,
    LIKERT_ANCHORS,
    POOR_RESPONSE_EXAMPLES,
    STYLE_FEATURES,
)


DEFAULT_FORM_ROOT = Path(
    os.environ.get(
        "ESCONV_SURVEY_FORM_ROOT",
        "artifacts/user_eval/google_forms/esconv_human_reviewed_likert_two_forms_v7",
    )
)
DEFAULT_DATABASE = Path(
    os.environ.get(
        "ESCONV_SURVEY_DATABASE",
        "artifacts/user_eval/web/esconv_likert_responses.sqlite3",
    )
)


def parse_runtime_args() -> argparse.Namespace:
    """Streamlitへ渡された追加引数を読む。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--form-root", type=Path, default=DEFAULT_FORM_ROOT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args, _ = parser.parse_known_args()
    return args


def apply_page_style() -> None:
    """比較しやすい固定参照パネルとレスポンシブCSSを適用する。"""
    st.markdown(
        """
        <style>
        .stApp {
            background: #f5f7f9;
            color: #18212b;
        }
        .block-container {
            max-width: 1480px;
            padding-top: 1.5rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3, h4, p, li, label {
            letter-spacing: 0 !important;
        }
        .survey-intro, .example-panel, .reference-panel {
            background: #ffffff;
            border: 1px solid #d7dde3;
            border-radius: 6px;
        }
        .survey-intro {
            padding: 22px 24px;
            margin: 12px 0 18px;
        }
        .survey-intro p, .survey-intro li {
            line-height: 1.75;
        }
        .style-list {
            margin: 12px 0 0 1.25rem;
            padding: 0;
        }
        .style-list li {
            margin: 8px 0;
        }
        .example-panel {
            padding: 18px 20px;
            height: 100%;
        }
        .example-panel.good {
            border-left: 4px solid #237a57;
        }
        .example-panel.poor {
            border-left: 4px solid #9b4b42;
        }
        .example-title {
            font-weight: 700;
            margin-bottom: 8px;
        }
        .example-response {
            background: #f6f8fa;
            border: 1px solid #e2e6ea;
            border-radius: 4px;
            padding: 12px 14px;
            line-height: 1.75;
            margin: 10px 0;
        }
        div[data-testid="stColumn"]:has(.reference-panel) {
            align-self: flex-start;
        }
        [class*="st-key-rating_scroll_container_"] {
            overscroll-behavior: contain;
            scrollbar-gutter: stable;
        }
        .st-key-evaluation_validation {
            position: fixed;
            top: 72px;
            right: 20px;
            left: auto;
            width: min(440px, calc(100vw - 40px));
            z-index: 1100;
            animation: survey-alert-in 180ms ease-out;
        }
        @keyframes survey-alert-in {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .st-key-evaluation_navigation {
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.98);
            border-top: 1px solid #d7dde3;
            box-shadow: 0 -4px 14px rgba(24, 33, 43, 0.08);
            padding: 10px max(18px, calc((100vw - 980px) / 2));
        }
        .st-key-evaluation_navigation
        div[data-testid="stHorizontalBlock"] {
            align-items: center;
        }
        .reference-panel {
            height: auto;
            max-height: none;
            overflow: visible;
            padding: 14px 16px;
            font-size: 0.9rem;
        }
        .reference-heading {
            font-size: 0.86rem;
            color: #52606d;
            font-weight: 700;
            margin: 0 0 5px;
        }
        .conversation-text {
            white-space: pre-wrap;
            line-height: 1.48;
            line-break: strict;
            text-wrap: pretty;
            background: #f7f8fa;
            border-left: 4px solid #6d7782;
            padding: 8px 10px;
            margin-bottom: 7px;
        }
        .response-card {
            border-top: 1px solid #dfe3e7;
            padding: 7px 2px 4px;
        }
        .response-label {
            font-weight: 750;
            margin-bottom: 2px;
        }
        .response-text {
            white-space: pre-wrap;
            line-height: 1.52;
            line-break: strict;
            text-wrap: pretty;
            overflow-wrap: anywhere;
        }
        .rating-guide {
            background: #eef3f7;
            border-left: 4px solid #3f6f8f;
            padding: 12px 14px;
            margin: 8px 0 20px;
            line-height: 1.65;
        }
        .axis-heading {
            margin-top: 18px;
            margin-bottom: 2px;
        }
        div[data-testid="stForm"] {
            border: 0;
            padding: 0;
        }
        div[role="radiogroup"] {
            gap: 0.3rem;
        }
        div[data-testid="stFormSubmitButton"] button {
            min-height: 44px;
        }
        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            div[data-testid="stColumn"]:has(.reference-panel) {
                position: relative;
                top: 0;
            }
            .reference-panel {
                height: auto;
                max-height: none;
                overflow: visible;
                margin-bottom: 20px;
                font-size: 0.95rem;
            }
            .survey-intro {
                padding: 17px;
            }
            .st-key-evaluation_navigation {
                padding: 8px 14px;
            }
            .st-key-evaluation_validation {
                top: 64px;
                right: 12px;
                width: min(400px, calc(100vw - 24px));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_html_panel(css_class: str, content: str) -> None:
    """整形済みの安全なHTMLパネルを表示する。"""
    st.markdown(
        f'<div class="{css_class}">{content}</div>',
        unsafe_allow_html=True,
    )


def render_start_screen(
    database: Path,
    experiments: dict[str, list[dict[str, Any]]],
    requested_experiment: str | None,
) -> None:
    """実験指示、例示、同意、氏名入力を表示する。"""
    st.title("相談支援応答の7段階評価")
    experiment_label = (
        f"実験{requested_experiment}"
        if requested_experiment is not None
        else "実験A/B自動割当"
    )
    st.caption(
        f"{experiment_label}・研究室内で実施する3つの匿名応答の比較評価です。"
    )
    features = "".join(f"<li>{html.escape(feature)}</li>" for feature in STYLE_FEATURES)
    render_html_panel(
        "survey-intro",
        (
            "<h3>実験指示</h3>"
            "<p>本実験では、カウンセリングの場面を想定した会話を"
            "評価してもらいます。ここでいう理想的な会話には、"
            "次のような特徴があります。</p>"
            f'<ul class="style-list">{features}</ul>'
            "<p>評価は全部で10件です。各評価では「これまでの会話」と、"
            "最後の相談者の発話に対する「応答A〜C」を示します。"
            "それぞれの応答について同じ7項目を1〜7で評価し、最後に"
            "最もふさわしい応答を1つ選んでください。</p>"
        ),
    )

    st.subheader("評価例")
    st.markdown(
        f"**相談者:** {html.escape(EXAMPLE_USER_UTTERANCE)}",
        unsafe_allow_html=True,
    )
    good_column, poor_column = st.columns([1, 1], gap="large")
    with good_column:
        render_html_panel(
            "example-panel good",
            (
                '<div class="example-title">良い応答例</div>'
                f'<div class="example-response">{html.escape(GOOD_RESPONSE_EXAMPLE)}</div>'
                f"<p>{html.escape(GOOD_RESPONSE_EXPLANATION)}</p>"
            ),
        )
    with poor_column:
        poor_html = ['<div class="example-title">良くない応答例</div>']
        for index, example in enumerate(POOR_RESPONSE_EXAMPLES, start=1):
            poor_html.extend(
                [
                    f"<strong>例{index}</strong>",
                    (
                        '<div class="example-response">'
                        f"{html.escape(example['response'])}</div>"
                    ),
                    f"<p>{html.escape(example['explanation'])}</p>",
                ]
            )
        render_html_panel("example-panel poor", "".join(poor_html))

    st.markdown("### 参加者情報と同意")
    st.write(
        "モデル名は表示されません。氏名と回答は研究目的で保存し、"
        "研究担当者だけが取り扱います。"
    )
    with st.form("survey_start_form"):
        full_name = st.text_input("氏名", placeholder="例: 山田 太郎")
        consent = st.checkbox(
            "説明を読み、氏名と回答を研究目的で保存・利用することに同意します。"
        )
        submitted = st.form_submit_button("評価を開始", type="primary")
    if not submitted:
        return
    if not full_name.strip():
        st.error("氏名を入力してください。")
        return
    if not consent:
        st.error("同意確認にチェックしてください。")
        return
    try:
        participant, _ = assign_participant(
            database,
            full_name,
            requested_experiment=requested_experiment,
        )
    except (OSError, ValueError) as exc:
        st.error(f"参加者情報を保存できませんでした: {exc}")
        return
    saved = load_participant_responses(database, participant.participant_id)
    st.session_state.esconv_survey_participant = participant_to_dict(participant)
    st.session_state.esconv_survey_index = first_unanswered_index(
        experiments[participant.experiment],
        saved,
    )
    st.rerun()


def participant_from_session() -> Participant:
    """Streamlit sessionから参加者割当を復元する。"""
    payload = st.session_state["esconv_survey_participant"]
    return Participant(**payload)


def requested_experiment_from_query() -> str | None:
    """URL queryから実験A/Bの固定指定を読む。"""
    raw_value = st.query_params.get("experiment")
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        raw_value = raw_value[-1] if raw_value else ""
    experiment = str(raw_value).strip().upper()
    if experiment not in {"A", "B"}:
        raise ValueError("URLのexperimentにはAまたはBを指定してください。")
    return experiment


def readable_text_html(value: Any) -> str:
    """既存改行を保ち、日本語の文末ごとに表示上の改行を加える。"""
    escaped = html.escape(str(value)).replace("\r\n", "\n").replace("\r", "\n")
    escaped = escaped.replace("\n", "<br>")
    escaped = re.sub(r"([。！？!?][」』】）)]*)", r"\1<br>", escaped)
    escaped = re.sub(r"(?:<br>\s*){2,}", "<br>", escaped)
    return re.sub(r"(?:<br>\s*)+$", "", escaped)


def build_reference_html(item: dict[str, Any]) -> str:
    """固定表示する会話と3応答を、モデル情報なしでHTML化する。"""
    conversation = readable_text_html(item["conversation"])
    parts = [
        '<div class="reference-panel">',
        '<div class="reference-heading">これまでの会話</div>',
        f'<div class="conversation-text">{conversation}</div>',
    ]
    for position in RESPONSE_POSITIONS:
        response = readable_text_html(item[f"response_{position.lower()}"])
        parts.extend(
            [
                '<div class="response-card">',
                f'<div class="response-label">応答{position}</div>',
                f'<div class="response-text">{response}</div>',
                "</div>",
            ]
        )
    parts.append("</div>")
    return "".join(parts)


def rating_default(
    saved: dict[str, Any] | None,
    axis_key: str,
    position: str,
) -> int | None:
    """保存済みratingがあれば返す。"""
    if not saved:
        return None
    try:
        value = int(saved["ratings"][axis_key][position])
    except (KeyError, TypeError, ValueError):
        return None
    return value if 1 <= value <= 7 else None


def find_missing_evaluation_fields(
    ratings: dict[str, dict[str, int | None]],
    final_choice: str | None,
) -> list[str]:
    """未回答の評価項目を表示順に返す。"""
    missing = [
        f"質問{axis_index + 1}・応答{position}"
        for axis_index, axis_key in enumerate(EXPECTED_AXIS_KEYS)
        for position in RESPONSE_POSITIONS
        if ratings[axis_key][position] is None
    ]
    if final_choice is None:
        missing.append("最後の質問")
    return missing


def first_unanswered_index(
    items: list[dict[str, Any]],
    saved: dict[str, dict[str, Any]],
) -> int:
    """最初の未回答item位置を返す。"""
    for index, item in enumerate(items):
        if str(item["item_id"]) not in saved:
            return index
    return len(items)


def render_completion(participant: Participant, total: int) -> None:
    """完了画面を表示する。"""
    st.title("回答ありがとうございました")
    st.success(f"{total}件すべての回答を保存しました。")
    st.write(
        f"{html.escape(participant.full_name)}さん、ご協力ありがとうございました。"
    )
    previous_column, switch_column = st.columns(2)
    with previous_column:
        if st.button("最後の評価を確認する", use_container_width=True):
            st.session_state.esconv_survey_index = max(0, total - 1)
            st.rerun()
    with switch_column:
        if st.button("別の参加者として開始", use_container_width=True):
            for key in list(st.session_state):
                if key.startswith("esconv_survey_") or key.startswith("rating_"):
                    del st.session_state[key]
            st.rerun()


def reset_evaluation_scroll() -> None:
    """評価画面への遷移時にページ本体を先頭へ戻す。"""
    st.iframe(
        """
        <script>
        const resetScroll = () => {
          window.parent.scrollTo({top: 0, left: 0, behavior: "auto"});
          const documentRoot = window.parent.document;
          documentRoot.querySelector('[data-testid="stAppViewContainer"]')
            ?.scrollTo({top: 0, left: 0, behavior: "auto"});
          documentRoot.querySelector('[data-testid="stMain"]')
            ?.scrollTo({top: 0, left: 0, behavior: "auto"});
        };
        resetScroll();
        window.setTimeout(resetScroll, 100);
        window.setTimeout(resetScroll, 300);
        window.setTimeout(resetScroll, 700);
        </script>
        """,
        height=1,
        width=1,
    )


def render_evaluation(
    *,
    participant: Participant,
    items: list[dict[str, Any]],
    database: Path,
) -> None:
    """会話を固定表示し、右側で7軸x3応答を評価する。"""
    saved_responses = load_participant_responses(database, participant.participant_id)
    total = len(items)
    current_index = int(
        st.session_state.get(
            "esconv_survey_index",
            first_unanswered_index(items, saved_responses),
        )
    )
    current_index = max(0, min(current_index, total))
    if current_index >= total:
        render_completion(participant, total)
        return
    item = items[current_index]
    item_id = str(item["item_id"])
    saved = saved_responses.get(item_id)
    valid_item_ids = {str(candidate["item_id"]) for candidate in items}
    answered_count = len(valid_item_ids.intersection(saved_responses))

    st.markdown(
        '<span class="evaluation-page-marker" aria-hidden="true"></span>',
        unsafe_allow_html=True,
    )
    reset_evaluation_scroll()
    st.title("相談支援応答の評価")
    st.progress(answered_count / total)
    st.caption(
        f"評価 {current_index + 1} / {total}　保存済み {answered_count} / {total}"
    )
    validation_notice = st.container(
        key="evaluation_validation",
        border=False,
    )
    with st.form(f"evaluation_form_{participant.participant_id}_{item_id}"):
        reference_column, rating_column = st.columns([1.08, 0.92], gap="large")
        with reference_column:
            st.markdown(build_reference_html(item), unsafe_allow_html=True)
        rating_scroll = rating_column.container(
            height=500,
            border=False,
            key=f"rating_scroll_container_{item_id}",
            autoscroll=False,
        )
        with rating_scroll:
            render_html_panel(
                "rating-guide",
                (
                    "各質問について、応答A〜Cをそれぞれ1〜7で評価してください。"
                    "1は「全く当てはまらない」、4は「どちらともいえない」、"
                    "7は「非常によく当てはまる」です。"
                ),
            )
            statements = item["likert_statements"]
            ratings: dict[str, dict[str, int | None]] = {}
            for question_number, statement in enumerate(statements, start=1):
                axis_key = str(statement["key"])
                st.markdown(
                    f'<h4 class="axis-heading">質問 {question_number}</h4>',
                    unsafe_allow_html=True,
                )
                st.write(str(statement["statement"]))
                ratings[axis_key] = {}
                for position in RESPONSE_POSITIONS:
                    ratings[axis_key][position] = st.radio(
                        f"応答{position}",
                        options=list(range(1, 8)),
                        index=(
                            rating_default(saved, axis_key, position) - 1
                            if rating_default(saved, axis_key, position) is not None
                            else None
                        ),
                        horizontal=True,
                        key=(
                            f"rating_{participant.participant_id}_{item_id}_"
                            f"{axis_key}_{position}"
                        ),
                    )
                st.divider()

            saved_choice = str(saved.get("final_choice") or "") if saved else ""
            st.markdown("#### 最後の質問")
            final_choice = st.radio(
                str(item["final_choice_question"]),
                options=list(FINAL_CHOICES),
                index=(
                    FINAL_CHOICES.index(saved_choice)
                    if saved_choice in FINAL_CHOICES
                    else None
                ),
                key=f"final_choice_{participant.participant_id}_{item_id}",
            )
            comment = st.text_area(
                "この評価についてのコメント（任意）",
                value=str(saved.get("comment") or "") if saved else "",
                key=f"comment_{participant.participant_id}_{item_id}",
            )
        navigation = st.container(
            key="evaluation_navigation",
            border=False,
        )
        with navigation:
            previous_column, next_column = st.columns([1, 2])
            with previous_column:
                previous = st.form_submit_button(
                    "前の評価へ",
                    disabled=current_index == 0,
                    use_container_width=True,
                )
            with next_column:
                submitted = st.form_submit_button(
                    "保存して次の評価へ",
                    type="primary",
                    use_container_width=True,
                )

    if previous:
        st.session_state.esconv_survey_index = max(0, current_index - 1)
        st.rerun()
    if not submitted:
        return
    missing = find_missing_evaluation_fields(ratings, final_choice)
    if missing:
        validation_notice.warning(
            "未回答があります。右側の質問を確認してください: "
            + "、".join(missing[:8])
        )
        return
    completed_ratings = {
        axis_key: {
            position: int(ratings[axis_key][position])
            for position in RESPONSE_POSITIONS
        }
        for axis_key in EXPECTED_AXIS_KEYS
    }
    try:
        save_response(
            database,
            participant=participant,
            item_id=item_id,
            ratings=completed_ratings,
            final_choice=str(final_choice),
            comment=comment,
        )
    except (OSError, ValueError) as exc:
        validation_notice.error(f"回答を保存できませんでした: {exc}")
        return
    st.session_state.esconv_survey_index = current_index + 1
    st.rerun()


def main() -> None:
    """Streamlit entrypoint。"""
    args = parse_runtime_args()
    st.set_page_config(
        page_title="相談支援応答の7段階評価",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_page_style()
    try:
        experiments = load_public_experiments(args.form_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"評価データを読み込めませんでした: {exc}")
        st.stop()
    try:
        requested_experiment = requested_experiment_from_query()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if "esconv_survey_participant" not in st.session_state:
        render_start_screen(args.database, experiments, requested_experiment)
        return
    participant = participant_from_session()
    if (
        requested_experiment is not None
        and participant.experiment != requested_experiment
    ):
        st.error(
            f"このブラウザでは実験{participant.experiment}を回答中です。"
            f"実験{participant.experiment}用URLへ戻ってください。"
        )
        st.stop()
    render_evaluation(
        participant=participant,
        items=experiments[participant.experiment],
        database=args.database,
    )


if __name__ == "__main__":
    main()
