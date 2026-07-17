"""BASiS vs Randomの人手A/B評価用Streamlitアプリ。"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ITEMS_PATH = Path("artifacts/user_eval/items/user_eval_items.jsonl")
DEFAULT_RESPONSES_DIR = Path("artifacts/user_eval/responses")
EVALUATION_AXES = (
    {
        "key": "emotion_reception",
        "label": "気持ちの受け止め",
        "question": "相談者の気持ちをより受け止めているのはどちらですか。",
        "description": "不安、つらさ、迷いなどを丁寧に受け止めているかを見てください。",
    },
    {
        "key": "advice_timing",
        "label": "助言のタイミング",
        "question": "助言や提案に進むタイミングがより自然なのはどちらですか。",
        "description": "すぐに解決策へ急ぎすぎず、必要な場面では自然に次へ進めているかを見てください。",
    },
    {
        "key": "contextual_response",
        "label": "話への合い方",
        "question": "相手の話に合った聞き返しや整理ができているのはどちらですか。",
        "description": "相手が話した内容を拾い、ずれの少ない受け止めや質問になっているかを見てください。",
    },
    {
        "key": "warmth",
        "label": "温かさ",
        "question": "温かく、相談者が話し続けやすいのはどちらですか。",
        "description": "責めたり突き放したりせず、安心して続けて話せそうかを見てください。",
    },
    {
        "key": "conversation_progress",
        "label": "会話の前進",
        "question": "必要に応じて会話を前に進めているのはどちらですか。",
        "description": "共感にとどまりすぎず、確認や整理によって次の話につながっているかを見てください。",
    },
)


def parse_runtime_args() -> argparse.Namespace:
    """Streamlit実行時の追加引数を読む。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--items", default=DEFAULT_ITEMS_PATH.as_posix())
    parser.add_argument("--responses-dir", default=DEFAULT_RESPONSES_DIR.as_posix())
    parser.add_argument(
        "--items-per-participant",
        type=int,
        default=0,
        help="0なら全件。10や15を指定すると参加者IDごとに決定的に部分集合を割り当てます。",
    )
    args, _ = parser.parse_known_args()
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONLを読む。"""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} をJSONとして読めません: {exc}") from exc
            if isinstance(payload, dict):
                records.append(payload)
    return records


def load_items(path: Path) -> list[dict[str, Any]]:
    """評価itemを表示順で読み込む。"""
    if not path.exists():
        raise FileNotFoundError(path)
    items = read_jsonl(path)
    return sorted(items, key=lambda item: int(item.get("display_index", 0)))


def participant_id_from_name(name: str) -> str:
    """氏名または参加者IDから発表用に使う匿名IDを作る。"""
    digest = hashlib.sha256(name.strip().encode("utf-8")).hexdigest()[:12]
    return f"p_{digest}"


def sanitize_session_id(value: str) -> str:
    """session_idとして使える文字だけ残す。"""
    cleaned = "".join(char for char in value.strip() if char.isalnum() or char in {"_", "-"})
    return cleaned[:80]


def new_session_id() -> str:
    """衝突しにくいsession_idを生成する。"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:16]}"


def response_file_path(responses_dir: Path, participant_id: str, session_id: str) -> Path:
    """セッション別回答ファイルのパスを返す。"""
    return responses_dir / f"{participant_id}_{session_id}.jsonl"


def deterministic_subset(items: list[dict[str, Any]], participant_id: str, count: int) -> list[dict[str, Any]]:
    """参加者IDごとに決定的な部分集合を返す。"""
    if count <= 0 or count >= len(items):
        return items
    digest = hashlib.sha256(participant_id.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    shuffled = list(items)
    import random

    random.Random(seed).shuffle(shuffled)
    selected = shuffled[:count]
    return sorted(selected, key=lambda item: int(item.get("display_index", 0)))


def lock_file(file: Any) -> None:
    """可能ならファイルロックを取る。"""
    try:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
    except Exception:
        return


def unlock_file(file: Any) -> None:
    """可能ならファイルロックを解除する。"""
    try:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    except Exception:
        return


def answer_record_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """同じセッション内の同じitem回答かを判定する。"""
    return (
        str(left.get("session_id") or "") == str(right.get("session_id") or "")
        and str(left.get("item_id") or "") == str(right.get("item_id") or "")
    )


def upsert_answer(record: dict[str, Any], path: Path) -> dict[str, Any]:
    """回答を保存する。同じitemがあれば置換し、なければ追加する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with lock_path.open("a+", encoding="utf-8") as lock:
        lock_file(lock)
        try:
            records = read_jsonl(path) if path.exists() else []
            saved_record = dict(record)
            updated = False
            output_records: list[dict[str, Any]] = []
            for existing in records:
                if answer_record_matches(existing, saved_record):
                    updated = True
                    revision = int(existing.get("revision") or 1) + 1
                    saved_record["revision"] = revision
                    saved_record["created_at"] = (
                        existing.get("created_at")
                        or existing.get("timestamp")
                        or saved_record.get("timestamp")
                    )
                    saved_record["updated_at"] = saved_record.get("timestamp")
                    output_records.append(saved_record)
                else:
                    output_records.append(existing)
            if not updated:
                saved_record["revision"] = 1
                saved_record["created_at"] = saved_record.get("timestamp")
                saved_record["updated_at"] = saved_record.get("timestamp")
                output_records.append(saved_record)

            with temp_path.open("w", encoding="utf-8") as file:
                for output_record in output_records:
                    file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, path)
            return {"updated": updated, "revision": saved_record["revision"]}
        finally:
            if temp_path.exists():
                temp_path.unlink()
            unlock_file(lock)


def load_answer_records_by_item_id(path: Path) -> dict[str, dict[str, Any]]:
    """保存済み回答をitem_idで引けるように読む。重複時は後の行を採用する。"""
    if not path.exists():
        return {}
    answers: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        item_id = str(record.get("item_id") or "").strip()
        if item_id:
            answers[item_id] = record
    return answers


def load_answered_item_ids(path: Path) -> set[str]:
    """保存済み回答のitem_id集合を読む。"""
    return set(load_answer_records_by_item_id(path))


def first_unanswered_index(items: list[dict[str, Any]], answered_ids: set[str]) -> int:
    """最初の未回答indexを返す。全回答済みならlen(items)。"""
    for index, item in enumerate(items):
        if str(item.get("item_id")) not in answered_ids:
            return index
    return len(items)


def apply_page_style() -> None:
    """評価画面用CSSを適用する。"""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1080px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .ue-card {
            border: 1px solid #d9dee8;
            border-radius: 8px;
            padding: 18px 20px;
            background: #ffffff;
            margin: 12px 0;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        .ue-card h3 {
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 1.05rem;
        }
        .ue-card p {
            margin: 0 0 10px 0;
            line-height: 1.75;
        }
        .ue-card ul {
            margin: 8px 0 0 1.1rem;
            padding: 0;
        }
        .ue-card li {
            margin: 6px 0;
            line-height: 1.65;
        }
        .ue-card strong {
            color: #0f172a;
        }
        .ue-muted {
            color: #475569;
            font-size: 0.95rem;
            line-height: 1.7;
        }
        .ue-compact {
            font-size: 0.92rem;
            line-height: 1.6;
        }
        .ue-viewpoints {
            border-left: 4px solid #2563eb;
            background: #f8fafc;
        }
        .ue-note {
            color: #64748b;
            font-size: 0.9rem;
        }
        .ue-prompt {
            font-size: 1.08rem;
            line-height: 1.8;
            white-space: pre-wrap;
        }
        .ue-response {
            font-size: 1.03rem;
            line-height: 1.8;
            white-space: pre-wrap;
        }
        .ue-history-line {
            margin: 7px 0;
            line-height: 1.7;
        }
        .ue-label {
            display: inline-block;
            min-width: 42px;
            color: #334155;
            font-weight: 700;
        }
        @media (max-width: 720px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .ue-card {
                padding: 14px 15px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_card(title: str, body: str) -> None:
    """HTMLカードを表示する。"""
    st.markdown(
        f"""
        <div class="ue-card">
          <h3>{html.escape(title)}</h3>
          <div class="ue-muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bullet_card(title: str, items: tuple[str, ...] | list[str], *, note: str = "") -> None:
    """箇条書きカードを表示する。"""
    list_html = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    note_html = f'<p class="ue-note">{html.escape(note)}</p>' if note else ""
    render_card(title, f"<ul>{list_html}</ul>{note_html}")


def render_paragraph_card(title: str, paragraphs: tuple[str, ...] | list[str]) -> None:
    """段落を分けたカードを表示する。"""
    body = "".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in paragraphs)
    render_card(title, body)


def render_viewpoints_card(*, compact: bool = False) -> None:
    """評価観点カードを表示する。"""
    list_html = "".join(
        "<li><strong>{label}</strong>: {question}</li>".format(
            label=html.escape(str(axis["label"])),
            question=html.escape(str(axis["question"])),
        )
        for axis in EVALUATION_AXES
    )
    class_name = "ue-card ue-viewpoints"
    text_class = "ue-muted ue-compact" if compact else "ue-muted"
    st.markdown(
        f"""
        <div class="{class_name}">
          <h3>評価観点</h3>
          <div class="{text_class}">
            <p>今回の評価観点は以下の5つです。各観点について、Model AとModel Bのどちらがより当てはまるかを選んでください。</p>
            <ul>{list_html}</ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_style_goal_card(*, compact: bool = False) -> None:
    """目的の会話スタイル説明を表示する。"""
    paragraphs = (
        "今回の基準は、カウンセリング場面で見られる相談支援らしい会話スタイルです。",
        "悩みや不安を話している相手に対して、まず気持ちを丁寧に受け止め、相手の言葉に沿って聞き返したり状況を整理したりする応答を重視します。",
        "すぐに正解や解決策を押しつけるのではなく、必要な場面では温かさを保ちながら自然に会話を前へ進めることも大切です。",
    )
    if compact:
        paragraphs = (
            "基準は、カウンセリング場面での相談支援らしい会話スタイルです。",
            "気持ちを受け止め、話に沿って聞き返し、急ぎすぎず自然に会話を進めているかを見てください。",
        )
    body = "".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in paragraphs)
    render_card("目的の会話スタイル", body)


def render_history(history: list[dict[str, Any]]) -> None:
    """会話履歴を表示する。"""
    if not history:
        return
    lines: list[str] = []
    for turn in history:
        speaker = html.escape(str(turn.get("speaker") or ""))
        text = html.escape(str(turn.get("text") or ""))
        lines.append(f'<div class="ue-history-line"><span class="ue-label">{speaker}</span>{text}</div>')
    render_card("これまでの会話", "\n".join(lines))


def render_text_card(title: str, text: str, css_class: str = "ue-response") -> None:
    """テキストカードを表示する。"""
    safe_text = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="ue-card">
          <h3>{html.escape(title)}</h3>
          <div class="{css_class}">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def rating_label(value: int) -> str:
    """5段階評価の表示ラベル。"""
    labels = {
        1: "1 = Aの方がかなり当てはまる",
        2: "2 = Aの方がやや当てはまる",
        3: "3 = どちらも同程度",
        4: "4 = Bの方がやや当てはまる",
        5: "5 = Bの方がかなり当てはまる",
    }
    return labels[value]


def rating_index(saved_answer: dict[str, Any] | None, axis_key: str) -> int | None:
    """保存済みaxis ratingをradio indexへ変換する。"""
    if not saved_answer:
        return None
    axis_ratings = saved_answer.get("axis_ratings")
    raw_rating = None
    if isinstance(axis_ratings, dict):
        raw_rating = axis_ratings.get(axis_key)
    if raw_rating is None:
        raw_rating = saved_answer.get("rating")
    try:
        rating = int(raw_rating)
    except (TypeError, ValueError):
        return None
    if rating < 1 or rating > 5:
        return None
    return rating - 1


def render_start_screen(items_path: Path, responses_dir: Path, all_items: list[dict[str, Any]], items_per: int) -> None:
    """開始画面を表示する。"""
    st.title("BASiS vs Random 応答比較評価")
    st.caption("研究室内で実施する、匿名A/B形式の応答比較評価です。")

    overview_cols = st.columns([1, 1])
    with overview_cols[0]:
        render_paragraph_card(
            "研究の概要",
            (
                "小規模な支援対話コーパスから得た会話スタイルを、学習済みモデルが再現できているかを調べます。",
                "画面には同じプロンプトに対する2つの応答を、Model A / Model Bとして匿名で表示します。",
            ),
        )
    with overview_cols[1]:
        render_paragraph_card(
            "評価の目的",
            (
                "BASiSで選別したデータを用いて学習したモデルの応答が、Randomで選別したモデルより目的スタイルに近いかを確認します。",
                "評価結果は研究目的で集計し、発表用の表やグラフに利用します。",
            ),
        )

    render_style_goal_card()

    info_cols = st.columns([1, 1, 1])
    with info_cols[0]:
        render_bullet_card(
            "操作手順",
            (
                "会話履歴と最新の発話を読む",
                "Model A / Model Bの応答を比較する",
                "5つの評価観点それぞれで5段階評価を入力する",
                "必要であれば任意コメントを書く",
                "保存して次の評価へ進む",
            ),
        )
    with info_cols[1]:
        render_bullet_card(
            "所要時間",
            (
                "30件の場合はおよそ15〜25分",
                "10〜15件に分割される場合もあります",
                "途中で中断する場合はsession_idを控える",
            ),
        )
    with info_cols[2]:
        render_bullet_card(
            "匿名化",
            (
                "評価中はモデル名を表示しない",
                "Model A / Bの順序はランダム化済み",
                "集計時にBASiS基準へ補正する",
            ),
        )

    render_viewpoints_card()
    render_paragraph_card(
        "個人情報の扱い",
        (
            "氏名または参加者IDは、研究室内で回答確認を行うためローカルに保存します。",
            "発表用の集計CSV、Markdownレポート、グラフには個人名を出力しません。",
        ),
    )

    st.info(f"評価item: {items_path.as_posix()} / 回答保存先: {responses_dir.as_posix()}")
    item_count = len(all_items) if items_per <= 0 else min(items_per, len(all_items))
    st.caption(f"このセッションの評価予定件数: {item_count}件")

    with st.form("start_form"):
        participant_name = st.text_input("氏名または参加者ID", placeholder="例: 山田太郎 / lab_member_01")
        resume_session_id = st.text_input("再開用session_id（通常は空欄）")
        consent = st.checkbox("上記の説明を読み、研究目的で回答が集計されることに同意します。")
        submitted = st.form_submit_button("評価を開始")

    if not submitted:
        return
    if not participant_name.strip():
        st.error("氏名または参加者IDを入力してください。")
        return
    if not consent:
        st.error("同意確認にチェックしてください。")
        return

    participant_id = participant_id_from_name(participant_name)
    session_id = sanitize_session_id(resume_session_id) if resume_session_id.strip() else new_session_id()
    if not session_id:
        session_id = new_session_id()
    response_path = response_file_path(responses_dir, participant_id, session_id)
    session_items = deterministic_subset(all_items, participant_id, items_per)
    answered_ids = load_answered_item_ids(response_path)
    st.session_state.user_eval_started = True
    st.session_state.user_eval_participant_name = participant_name.strip()
    st.session_state.user_eval_participant_id = participant_id
    st.session_state.user_eval_session_id = session_id
    st.session_state.user_eval_response_path = response_path.as_posix()
    st.session_state.user_eval_items = session_items
    st.session_state.user_eval_current_index = first_unanswered_index(session_items, answered_ids)
    st.rerun()


def build_answer_record(item: dict[str, Any], axis_ratings: dict[str, int], comment: str) -> dict[str, Any]:
    """回答保存用レコードを作る。"""
    return {
        "participant_name": st.session_state.user_eval_participant_name,
        "participant_id": st.session_state.user_eval_participant_id,
        "session_id": st.session_state.user_eval_session_id,
        "item_id": item["item_id"],
        "display_index": item.get("display_index"),
        "category": item.get("category"),
        "stratum": item.get("stratum"),
        "history": item.get("history", []),
        "prompt": item.get("prompt", ""),
        "model_a_response": item.get("model_a_response", ""),
        "model_b_response": item.get("model_b_response", ""),
        "model_a_source": item.get("model_a_source", ""),
        "model_b_source": item.get("model_b_source", ""),
        "displayed_order": item.get("displayed_order", ""),
        "basis_position": item.get("basis_position", ""),
        "random_position": item.get("random_position", ""),
        "axis_ratings": axis_ratings,
        "comment": comment.strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def render_completion() -> None:
    """完了画面を表示する。"""
    st.title("回答ありがとうございました")
    render_card(
        "評価完了",
        (
            "すべての回答が保存されました。"
            f"<br>session_id: <code>{html.escape(st.session_state.user_eval_session_id)}</code>"
            f"<br>保存先: <code>{html.escape(st.session_state.user_eval_response_path)}</code>"
        ),
    )
    col_back, col_new = st.columns([1, 1])
    with col_back:
        if st.button("前の評価を修正する"):
            items: list[dict[str, Any]] = st.session_state.user_eval_items
            st.session_state.user_eval_current_index = max(0, len(items) - 1)
            st.rerun()
    with col_new:
        new_session = st.button("新しいセッションを開始する")
    if new_session:
        for key in list(st.session_state.keys()):
            if key.startswith("user_eval_"):
                del st.session_state[key]
        st.rerun()


def render_eval_screen() -> None:
    """評価画面を表示する。"""
    items: list[dict[str, Any]] = st.session_state.user_eval_items
    response_path = Path(st.session_state.user_eval_response_path)
    saved_answers = load_answer_records_by_item_id(response_path)
    answered_ids = set(saved_answers)
    current_index = int(
        st.session_state.get(
            "user_eval_current_index",
            first_unanswered_index(items, answered_ids),
        )
    )
    if current_index < 0:
        current_index = 0
    if current_index > len(items):
        current_index = len(items)
    st.session_state.user_eval_current_index = current_index
    if current_index >= len(items):
        render_completion()
        return

    item = items[current_index]
    item_id = str(item.get("item_id") or "")
    saved_answer = saved_answers.get(item_id)
    answered_count = len(answered_ids.intersection({str(record["item_id"]) for record in items}))
    total = len(items)
    st.title("応答比較評価")
    st.progress(answered_count / total if total else 1.0)
    st.caption(
        f"現在の評価: {current_index + 1} / {total}  "
        f"保存済み: {answered_count} / {total}  "
        f"session_id: {st.session_state.user_eval_session_id}"
    )

    render_style_goal_card(compact=True)
    render_viewpoints_card(compact=True)
    render_history(item.get("history", []))
    render_text_card("評価用プロンプト", str(item.get("prompt") or ""), css_class="ue-prompt")

    col_a, col_b = st.columns(2)
    with col_a:
        render_text_card("Model A", str(item.get("model_a_response") or ""))
    with col_b:
        render_text_card("Model B", str(item.get("model_b_response") or ""))

    with st.form(f"answer_form_{item['item_id']}"):
        st.markdown("#### 評価")
        st.caption("各観点について、Model AとModel Bのどちらがより当てはまるかを選んでください。")
        axis_ratings: dict[str, int | None] = {}
        for axis in EVALUATION_AXES:
            axis_key = str(axis["key"])
            st.markdown(f"**{html.escape(str(axis['label']))}**")
            st.caption(str(axis["description"]))
            axis_ratings[axis_key] = st.radio(
                str(axis["question"]),
                options=[1, 2, 3, 4, 5],
                index=rating_index(saved_answer, axis_key),
                format_func=rating_label,
                key=f"axis_rating_{item_id}_{axis_key}",
            )
        comment = st.text_area(
            "任意コメント",
            value=str(saved_answer.get("comment") or "") if saved_answer else "",
            placeholder="判断理由や気になった点があれば入力してください。",
        )
        button_cols = st.columns([1, 1, 2])
        with button_cols[0]:
            previous_submitted = st.form_submit_button("前へ戻る", disabled=current_index == 0)
        with button_cols[1]:
            submitted = st.form_submit_button("保存して次へ", type="primary")

    if previous_submitted:
        st.session_state.user_eval_current_index = max(0, current_index - 1)
        st.rerun()
    if not submitted:
        return
    missing_axes = [
        str(axis["label"])
        for axis in EVALUATION_AXES
        if axis_ratings.get(str(axis["key"])) is None
    ]
    if missing_axes:
        st.error("未入力の評価観点があります: " + "、".join(missing_axes))
        return
    completed_axis_ratings = {
        str(axis["key"]): int(axis_ratings[str(axis["key"])])
        for axis in EVALUATION_AXES
    }
    record = build_answer_record(item, completed_axis_ratings, comment)
    save_result = upsert_answer(record, response_path)
    if save_result["updated"]:
        st.toast("回答を更新しました。", icon="✓")
    else:
        st.toast("回答を保存しました。", icon="✓")
    st.session_state.user_eval_current_index = min(total, current_index + 1)
    st.rerun()


def main() -> None:
    """Streamlit entrypoint。"""
    args = parse_runtime_args()
    items_path = Path(args.items)
    responses_dir = Path(args.responses_dir)
    st.set_page_config(page_title="BASiS vs Random 評価", layout="wide")
    apply_page_style()

    try:
        all_items = load_items(items_path)
    except FileNotFoundError:
        st.error(
            "評価itemが見つかりません。先に "
            "`python3 scripts/prepare_user_eval_items.py` を実行してください。"
        )
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if not st.session_state.get("user_eval_started"):
        render_start_screen(items_path, responses_dir, all_items, int(args.items_per_participant))
    else:
        render_eval_screen()


if __name__ == "__main__":
    main()
