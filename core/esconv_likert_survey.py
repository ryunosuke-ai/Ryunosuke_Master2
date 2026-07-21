"""ESConv 3モデルLikertユーザ評価の公開item読込と回答保存。"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SURVEY_VERSION = "esconv_likert_web.v1"
EXPERIMENTS = ("A", "B")
RESPONSE_POSITIONS = ("A", "B", "C")
LIKERT_MIN = 1
LIKERT_MAX = 7
FINAL_CHOICES = ("応答A", "応答B", "応答C", "ほぼ同じ", "判断できない")
EXPECTED_AXIS_KEYS = (
    "style_strength",
    "esconv_tone_similarity",
    "supporter_role_consistency",
    "non_directive_support_style",
    "premature_advice_avoidance",
    "content_preservation",
    "naturalness",
)


@dataclass(frozen=True)
class Participant:
    """参加者の保存済み割当。"""

    participant_id: str
    full_name: str
    experiment: str
    consented_at: str


def utc_now() -> str:
    """UTCのISO日時を返す。"""
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONLを厳密に読む。"""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} が不正なJSONです。") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} がJSON objectではありません。")
            rows.append(row)
    return rows


def validate_public_item(item: dict[str, Any], *, experiment: str) -> None:
    """参加者へ表示してよい公開itemか検証する。"""
    required = {
        "item_id",
        "item_number",
        "conversation",
        "response_a",
        "response_b",
        "response_c",
        "likert_statements",
        "final_choice_question",
        "final_choice_options",
    }
    missing = sorted(required - set(item))
    if missing:
        raise ValueError(f"実験{experiment}の公開itemに項目が不足しています: {missing}")
    forbidden = {
        "position_to_model",
        "prompt_id",
        "oracle_axis_scores",
        "representative_means",
        "basis_advantage_over_best_control",
    }
    leaked = sorted(forbidden.intersection(item))
    if leaked:
        raise ValueError(f"公開itemに非公開情報が含まれます: {leaked}")
    if any(not str(item[f"response_{position.lower()}"]).strip() for position in RESPONSE_POSITIONS):
        raise ValueError(f"{item['item_id']}: 空の応答があります。")
    statements = item["likert_statements"]
    if not isinstance(statements, list):
        raise ValueError(f"{item['item_id']}: likert_statementsがlistではありません。")
    axis_keys = tuple(str(statement.get("key") or "") for statement in statements)
    if axis_keys != EXPECTED_AXIS_KEYS:
        raise ValueError(f"{item['item_id']}: 評価軸または順序が不正です。")
    if tuple(item["final_choice_options"]) != FINAL_CHOICES:
        raise ValueError(f"{item['item_id']}: 最終選択肢が不正です。")


def load_public_experiments(form_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Google Form用公開JSONLから実験A/Bを読む。"""
    experiments: dict[str, list[dict[str, Any]]] = {}
    for experiment in EXPERIMENTS:
        path = (
            form_root
            / f"experiment_{experiment.lower()}"
            / "form_items_public.jsonl"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        if len(rows) != 10:
            raise ValueError(f"実験{experiment}は10件必要です: {len(rows)}")
        ids = [str(row.get("item_id") or "") for row in rows]
        if len(set(ids)) != len(ids) or any(not item_id for item_id in ids):
            raise ValueError(f"実験{experiment}のitem_idが空または重複しています。")
        for row in rows:
            validate_public_item(row, experiment=experiment)
        experiments[experiment] = sorted(
            rows,
            key=lambda row: int(row["item_number"]),
        )
    return experiments


def normalize_full_name(name: str) -> str:
    """氏名表記を照合用に正規化する。"""
    normalized = unicodedata.normalize("NFKC", name)
    return " ".join(normalized.strip().split())


def name_key(name: str) -> str:
    """氏名からDB照合用hashを作る。"""
    normalized = normalize_full_name(name)
    return hashlib.sha256(normalized.casefold().encode("utf-8")).hexdigest()


def connect_database(path: Path) -> sqlite3.Connection:
    """同時回答に耐えるSQLite接続を開く。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_database(path: Path) -> None:
    """回答DBを初期化する。"""
    with connect_database(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS survey_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                name_key TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                experiment TEXT NOT NULL CHECK (experiment IN ('A', 'B')),
                consented_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS responses (
                participant_id TEXT NOT NULL,
                experiment TEXT NOT NULL CHECK (experiment IN ('A', 'B')),
                item_id TEXT NOT NULL,
                ratings_json TEXT NOT NULL,
                final_choice TEXT NOT NULL,
                comment TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (participant_id, item_id),
                FOREIGN KEY (participant_id)
                    REFERENCES participants(participant_id) ON DELETE CASCADE
            );
            """
        )
        existing = connection.execute(
            "SELECT value FROM survey_metadata WHERE key = 'survey_version'"
        ).fetchone()
        if existing and existing["value"] != SURVEY_VERSION:
            raise ValueError(
                "回答DBのsurvey versionが一致しません: "
                f"{existing['value']} != {SURVEY_VERSION}"
            )
        connection.execute(
            "INSERT OR IGNORE INTO survey_metadata(key, value) VALUES (?, ?)",
            ("survey_version", SURVEY_VERSION),
        )


def assign_participant(
    path: Path,
    full_name: str,
    *,
    requested_experiment: str | None = None,
) -> tuple[Participant, bool]:
    """氏名を既存割当に戻すか、指定または人数の少ない実験へ割り当てる。"""
    normalized_name = normalize_full_name(full_name)
    if not normalized_name:
        raise ValueError("氏名を入力してください。")
    if requested_experiment is not None and requested_experiment not in EXPERIMENTS:
        raise ValueError(f"実験指定が不正です: {requested_experiment}")
    initialize_database(path)
    current_time = utc_now()
    with connect_database(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT participant_id, full_name, experiment, consented_at "
            "FROM participants WHERE name_key = ?",
            (name_key(normalized_name),),
        ).fetchone()
        if existing:
            if (
                requested_experiment is not None
                and existing["experiment"] != requested_experiment
            ):
                raise ValueError(
                    f"この氏名は実験{existing['experiment']}へ割当済みです。"
                    f"実験{existing['experiment']}用URLを開いてください。"
                )
            connection.execute(
                "UPDATE participants SET last_seen_at = ? WHERE participant_id = ?",
                (current_time, existing["participant_id"]),
            )
            connection.commit()
            return Participant(**dict(existing)), False

        counts = {
            experiment: int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM participants WHERE experiment = ?",
                    (experiment,),
                ).fetchone()["count"]
            )
            for experiment in EXPERIMENTS
        }
        experiment = requested_experiment or (
            "A" if counts["A"] <= counts["B"] else "B"
        )
        participant = Participant(
            participant_id=f"p_{uuid.uuid4().hex}",
            full_name=normalized_name,
            experiment=experiment,
            consented_at=current_time,
        )
        connection.execute(
            """
            INSERT INTO participants(
                participant_id, name_key, full_name, experiment,
                consented_at, created_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                participant.participant_id,
                name_key(normalized_name),
                participant.full_name,
                participant.experiment,
                participant.consented_at,
                current_time,
                current_time,
            ),
        )
        connection.commit()
        return participant, True


def validate_ratings(ratings: dict[str, dict[str, int]]) -> None:
    """7軸x3応答の回答値を検証する。"""
    if tuple(ratings) != EXPECTED_AXIS_KEYS:
        raise ValueError("7つの評価軸が不足しているか、順序が不正です。")
    for axis_key, position_scores in ratings.items():
        if tuple(position_scores) != RESPONSE_POSITIONS:
            raise ValueError(f"{axis_key}: 応答A/B/Cの評価が不足しています。")
        for position, score in position_scores.items():
            if isinstance(score, bool) or not isinstance(score, int):
                raise ValueError(f"{axis_key}/{position}: 評価値が整数ではありません。")
            if score < LIKERT_MIN or score > LIKERT_MAX:
                raise ValueError(f"{axis_key}/{position}: 評価値は1〜7です。")


def save_response(
    path: Path,
    *,
    participant: Participant,
    item_id: str,
    ratings: dict[str, dict[str, int]],
    final_choice: str,
    comment: str,
) -> bool:
    """1itemの回答をupsertし、新規保存ならTrueを返す。"""
    validate_ratings(ratings)
    if final_choice not in FINAL_CHOICES:
        raise ValueError("最終選択が不正です。")
    if not item_id.strip():
        raise ValueError("item_idが空です。")
    initialize_database(path)
    current_time = utc_now()
    ratings_json = json.dumps(ratings, ensure_ascii=False, separators=(",", ":"))
    with connect_database(path) as connection:
        existing = connection.execute(
            "SELECT 1 FROM responses WHERE participant_id = ? AND item_id = ?",
            (participant.participant_id, item_id),
        ).fetchone()
        connection.execute(
            """
            INSERT INTO responses(
                participant_id, experiment, item_id, ratings_json,
                final_choice, comment, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(participant_id, item_id) DO UPDATE SET
                ratings_json = excluded.ratings_json,
                final_choice = excluded.final_choice,
                comment = excluded.comment,
                updated_at = excluded.updated_at
            """,
            (
                participant.participant_id,
                participant.experiment,
                item_id,
                ratings_json,
                final_choice,
                comment.strip(),
                current_time,
                current_time,
            ),
        )
        connection.commit()
    return existing is None


def load_participant_responses(
    path: Path,
    participant_id: str,
) -> dict[str, dict[str, Any]]:
    """参加者の回答をitem_idで返す。"""
    initialize_database(path)
    with connect_database(path) as connection:
        rows = connection.execute(
            """
            SELECT item_id, ratings_json, final_choice, comment,
                   created_at, updated_at
            FROM responses
            WHERE participant_id = ?
            ORDER BY item_id
            """,
            (participant_id,),
        ).fetchall()
    return {
        str(row["item_id"]): {
            "item_id": row["item_id"],
            "ratings": json.loads(row["ratings_json"]),
            "final_choice": row["final_choice"],
            "comment": row["comment"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    }


def export_responses_csv(path: Path, output: Path) -> int:
    """個人情報を含む回答DBを研究者用CSVへ出力する。"""
    initialize_database(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with connect_database(path) as connection:
        rows = connection.execute(
            """
            SELECT p.full_name, p.participant_id, p.experiment,
                   p.consented_at, r.item_id, r.ratings_json,
                   r.final_choice, r.comment, r.created_at, r.updated_at
            FROM responses r
            JOIN participants p ON p.participant_id = r.participant_id
            ORDER BY p.created_at, r.item_id
            """
        ).fetchall()
    fields = (
        "full_name",
        "participant_id",
        "experiment",
        "consented_at",
        "item_id",
        "axis_key",
        "response_position",
        "rating",
        "final_choice",
        "comment",
        "created_at",
        "updated_at",
    )
    written = 0
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            ratings = json.loads(row["ratings_json"])
            common = {
                key: row[key]
                for key in (
                    "full_name",
                    "participant_id",
                    "experiment",
                    "consented_at",
                    "item_id",
                    "final_choice",
                    "comment",
                    "created_at",
                    "updated_at",
                )
            }
            for axis_key in EXPECTED_AXIS_KEYS:
                for position in RESPONSE_POSITIONS:
                    writer.writerow(
                        {
                            **common,
                            "axis_key": axis_key,
                            "response_position": position,
                            "rating": ratings[axis_key][position],
                        }
                    )
                    written += 1
    return written


def participant_to_dict(participant: Participant) -> dict[str, str]:
    """Streamlit sessionへ保存可能な辞書へ変換する。"""
    return asdict(participant)
