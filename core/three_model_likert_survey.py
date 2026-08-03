"""データセット非依存の3モデルLikert人手評価とSQLite保存。"""

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

import yaml


EXPERIMENTS = ("A", "B")
RESPONSE_POSITIONS = ("A", "B", "C")
LIKERT_MIN = 1
LIKERT_MAX = 7
FINAL_CHOICES = ("応答A", "応答B", "応答C", "ほぼ同じ", "判断できない")
FINAL_CHOICE_REASON_QUESTION = "そう選んだ理由を教えてください。"
LEGACY_FINAL_CHOICE_REASON = "理由なし"


@dataclass(frozen=True)
class Participant:
    """参加者の保存済み割当。"""

    participant_id: str
    full_name: str
    experiment: str
    consented_at: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} が不正なJSONです。") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} がobjectではありません。")
            rows.append(row)
    return rows


def load_definition(path: Path) -> dict[str, Any]:
    definition = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"dataset", "survey_version", "page_title", "evaluation_title", "intro", "style_features", "example", "axes", "final_choice_question"}
    missing = required - set(definition or {})
    if missing:
        raise ValueError(f"人手評価configが不足しています: {sorted(missing)}")
    axes = definition["axes"]
    keys = [str(axis.get("key") or "") for axis in axes]
    if not 5 <= len(keys) <= 10 or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("評価軸は重複のない5〜10項目にしてください。")
    if len(definition["style_features"]) != 3:
        raise ValueError("会話スタイルの説明は3項目にしてください。")
    selection = definition.get("selection")
    if selection is not None:
        if not isinstance(selection, dict):
            raise ValueError("selectionはmappingで指定してください。")
        oracle_axes = selection.get("oracle_axes")
        if (
            not isinstance(oracle_axes, list)
            or not oracle_axes
            or any(not str(axis).strip() for axis in oracle_axes)
            or len({str(axis) for axis in oracle_axes}) != len(oracle_axes)
        ):
            raise ValueError("selection.oracle_axesは重複のない1項目以上にしてください。")
        min_axis_wins = int(selection.get("min_axis_wins", 1))
        if not 1 <= min_axis_wins <= len(oracle_axes):
            raise ValueError("selection.min_axis_winsがoracle_axes数の範囲外です。")
        max_similarity = float(selection.get("max_pairwise_text_similarity", 0.92))
        if not 0.0 <= max_similarity < 1.0:
            raise ValueError(
                "selection.max_pairwise_text_similarityは0以上1未満です。"
            )
        if not isinstance(selection.get("require_all_strata", True), bool):
            raise ValueError("selection.require_all_strataは真偽値で指定してください。")
        for key in ("stratum_penalty", "conversation_penalty"):
            if float(selection.get(key, 0.0)) < 0.0:
                raise ValueError(f"selection.{key}は0以上にしてください。")
        exclusions = selection.get("human_review_exclusions", [])
        if not isinstance(exclusions, list):
            raise ValueError(
                "selection.human_review_exclusionsはlistで指定してください。"
            )
        exclusion_ids = []
        for row in exclusions:
            if (
                not isinstance(row, dict)
                or not str(row.get("sample_id") or "").strip()
                or not str(row.get("reason") or "").strip()
            ):
                raise ValueError(
                    "human_review_exclusionsにはsample_idとreasonが必要です。"
                )
            exclusion_ids.append(str(row["sample_id"]))
        if len(exclusion_ids) != len(set(exclusion_ids)):
            raise ValueError("human_review_exclusionsのsample_idが重複しています。")
    return definition


def axis_keys(definition: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(axis["key"]) for axis in definition["axes"])


def validate_public_item(item: dict[str, Any], definition: dict[str, Any]) -> None:
    required = {"item_id", "item_number", "conversation", "response_a", "response_b", "response_c", "likert_statements", "final_choice_question", "final_choice_options"}
    missing = required - set(item)
    if missing:
        raise ValueError(f"公開itemに項目が不足しています: {sorted(missing)}")
    forbidden = {"position_to_model", "sample_id", "oracle_axis_scores", "basis_advantage", "model_name"}
    leaked = sorted(forbidden.intersection(item))
    if leaked:
        raise ValueError(f"公開itemに非公開情報が含まれます: {leaked}")
    if any(not str(item[f"response_{position.lower()}"]).strip() for position in RESPONSE_POSITIONS):
        raise ValueError(f"{item['item_id']}: 空の応答があります。")
    keys = tuple(str(row.get("key") or "") for row in item["likert_statements"])
    if keys != axis_keys(definition):
        raise ValueError(f"{item['item_id']}: 評価軸または順序が不正です。")
    if tuple(item["final_choice_options"]) != FINAL_CHOICES:
        raise ValueError(f"{item['item_id']}: 最終選択肢が不正です。")


def load_public_experiments(form_root: Path, definition: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    manifest = json.loads((form_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != definition["dataset"] or manifest.get("survey_version") != definition["survey_version"]:
        raise ValueError("公開itemと評価configのdataset/versionが一致しません。")
    configured_experiments = tuple(
        str(value).upper()
        for value in manifest.get("experiments", EXPERIMENTS)
    )
    if (
        not configured_experiments
        or any(value not in EXPERIMENTS for value in configured_experiments)
        or len(set(configured_experiments)) != len(configured_experiments)
    ):
        raise ValueError("manifest.experimentsは重複のないA/Bで指定してください。")
    experiments = {}
    expected = manifest.get("items_per_experiment")
    for experiment in configured_experiments:
        path = form_root / f"experiment_{experiment.lower()}" / "form_items_public.jsonl"
        rows = read_jsonl(path)
        if expected is not None and len(rows) != int(expected):
            raise ValueError(f"実験{experiment}の件数が不正です: {len(rows)}/{expected}")
        ids = [str(row.get("item_id") or "") for row in rows]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"実験{experiment}のitem_idが空または重複しています。")
        for row in rows:
            validate_public_item(row, definition)
        experiments[experiment] = sorted(rows, key=lambda row: int(row["item_number"]))
    return experiments


def normalize_full_name(name: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", name).strip().split())


def name_key(name: str) -> str:
    return hashlib.sha256(normalize_full_name(name).casefold().encode()).hexdigest()


def connect_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_database(path: Path, definition: dict[str, Any]) -> None:
    with connect_database(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS survey_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS participants (
              participant_id TEXT PRIMARY KEY, name_key TEXT NOT NULL UNIQUE,
              full_name TEXT NOT NULL, experiment TEXT NOT NULL CHECK (experiment IN ('A','B')),
              consented_at TEXT NOT NULL, created_at TEXT NOT NULL, last_seen_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS responses (
              participant_id TEXT NOT NULL, experiment TEXT NOT NULL CHECK (experiment IN ('A','B')),
              item_id TEXT NOT NULL, ratings_json TEXT NOT NULL, final_choice TEXT NOT NULL,
              final_choice_reason TEXT NOT NULL DEFAULT '理由なし',
              comment TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              PRIMARY KEY (participant_id,item_id),
              FOREIGN KEY (participant_id) REFERENCES participants(participant_id) ON DELETE CASCADE);
            """
        )
        response_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(responses)").fetchall()
        }
        if "final_choice_reason" not in response_columns:
            connection.execute(
                "ALTER TABLE responses ADD COLUMN final_choice_reason "
                "TEXT NOT NULL DEFAULT '理由なし'"
            )
        expected = {"survey_version": definition["survey_version"], "dataset": definition["dataset"]}
        for key, value in expected.items():
            row = connection.execute("SELECT value FROM survey_metadata WHERE key=?", (key,)).fetchone()
            if row and row["value"] != value:
                raise ValueError(f"回答DBの{key}が一致しません: {row['value']} != {value}")
            connection.execute("INSERT OR IGNORE INTO survey_metadata(key,value) VALUES (?,?)", (key, value))


def assign_participant(path: Path, definition: dict[str, Any], full_name: str, *, requested_experiment: str | None = None) -> tuple[Participant, bool]:
    normalized = normalize_full_name(full_name)
    if not normalized:
        raise ValueError("氏名を入力してください。")
    if requested_experiment is not None and requested_experiment not in EXPERIMENTS:
        raise ValueError("実験指定はAまたはBです。")
    initialize_database(path, definition)
    now = utc_now()
    with connect_database(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT participant_id,full_name,experiment,consented_at FROM participants WHERE name_key=?", (name_key(normalized),)).fetchone()
        if row:
            if requested_experiment and row["experiment"] != requested_experiment:
                raise ValueError(f"この氏名は実験{row['experiment']}へ割当済みです。")
            connection.execute("UPDATE participants SET last_seen_at=? WHERE participant_id=?", (now, row["participant_id"]))
            return Participant(**dict(row)), False
        counts = {name: connection.execute("SELECT COUNT(*) count FROM participants WHERE experiment=?", (name,)).fetchone()["count"] for name in EXPERIMENTS}
        experiment = requested_experiment or ("A" if counts["A"] <= counts["B"] else "B")
        participant = Participant(f"p_{uuid.uuid4().hex}", normalized, experiment, now)
        connection.execute("INSERT INTO participants VALUES (?,?,?,?,?,?,?)", (participant.participant_id, name_key(normalized), normalized, experiment, now, now, now))
        return participant, True


def validate_ratings(ratings: dict[str, dict[str, int]], definition: dict[str, Any]) -> None:
    if tuple(ratings) != axis_keys(definition):
        raise ValueError("評価軸が不足しているか、順序が不正です。")
    for axis, values in ratings.items():
        if tuple(values) != RESPONSE_POSITIONS:
            raise ValueError(f"{axis}: 応答A/B/Cが不足しています。")
        if any(isinstance(value, bool) or not isinstance(value, int) or not LIKERT_MIN <= value <= LIKERT_MAX for value in values.values()):
            raise ValueError(f"{axis}: 評価値は1〜7の整数です。")


def save_response(
    path: Path,
    definition: dict[str, Any],
    *,
    participant: Participant,
    item_id: str,
    ratings: dict[str, dict[str, int]],
    final_choice: str,
    final_choice_reason: str,
    comment: str,
) -> bool:
    validate_ratings(ratings, definition)
    if final_choice not in FINAL_CHOICES:
        raise ValueError("最終選択が不正です。")
    normalized_reason = final_choice_reason.strip()
    if not normalized_reason:
        raise ValueError("最終選択の理由を入力してください。")
    initialize_database(path, definition)
    now = utc_now()
    with connect_database(path) as connection:
        existing = connection.execute("SELECT 1 FROM responses WHERE participant_id=? AND item_id=?", (participant.participant_id, item_id)).fetchone()
        connection.execute(
            """INSERT INTO responses(
                participant_id,experiment,item_id,ratings_json,final_choice,
                final_choice_reason,comment,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(participant_id,item_id) DO UPDATE SET ratings_json=excluded.ratings_json,
            final_choice=excluded.final_choice,
            final_choice_reason=excluded.final_choice_reason,
            comment=excluded.comment,updated_at=excluded.updated_at""",
            (
                participant.participant_id,
                participant.experiment,
                item_id,
                json.dumps(ratings, ensure_ascii=False),
                final_choice,
                normalized_reason,
                comment.strip(),
                now,
                now,
            ),
        )
    return existing is None


def load_participant_responses(path: Path, definition: dict[str, Any], participant_id: str) -> dict[str, dict[str, Any]]:
    initialize_database(path, definition)
    with connect_database(path) as connection:
        rows = connection.execute("SELECT * FROM responses WHERE participant_id=? ORDER BY item_id", (participant_id,)).fetchall()
    return {row["item_id"]: {**dict(row), "ratings": json.loads(row["ratings_json"])} for row in rows}


def export_responses_csv(path: Path, definition: dict[str, Any], output: Path) -> int:
    initialize_database(path, definition)
    with connect_database(path) as connection:
        rows = connection.execute("SELECT p.full_name,p.participant_id,p.experiment,p.consented_at,r.* FROM responses r JOIN participants p ON p.participant_id=r.participant_id ORDER BY p.created_at,r.item_id").fetchall()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "full_name",
        "participant_id",
        "experiment",
        "item_id",
        "axis_key",
        "response_position",
        "rating",
        "final_choice",
        "final_choice_reason",
        "comment",
        "created_at",
        "updated_at",
    ]
    written = 0
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for row in rows:
            ratings = json.loads(row["ratings_json"])
            for axis in axis_keys(definition):
                for position in RESPONSE_POSITIONS:
                    writer.writerow(
                        {
                            "dataset": definition["dataset"],
                            "full_name": row["full_name"],
                            "participant_id": row["participant_id"],
                            "experiment": row["experiment"],
                            "item_id": row["item_id"],
                            "axis_key": axis,
                            "response_position": position,
                            "rating": ratings[axis][position],
                            "final_choice": row["final_choice"],
                            "final_choice_reason": row["final_choice_reason"],
                            "comment": row["comment"],
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"],
                        }
                    )
                    written += 1
    return written


def participant_to_dict(participant: Participant) -> dict[str, str]:
    return asdict(participant)
