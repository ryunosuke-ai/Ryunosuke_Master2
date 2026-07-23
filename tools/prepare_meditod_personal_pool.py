"""MediTOD用の個人健康相談候補を統合し、DPO再開成果物を監査する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from core.mathdial_basis import load_yaml
from tools.wildchat_health import (
    HEALTH_FILTER_VERSION,
    is_personal_health_consultation,
)

AUDIT_VERSION = "meditod_dpo_resume_audit.v1"


def read_jsonl(path: Path | str, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    source = Path(path)
    if missing_ok and not source.exists():
        return []
    return [
        json.loads(line)
        for line in source.open(encoding="utf-8")
        if line.strip()
    ]


def write_jsonl(rows: list[dict[str, Any]], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)


def record_key(row: dict[str, Any], *, source: bool) -> tuple[str, int]:
    id_key = "conversation_id" if source else "source_dialogue_id"
    return str(row[id_key]), int(row["turn_index"])


def merge_personal_candidates(
    conversation_paths: list[Path | str],
    candidate_paths: list[Path | str],
    *,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    """複数候補集合を個人健康相談だけに絞って重複統合する。"""
    if len(conversation_paths) != len(candidate_paths):
        raise ValueError("conversationsとcandidatesの指定数が一致しません。")
    allowed_ids: set[str] = set()
    total_conversations = personal_conversations = 0
    for path in conversation_paths:
        for conversation in read_jsonl(path):
            total_conversations += 1
            if not is_personal_health_consultation(conversation, config):
                continue
            allowed_ids.add(str(conversation["conversation_id"]))
            personal_conversations += 1

    merged: dict[tuple[str, int], dict[str, Any]] = {}
    rejected_records = 0
    for path in candidate_paths:
        for row in read_jsonl(path):
            if str(row["conversation_id"]) not in allowed_ids:
                rejected_records += 1
                continue
            metadata = dict(row.get("metadata", {}))
            metadata.update(
                {
                    "personal_health_consultation": True,
                    "health_filter_version": HEALTH_FILTER_VERSION,
                }
            )
            enriched = {**row, "metadata": metadata}
            merged.setdefault(record_key(enriched, source=True), enriched)
    rows = list(merged.values())
    report = {
        "health_filter_version": HEALTH_FILTER_VERSION,
        "source_conversations": total_conversations,
        "personal_conversations": len(allowed_ids),
        "personal_conversation_rows_seen": personal_conversations,
        "personal_candidate_records": len(rows),
        "rejected_candidate_records": rejected_records,
    }
    return rows, allowed_ids, report


def is_retryable_fidelity_error(row: dict[str, Any]) -> bool:
    """旧医療情報保持検査だけが原因のsample errorを再処理対象にする。"""
    if row.get("skip_reason") != "sample_error":
        return False
    message = str(row.get("error_message", ""))
    return (
        "MediTOD翻訳で医療情報が失われました" in message
        or "MediTOD医療情報保持の再翻訳に失敗しました" in message
    )


def audit_resume_records(
    *,
    accepted: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    allowed_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """採択済みをdomain監査し、旧fidelity失敗だけを再試行可能に戻す。"""
    kept_accepted: list[dict[str, Any]] = []
    quarantined_accepted: list[dict[str, Any]] = []
    for row in accepted:
        target = (
            kept_accepted
            if str(row.get("source_dialogue_id", "")) in allowed_ids
            else quarantined_accepted
        )
        target.append(row)

    kept_skipped: list[dict[str, Any]] = []
    retryable_fidelity: list[dict[str, Any]] = []
    quarantined_skipped: list[dict[str, Any]] = []
    for row in skipped:
        if str(row.get("source_dialogue_id", "")) not in allowed_ids:
            quarantined_skipped.append(row)
        elif is_retryable_fidelity_error(row):
            retryable_fidelity.append(row)
        else:
            kept_skipped.append(row)
    return {
        "accepted": kept_accepted,
        "accepted_quarantine": quarantined_accepted,
        "skipped": kept_skipped,
        "fidelity_retry": retryable_fidelity,
        "skipped_quarantine": quarantined_skipped,
    }


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_accumulated_quarantine(
    rows: list[dict[str, Any]],
    path: Path,
) -> int:
    """監査履歴を失わないよう既存quarantineへsource key単位で追記統合する。"""
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_jsonl(path, missing_ok=True) + rows:
        if "source_dialogue_id" not in row or "turn_index" not in row:
            continue
        merged[record_key(row, source=False)] = row
    write_jsonl(list(merged.values()), path)
    return len(merged)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MediTOD用の個人健康相談候補を統合し、既存DPOを監査します。"
    )
    parser.add_argument("--config", default="configs/datasets/wildchat_health.yaml")
    parser.add_argument("--conversations", action="append", required=True)
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--accepted")
    parser.add_argument("--skipped")
    parser.add_argument("--quarantine-dir")
    args = parser.parse_args()

    rows, allowed_ids, report = merge_personal_candidates(
        [Path(value) for value in args.conversations],
        [Path(value) for value in args.candidates],
        config=load_yaml(args.config),
    )
    write_jsonl(rows, args.output)
    report["audit_version"] = AUDIT_VERSION

    if args.accepted and args.skipped and args.quarantine_dir:
        accepted_path = Path(args.accepted)
        skipped_path = Path(args.skipped)
        quarantine = Path(args.quarantine_dir)
        audit = audit_resume_records(
            accepted=read_jsonl(accepted_path, missing_ok=True),
            skipped=read_jsonl(skipped_path, missing_ok=True),
            allowed_ids=allowed_ids,
        )
        # 隔離先を先に永続化する。途中終了しても、元ファイルから除いた
        # レコードが監査記録なしで失われない順序にする。
        accepted_quarantine_total = write_accumulated_quarantine(
            audit["accepted_quarantine"],
            quarantine / "basis_selected_non_personal_quarantine.jsonl",
        )
        fidelity_retry_total = write_accumulated_quarantine(
            audit["fidelity_retry"],
            quarantine / "basis_selected_fidelity_retry_v1.jsonl",
        )
        skipped_quarantine_total = write_accumulated_quarantine(
            audit["skipped_quarantine"],
            quarantine / "basis_skipped_non_personal_quarantine.jsonl",
        )
        write_jsonl(audit["accepted"], accepted_path)
        write_jsonl(audit["skipped"], skipped_path)
        report["resume_audit"] = {
            key: len(value)
            for key, value in audit.items()
        }
        report["resume_audit"].update(
            {
                "accepted_quarantine_total": accepted_quarantine_total,
                "fidelity_retry_total": fidelity_retry_total,
                "skipped_quarantine_total": skipped_quarantine_total,
            }
        )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["output_sha256"] = file_hash(Path(args.output))
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
