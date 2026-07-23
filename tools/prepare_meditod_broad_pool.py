"""MediTOD広域健康候補の再利用検証とDPO再開監査。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.mathdial_basis import load_yaml
from tools.translate_and_generate_dpo import (
    MEDITOD_MEDICAL_FIDELITY_VERSION,
    bayes_model_version,
    meditod_translation_fidelity_details,
    meditod_translation_fidelity_errors,
    passes_thresholds,
)
from tools.wildchat_health import health_conversation_diagnostic_category


AUDIT_VERSION = "meditod_broad_pool_resume_audit.v1"


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


def file_hash(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_key(row: dict[str, Any], *, source: bool) -> tuple[str, int]:
    id_key = "conversation_id" if source else "source_dialogue_id"
    return str(row[id_key]), int(row["turn_index"])


def verify_broad_pool_artifacts(
    *,
    config: dict[str, Any],
    conversations: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    manifest: dict[str, Any],
    statistics: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """全走査済み広域候補を再利用できることを検証する。"""
    manifest_statistics = dict(manifest.get("statistics", {}))
    if manifest.get("dataset") != config["dataset_name"]:
        raise ValueError("WildChat広域候補のdatasetが一致しません。")
    if manifest.get("revision") != config["revision"]:
        raise ValueError("WildChat広域候補のrevisionが一致しません。")
    if int(manifest.get("stream_shuffle_seed", -1)) != seed:
        raise ValueError("WildChat広域候補のseedが一致しません。")
    if int(manifest_statistics.get("stream_exhausted", 0)) != 1:
        raise ValueError("WildChat広域候補は全体走査を完了していません。")
    if int(statistics.get("stream_exhausted", 0)) != 1:
        raise ValueError("WildChat統計上で全体走査が完了していません。")
    expected_conversations = int(
        manifest_statistics.get("general_conversations", -1)
    )
    expected_candidates = int(
        manifest_statistics.get("general_candidate_records", -1)
    )
    if len(conversations) != expected_conversations:
        raise ValueError(
            "WildChat広域会話数がmanifestと一致しません: "
            f"{len(conversations)}/{expected_conversations}"
        )
    if len(candidates) != expected_candidates:
        raise ValueError(
            "WildChat広域候補数がmanifestと一致しません: "
            f"{len(candidates)}/{expected_candidates}"
        )
    conversation_ids = {
        str(conversation["conversation_id"]) for conversation in conversations
    }
    orphaned = [
        record_key(row, source=True)
        for row in candidates
        if str(row["conversation_id"]) not in conversation_ids
    ]
    if orphaned:
        raise ValueError(f"会話本体のないWildChat候補があります: {orphaned[:5]}")
    categories = Counter(
        health_conversation_diagnostic_category(conversation, config)
        for conversation in conversations
    )
    candidate_categories = Counter()
    category_by_id = {
        str(conversation["conversation_id"]): (
            health_conversation_diagnostic_category(conversation, config)
        )
        for conversation in conversations
    }
    for row in candidates:
        candidate_categories[category_by_id[str(row["conversation_id"])]] += 1
    return {
        "audit_version": AUDIT_VERSION,
        "dataset": manifest["dataset"],
        "revision": manifest["revision"],
        "seed": seed,
        "stream_rows": int(manifest_statistics.get("stream_rows", 0)),
        "stream_exhausted": True,
        "broad_conversations": len(conversations),
        "broad_candidate_records": len(candidates),
        "diagnostic_only_personal_filter": True,
        "personal_filter_affects_main_eligibility": False,
        "conversation_diagnostic_categories": dict(categories),
        "candidate_diagnostic_categories": dict(candidate_categories),
        "legacy_manifest_filter_config": manifest.get("config", {}),
    }


def _source_prompt_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(str(row.get("prompt", "")).encode("utf-8")).hexdigest()


def _dpo_conditions_match(
    row: dict[str, Any],
    source: dict[str, Any],
    *,
    expected_generation_model: str,
    expected_scoring_model: str,
    expected_bayes_version: str,
    expected_candidates: int,
    min_score_gap: float,
    min_chosen_posterior: float,
    max_rejected_posterior: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    metadata = dict(row.get("metadata", {}))
    if metadata.get("source_prompt_hash") != _source_prompt_hash(source):
        reasons.append("source_prompt_hash")
    if (
        row.get("model_used_for_translation", metadata.get("generation_model"))
        != expected_generation_model
    ):
        reasons.append("generation_model")
    if (
        row.get("model_used_for_scoring", metadata.get("scoring_model"))
        != expected_scoring_model
    ):
        reasons.append("scoring_model")
    if (
        row.get("bayesian_model_version", metadata.get("bayes_model_version"))
        != expected_bayes_version
    ):
        reasons.append("bayes_model_version")
    if int(metadata.get("rejected_candidates", -1)) != expected_candidates:
        reasons.append("rejected_candidates")
    if metadata.get("translated_prompt_hash") != metadata.get(
        "rejected_prompt_hash"
    ):
        reasons.append("context_hash")
    if passes_thresholds(
        row,
        min_score_gap=min_score_gap,
        min_chosen_posterior=min_chosen_posterior,
        max_rejected_posterior=max_rejected_posterior,
    ) is None:
        reasons.append("score_thresholds")
    payload = {
        "translated_prompt": row.get("raw_translated_prompt")
        or metadata.get("raw_translated_prompt", ""),
        "translated_chosen": row.get("translated_chosen") or row.get("chosen", ""),
        "rejected_candidates": [row.get("rejected", "")],
    }
    fidelity_errors = meditod_translation_fidelity_errors(source, payload)
    if fidelity_errors:
        reasons.append("medical_fidelity_v3")
    return not reasons, reasons


def _skip_is_retryable_fidelity(row: dict[str, Any]) -> bool:
    message = str(row.get("error_message", ""))
    return row.get("skip_reason") == "sample_error" and (
        "MediTOD翻訳で医療情報が失われました" in message
        or "MediTOD医療情報保持の再翻訳に失敗しました" in message
    )


def restore_broad_resume_records(
    *,
    sources: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    quarantined_accepted: list[dict[str, Any]],
    quarantined_skipped: list[dict[str, Any]],
    fidelity_retry: list[dict[str, Any]],
    expected_generation_model: str,
    expected_scoring_model: str,
    expected_bayes_version: str,
    expected_candidates: int,
    min_score_gap: float,
    min_chosen_posterior: float,
    max_rejected_posterior: float,
) -> dict[str, Any]:
    """広域方針に合う既存DPO結果を復元し、不一致だけ再処理へ戻す。"""
    source_by_key = {record_key(row, source=True): row for row in sources}
    accepted_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    rejected_rows: list[dict[str, Any]] = []
    restored = 0
    for row in accepted + quarantined_accepted:
        key = record_key(row, source=False)
        source = source_by_key.get(key)
        if source is None:
            rejected_rows.append(
                {**row, "broad_restore_rejection_reasons": ["source_not_in_broad_pool"]}
            )
            continue
        valid, reasons = _dpo_conditions_match(
            row,
            source,
            expected_generation_model=expected_generation_model,
            expected_scoring_model=expected_scoring_model,
            expected_bayes_version=expected_bayes_version,
            expected_candidates=expected_candidates,
            min_score_gap=min_score_gap,
            min_chosen_posterior=min_chosen_posterior,
            max_rejected_posterior=max_rejected_posterior,
        )
        if not valid:
            rejected_rows.append(
                {**row, "broad_restore_rejection_reasons": reasons}
            )
            continue
        restored += int(key not in accepted_by_key and row in quarantined_accepted)
        restored_row = dict(row)
        previous_fidelity_version = restored_row.get("medical_fidelity_version")
        restored_row["medical_fidelity_version"] = MEDITOD_MEDICAL_FIDELITY_VERSION
        restored_row["medical_fidelity_details"] = (
            meditod_translation_fidelity_details(source)
        )
        restored_row.setdefault("metadata", {})[
            "medical_fidelity_version"
        ] = MEDITOD_MEDICAL_FIDELITY_VERSION
        restored_row["metadata"]["previous_medical_fidelity_version"] = (
            previous_fidelity_version
        )
        restored_row["metadata"]["medical_fidelity_details"] = restored_row[
            "medical_fidelity_details"
        ]
        restored_row["metadata"]["resume_migration"] = (
            "target3000_broad_health_fidelity_v3"
        )
        accepted_by_key[key] = restored_row

    skipped_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    requeued_fidelity = 0
    requeued_condition_mismatch = 0
    for row in skipped + quarantined_skipped + fidelity_retry:
        key = record_key(row, source=False)
        if key in accepted_by_key or key not in source_by_key:
            continue
        if _skip_is_retryable_fidelity(row):
            requeued_fidelity += 1
            continue
        source_hash = row.get("source_prompt_hash") or row.get(
            "metadata", {}
        ).get("source_prompt_hash")
        if source_hash and source_hash != _source_prompt_hash(source_by_key[key]):
            requeued_condition_mismatch += 1
            continue
        skipped_by_key[key] = row
    return {
        "accepted": list(accepted_by_key.values()),
        "skipped": list(skipped_by_key.values()),
        "rejected": rejected_rows,
        "report": {
            "accepted_total": len(accepted_by_key),
            "accepted_restored_from_personal_quarantine": restored,
            "accepted_requeued_for_condition_or_fidelity": len(rejected_rows),
            "skipped_total": len(skipped_by_key),
            "fidelity_errors_requeued": requeued_fidelity,
            "condition_mismatch_skips_requeued": requeued_condition_mismatch,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MediTODの広域健康候補とDPO再開成果物を監査します。"
    )
    parser.add_argument("--config", default="configs/datasets/wildchat_health.yaml")
    parser.add_argument("--conversations", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--reuse-manifest", required=True)
    parser.add_argument("--diagnostic-report", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accepted")
    parser.add_argument("--skipped")
    parser.add_argument("--quarantine-dir")
    parser.add_argument("--bayes-model")
    parser.add_argument("--generation-model")
    parser.add_argument("--scoring-model")
    parser.add_argument("--rejected-candidates", type=int, default=4)
    parser.add_argument("--min-score-gap", type=float, default=0.20)
    parser.add_argument("--min-chosen-posterior", type=float, default=0.70)
    parser.add_argument("--max-rejected-posterior", type=float, default=0.55)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if config.get("require_personal_consultation", False):
        raise ValueError("MediTOD主実験ではpersonal consultationを必須化できません。")
    conversation_path = Path(args.conversations)
    candidate_path = Path(args.candidates)
    manifest_path = Path(args.manifest)
    statistics_path = Path(args.statistics)
    conversations = read_jsonl(conversation_path)
    candidates = read_jsonl(candidate_path)
    report = verify_broad_pool_artifacts(
        config=config,
        conversations=conversations,
        candidates=candidates,
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        statistics=json.loads(statistics_path.read_text(encoding="utf-8")),
        seed=args.seed,
    )
    report["artifact_hashes"] = {
        str(path): file_hash(path)
        for path in (
            conversation_path,
            candidate_path,
            manifest_path,
            statistics_path,
        )
    }
    report["verified_at"] = datetime.now(timezone.utc).isoformat()

    if args.accepted or args.skipped or args.quarantine_dir:
        required = (
            args.accepted,
            args.skipped,
            args.quarantine_dir,
            args.bayes_model,
            args.generation_model,
            args.scoring_model,
        )
        if not all(required):
            raise ValueError("DPO再開監査の引数が不足しています。")
        quarantine = Path(args.quarantine_dir)
        restored = restore_broad_resume_records(
            sources=candidates,
            accepted=read_jsonl(args.accepted, missing_ok=True),
            skipped=read_jsonl(args.skipped, missing_ok=True),
            quarantined_accepted=read_jsonl(
                quarantine / "basis_selected_non_personal_quarantine.jsonl",
                missing_ok=True,
            ),
            quarantined_skipped=read_jsonl(
                quarantine / "basis_skipped_non_personal_quarantine.jsonl",
                missing_ok=True,
            ),
            fidelity_retry=read_jsonl(
                quarantine / "basis_selected_fidelity_retry_v1.jsonl",
                missing_ok=True,
            ),
            expected_generation_model=args.generation_model,
            expected_scoring_model=args.scoring_model,
            expected_bayes_version=bayes_model_version(args.bayes_model),
            expected_candidates=args.rejected_candidates,
            min_score_gap=args.min_score_gap,
            min_chosen_posterior=args.min_chosen_posterior,
            max_rejected_posterior=args.max_rejected_posterior,
        )
        write_jsonl(restored["accepted"], args.accepted)
        write_jsonl(restored["skipped"], args.skipped)
        write_jsonl(
            restored["rejected"],
            quarantine / "basis_broad_restore_rejected.jsonl",
        )
        report["resume_restore"] = restored["report"]

    reuse_manifest = Path(args.reuse_manifest)
    reuse_manifest.parent.mkdir(parents=True, exist_ok=True)
    reuse_manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    diagnostic = {
        key: report[key]
        for key in (
            "audit_version",
            "broad_conversations",
            "broad_candidate_records",
            "diagnostic_only_personal_filter",
            "personal_filter_affects_main_eligibility",
            "conversation_diagnostic_categories",
            "candidate_diagnostic_categories",
        )
    }
    if "resume_restore" in report:
        diagnostic["resume_restore"] = report["resume_restore"]
    diagnostic_path = Path(args.diagnostic_report)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
