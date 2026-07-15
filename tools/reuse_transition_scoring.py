"""完了済みの状態遷移scoringを入力照合して別runへ再利用する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["conversation_id"]), int(record["turn_index"])


def line_count(path: Path) -> int:
    """非空JSONL行数を数える。"""
    with path.open(encoding="utf-8") as file:
        return sum(bool(line.strip()) for line in file)


def source_signature(record: dict[str, Any]) -> str:
    """prompt/response一致確認用の固定長hashを返す。"""
    payload = json.dumps(
        [str(record.get("prompt", "")), str(record.get("response", ""))],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_signatures(path: Path, minimum_records: int) -> dict[tuple[str, int], str]:
    """優先候補全体を巨大本文を保持せずkey/hashへ変換する。"""
    signatures: dict[tuple[str, int], str] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            key = record_key(row)
            if key in signatures:
                raise ValueError(f"scoring入力prefixに重複keyがあります: {key}")
            signatures[key] = source_signature(row)
    if len(signatures) < minimum_records:
        raise ValueError(
            f"scoring入力件数が期待値未満です: {len(signatures)}/{minimum_records}"
        )
    return signatures


def link_verified(source: Path, target: Path) -> None:
    """既存内容を壊さず検証済み成果物をsymlinkする。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.resolve() == source.resolve():
            return
        if sha256(target) != sha256(source):
            raise ValueError(f"再利用先に内容の異なるファイルがあります: {target}")
        return
    target.symlink_to(source.resolve())


def marker_hash(hashes: dict[str, Any], path: Path) -> str | None:
    """絶対・相対表記の違いを許容してmarker内hashを取得する。"""
    resolved = path.resolve()
    for name, value in hashes.items():
        candidate = Path(name)
        try:
            if candidate.resolve() == resolved:
                return str(value)
        except OSError:
            continue
    return None


def validate_and_reuse_scoring(
    *,
    source_root: Path,
    target_root: Path,
    expected_records: int | None,
) -> dict[str, Any]:
    """候補・モデル・各sampleを照合し、raw scoringだけを複製する。"""
    source_metadata = json.loads(
        (source_root / "run_metadata.json").read_text(encoding="utf-8")
    )
    target_metadata = json.loads(
        (target_root / "run_metadata.json").read_text(encoding="utf-8")
    )
    for path in (("models", "scoring"), ("scoring", "preset"), ("scoring", "preset_version")):
        source_value: Any = source_metadata
        target_value: Any = target_metadata
        for key in path:
            source_value = source_value.get(key) if isinstance(source_value, dict) else None
            target_value = target_value.get(key) if isinstance(target_value, dict) else None
        if source_value != target_value:
            raise ValueError(f"scoring再利用元と再利用先の実験条件が一致しません: {'.'.join(path)}")
    # scoringは粗候補を優先順位付けしたJSONLを入力にするため、その実入力を照合する。
    relative_input = Path("scoring/prioritized_candidates.jsonl")
    relative_model = Path("basis_model/mathdial_transition_compat.json")
    relative_scored = Path("scoring/wildchat_scored_raw.jsonl")
    source_scored = source_root / relative_scored
    source_model = source_root / relative_model
    target_model = target_root / relative_model
    if not source_model.is_file() or not target_model.is_file() or sha256(source_model) != sha256(target_model):
        raise ValueError("scoring再利用元と再利用先のベイズモデルが一致しません。")
    source_candidates = source_root / "wildchat/general_tutoring_candidates.jsonl"
    target_candidates = target_root / "wildchat/general_tutoring_candidates.jsonl"
    if not source_candidates.is_file() or not target_candidates.is_file():
        raise ValueError("scoring再利用に必要なWildChat候補がありません。")
    if source_candidates.resolve() != target_candidates.resolve() and sha256(source_candidates) != sha256(target_candidates):
        raise ValueError("scoring再利用元と再利用先のWildChat候補が一致しません。")
    source_input = source_root / relative_input
    target_input = target_root / relative_input
    if not source_input.is_file():
        raise ValueError(f"再利用元の優先候補がありません: {source_input}")
    link_verified(source_input, target_input)

    continuation_path = source_root / "stage_state/scoring_small_batch_CONTINUATION_SUCCESS.json"
    continuation = (
        json.loads(continuation_path.read_text(encoding="utf-8"))
        if continuation_path.exists()
        else {}
    )
    if expected_records is None and continuation.get("scored_records"):
        expected_records = int(continuation["scored_records"])
    if expected_records is None:
        expected_records = int(
            source_metadata.get("early_stop", {}).get("wildchat_scoring_records", 0)
        )
    if expected_records <= 0 and source_scored.is_file():
        expected_records = sum(
            bool(line.strip())
            for line in source_scored.open(encoding="utf-8")
        )
    if expected_records <= 0:
        raise ValueError("再利用するscoring期待件数を決定できません。")
    expected_by_key = candidate_signatures(source_input, expected_records)
    pilot_report_path = source_root / "scoring/pilot_diagnostics.json"
    pilot_report = json.loads(pilot_report_path.read_text(encoding="utf-8"))
    if not isinstance(pilot_report, dict) or not pilot_report.get("passed"):
        raise ValueError("再利用元scoringのpilot品質gateが合格していません。")
    seen: set[tuple[str, int]] = set()
    with source_scored.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            key = record_key(row)
            expected_signature = expected_by_key.get(key)
            if key in seen or expected_signature is None:
                raise ValueError(f"再利用元scoringのsample keyが不正です: {key}")
            seen.add(key)
            if source_signature(row) != expected_signature:
                raise ValueError(f"再利用元scoringのprompt/responseが候補入力と一致しません: {key}")
            if not isinstance(row.get("state_posteriors"), dict) or not row.get("observation"):
                raise ValueError(f"再利用元scoringのベイズ出力が不足しています: {key}")
    if len(seen) != expected_records:
        raise ValueError(f"再利用元scoring件数が不足しています: {len(seen)}/{expected_records}")

    hashes = continuation.get("hashes", {})
    expected_raw_hash = marker_hash(hashes, source_scored)
    if expected_raw_hash and sha256(source_scored) != expected_raw_hash:
        raise ValueError("再利用元raw scoringが完了markerのhashと一致しません。")
    target_scored = target_root / relative_scored
    link_verified(source_scored, target_scored)
    target_pilot = target_root / "scoring/pilot_diagnostics.json"
    link_verified(pilot_report_path, target_pilot)
    source_enriched = source_root / "scoring/wildchat_scored.jsonl"
    target_enriched = target_root / "scoring/wildchat_scored.jsonl"
    if source_enriched.is_file():
        expected_enriched_hash = marker_hash(hashes, source_enriched)
        if expected_enriched_hash and sha256(source_enriched) != expected_enriched_hash:
            raise ValueError("再利用元enriched scoringが完了markerのhashと一致しません。")
        if line_count(source_enriched) != expected_records:
            raise ValueError("再利用元enriched scoringの件数がrawと一致しません。")
        link_verified(source_enriched, target_enriched)
    report = {
        "source_run": str(source_root),
        "target_run": str(target_root),
        "records": len(seen),
        "source_input_sha256": sha256(source_root / relative_input),
        "target_input_sha256": sha256(target_root / relative_input),
        "model_sha256": sha256(target_root / relative_model),
        "scoring_sha256": sha256(target_scored),
        "scoring_model": target_metadata.get("models", {}).get("scoring"),
        "scoring_preset": target_metadata.get("scoring", {}).get("preset"),
        "scoring_preset_version": target_metadata.get("scoring", {}).get(
            "preset_version"
        ),
        "reuse_method": "symlink",
        "enriched_scoring_reused": source_enriched.is_file(),
    }
    report_path = target_root / "scoring/reuse_scoring_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="完了済みscoringの安全な再利用")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--expected-records", type=int)
    args = parser.parse_args()
    report = validate_and_reuse_scoring(
        source_root=Path(args.source_root).resolve(),
        target_root=Path(args.target_root).resolve(),
        expected_records=args.expected_records,
    )
    print(
        f"[reuse scoring] verified records={report['records']} from {args.source_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
