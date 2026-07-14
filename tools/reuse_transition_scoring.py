"""完了済みの状態遷移scoringを入力照合して別runへ再利用する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from tools.score_dialogue_with_transition_bayes_model import (
    read_limited_records,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["conversation_id"]), int(record["turn_index"])


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
    relative_input = Path("wildchat/general_tutoring_candidates.jsonl")
    relative_model = Path("basis_model/mathdial_transition_compat.json")
    relative_scored = Path("scoring/wildchat_scored_raw.jsonl")
    source_model = source_root / relative_model
    target_model = target_root / relative_model
    if not source_model.is_file() or not target_model.is_file() or sha256(source_model) != sha256(target_model):
        raise ValueError("scoring再利用元と再利用先のベイズモデルが一致しません。")
    if expected_records is None:
        expected_records = int(
            source_metadata.get("early_stop", {}).get("wildchat_scoring_records", 0)
        )
    if expected_records <= 0:
        raise ValueError("再利用するscoring期待件数を決定できません。")
    source_expected = read_limited_records(
        source_root / relative_input,
        expected_records,
        include_crossing_conversation=False,
    )
    target_expected = read_limited_records(
        target_root / relative_input,
        expected_records,
        include_crossing_conversation=False,
    )
    if len(source_expected) != expected_records or len(target_expected) != expected_records:
        raise ValueError(
            "scoring入力件数が期待値に一致しません: "
            f"source={len(source_expected)} target={len(target_expected)} expected={expected_records}"
        )
    expected_by_key = {record_key(row): row for row in target_expected}
    for source_row in source_expected:
        target_row = expected_by_key.get(record_key(source_row))
        if target_row is None or any(
            str(source_row.get(field, "")) != str(target_row.get(field, ""))
            for field in ("prompt", "response")
        ):
            raise ValueError("scoring再利用元と再利用先の候補prefixが一致しません。")
    source_scored = source_root / relative_scored
    pilot_report_path = source_root / "scoring/pilot_diagnostics.json"
    pilot_report = json.loads(pilot_report_path.read_text(encoding="utf-8"))
    if not isinstance(pilot_report, dict) or not pilot_report.get("passed"):
        raise ValueError("再利用元scoringのpilot品質gateが合格していません。")
    scored = [
        json.loads(line)
        for line in source_scored.open(encoding="utf-8")
        if line.strip()
    ]
    if len(scored) != expected_records:
        raise ValueError(
            f"再利用元scoring件数が不足しています: {len(scored)}/{expected_records}"
        )
    seen: set[tuple[str, int]] = set()
    for row in scored:
        key = record_key(row)
        expected_row = expected_by_key.get(key)
        if key in seen or expected_row is None:
            raise ValueError(f"再利用元scoringのsample keyが不正です: {key}")
        seen.add(key)
        for field in ("prompt", "response"):
            if str(row.get(field, "")) != str(expected_row.get(field, "")):
                raise ValueError(f"再利用元scoringの{field}が候補入力と一致しません: {key}")
        if not isinstance(row.get("state_posteriors"), dict) or not row.get("observation"):
            raise ValueError(f"再利用元scoringのベイズ出力が不足しています: {key}")
    target_scored = target_root / relative_scored
    target_scored.parent.mkdir(parents=True, exist_ok=True)
    if target_scored.exists() and sha256(target_scored) != sha256(source_scored):
        raise ValueError("再利用先に内容の異なるscoring結果があります。")
    if not target_scored.exists():
        shutil.copy2(source_scored, target_scored)
    target_pilot = target_root / "scoring/pilot_diagnostics.json"
    if not target_pilot.exists():
        shutil.copy2(pilot_report_path, target_pilot)
    report = {
        "source_run": str(source_root),
        "target_run": str(target_root),
        "records": len(scored),
        "source_input_sha256": sha256(source_root / relative_input),
        "target_input_sha256": sha256(target_root / relative_input),
        "model_sha256": sha256(target_root / relative_model),
        "scoring_sha256": sha256(target_scored),
        "scoring_model": target_metadata.get("models", {}).get("scoring"),
        "scoring_preset": target_metadata.get("scoring", {}).get("preset"),
        "scoring_preset_version": target_metadata.get("scoring", {}).get(
            "preset_version"
        ),
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
