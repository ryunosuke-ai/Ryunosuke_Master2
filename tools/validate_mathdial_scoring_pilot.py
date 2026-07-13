"""MathDial本スコアリング前の少数pilot品質を検証する。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from core.transition_bayes_model import load_transition_bayes_model
from tools.jsonl_utils import read_jsonl_records


def summarize_pilot(
    records: list[dict[str, Any]],
    *,
    allowed_observations: set[str],
    required_records: int,
    max_fallback_rate: float,
    max_invalid_rate: float,
    min_observations: int,
) -> dict[str, Any]:
    """pilot出力の分類品質とgate結果を返す。"""
    total = len(records)
    fallback = [row for row in records if row.get("llm_error")]
    invalid = [
        row for row in records
        if row.get("llm_error_kind") == "invalid_observation"
        or row.get("llm_retry") == "invalid_observation_retry"
        or str(row.get("observation", "")) not in allowed_observations
    ]
    valid = [
        row for row in records
        if not row.get("llm_error")
        and str(row.get("observation", "")) in allowed_observations
    ]
    distribution = Counter(str(row["observation"]) for row in valid)
    retry_distribution = Counter(
        str(row.get("llm_retry")) for row in records if row.get("llm_retry")
    )
    fallback_reasons = Counter(
        str(row.get("llm_error_kind", "unknown")) for row in fallback
    )
    fallback_rate = len(fallback) / max(1, total)
    invalid_rate = len(invalid) / max(1, total)
    checks = {
        "record_count": total >= required_records,
        "fallback_rate": fallback_rate <= max_fallback_rate,
        "invalid_or_json_rate": invalid_rate <= max_invalid_rate,
        "observation_diversity": len(distribution) >= min_observations,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "records": total,
        "required_records": required_records,
        "fallback_count": len(fallback),
        "fallback_rate": fallback_rate,
        "max_fallback_rate": max_fallback_rate,
        "invalid_or_json_count": len(invalid),
        "invalid_or_json_rate": invalid_rate,
        "max_invalid_or_json_rate": max_invalid_rate,
        "valid_observation_distribution": dict(sorted(distribution.items())),
        "minimum_valid_observations": min_observations,
        "semantic_retry_distribution": dict(sorted(retry_distribution.items())),
        "semantic_retry_rate": sum(retry_distribution.values()) / max(1, total),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MathDial scoring pilot品質gate")
    parser.add_argument("--input", required=True)
    parser.add_argument("--bayes-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--required-records", type=int, default=200)
    parser.add_argument("--max-fallback-rate", type=float, default=0.01)
    parser.add_argument("--max-invalid-rate", type=float, default=0.01)
    parser.add_argument("--min-observations", type=int, default=2)
    args = parser.parse_args()
    rows, skipped = read_jsonl_records(
        args.input, missing_ok=False, strict=False, label="MathDial scoring pilot"
    )
    model = load_transition_bayes_model(args.bayes_model)
    report = summarize_pilot(
        [row for row in rows if isinstance(row, dict)],
        allowed_observations=set(model.observations),
        required_records=args.required_records,
        max_fallback_rate=args.max_fallback_rate,
        max_invalid_rate=args.max_invalid_rate,
        min_observations=args.min_observations,
    )
    report["malformed_jsonl_lines"] = skipped
    if skipped:
        report["passed"] = False
        report["checks"]["jsonl_integrity"] = False
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[scoring_pilot] "
        f"records={report['records']} fallback={report['fallback_rate']:.4%} "
        f"invalid={report['invalid_or_json_rate']:.4%} "
        f"observations={len(report['valid_observation_distribution'])}",
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit("MathDial scoring pilot品質gateに不合格です。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
