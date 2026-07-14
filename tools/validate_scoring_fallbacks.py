"""本スコアリングのfallbackを監査し、警告と致命条件を分離する。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def classify_fallback(record: dict[str, Any]) -> str:
    """fallback理由を再現可能な粗分類へまとめる。"""
    error = str(record.get("llm_error", ""))
    lowered = error.lower()
    if "429" in lowered or "ratelimit" in lowered or "rate limit" in lowered:
        return "rate_limit"
    if "content_filter" in lowered:
        return "content_filter"
    if "max_output_tokens" in lowered:
        return "max_output_tokens"
    if record.get("llm_error_kind") == "invalid_observation":
        return "invalid_observation"
    return str(record.get("llm_error_kind") or "other")


def summarize_fallbacks(
    records: list[dict[str, Any]],
    *,
    warning_rate: float,
    fatal_rate: float,
) -> dict[str, Any]:
    """fallback率と理由を集計する。"""
    fallback = [row for row in records if row.get("llm_error")]
    total = len(records)
    rate = len(fallback) / max(1, total)
    conversations = Counter(str(row.get("conversation_id", "")) for row in fallback)
    return {
        "passed": total > 0 and rate <= fatal_rate,
        "warning": rate > warning_rate,
        "records": total,
        "fallback_count": len(fallback),
        "fallback_rate": rate,
        "warning_rate": warning_rate,
        "fatal_rate": fatal_rate,
        "fallback_reason_distribution": dict(
            sorted(Counter(classify_fallback(row) for row in fallback).items())
        ),
        "fallback_conversations": len(conversations),
        "maximum_fallbacks_in_one_conversation": max(conversations.values(), default=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="scoring fallback監査")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warning-rate", type=float, default=0.01)
    parser.add_argument("--fatal-rate", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.warning_rate <= args.fatal_rate <= 1:
        raise SystemExit("fallback閾値は 0 <= warning <= fatal <= 1 が必要です。")
    records = [
        json.loads(line)
        for line in Path(args.input).open(encoding="utf-8")
        if line.strip()
    ]
    report = summarize_fallbacks(
        records,
        warning_rate=args.warning_rate,
        fatal_rate=args.fatal_rate,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    level = "WARN" if report["warning"] else "OK"
    print(
        f"[scoring fallback {level}] {report['fallback_count']}/{report['records']} "
        f"rate={report['fallback_rate']:.4%} warning={args.warning_rate:.4%} "
        f"fatal={args.fatal_rate:.4%}",
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(
            "scoring fallback率が致命上限を超えました: "
            f"{report['fallback_rate']:.4%} > {args.fatal_rate:.4%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
