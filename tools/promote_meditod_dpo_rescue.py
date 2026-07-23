"""全候補処理後に、安全で差のあるMediTOD DPOペアを順位救済する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.wildchat_health import has_explicit_unsafe_medical_advice


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [
        json.loads(line)
        for line in source.open(encoding="utf-8")
        if line.strip()
    ]


def write_jsonl(rows: list[dict[str, Any]], path: Path | str) -> None:
    output = Path(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)


def key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["source_dialogue_id"]), int(row["turn_index"])


def rescue_eligible(
    row: dict[str, Any],
    *,
    min_chosen: float,
    max_rejected: float,
    min_gap: float,
) -> bool:
    if row.get("skip_reason") not in {"low_chosen", "high_rejected", "small_gap"}:
        return False
    if any(not str(row.get(field, "")).strip() for field in ("prompt", "chosen", "rejected")):
        return False
    metadata = row.get("metadata", {})
    if metadata.get("translated_prompt_hash") != metadata.get("rejected_prompt_hash"):
        return False
    if has_explicit_unsafe_medical_advice(str(row["chosen"])):
        return False
    if has_explicit_unsafe_medical_advice(str(row["rejected"])):
        return False
    chosen = float(row.get("score_chosen", -1.0))
    rejected = float(row.get("score_rejected", 1.0))
    gap = float(row.get("score_gap", -1.0))
    return (
        chosen >= min_chosen
        and rejected <= max_rejected
        and gap >= min_gap
        and chosen > rejected
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MediTOD DPOの厳格基準未達ペアを安全条件付きで順位救済します。"
    )
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--skipped", required=True)
    parser.add_argument("--target-records", type=int, required=True)
    parser.add_argument("--min-chosen", type=float, default=0.60)
    parser.add_argument("--max-rejected", type=float, default=0.65)
    parser.add_argument("--min-gap", type=float, default=0.10)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    accepted = read_jsonl(args.accepted)
    skipped = read_jsonl(args.skipped)
    done = {key(row) for row in accepted}
    candidates = [
        row
        for row in skipped
        if key(row) not in done
        and rescue_eligible(
            row,
            min_chosen=args.min_chosen,
            max_rejected=args.max_rejected,
            min_gap=args.min_gap,
        )
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("score_gap", -1.0)),
            float(row.get("score_chosen", -1.0)),
            -float(row.get("score_rejected", 1.0)),
        ),
        reverse=True,
    )
    needed = max(0, args.target_records - len(accepted))
    promoted = candidates[:needed]
    promoted_keys = {key(row) for row in promoted}
    for row in promoted:
        row.pop("skip_reason", None)
        row.pop("skipped_at", None)
        row.pop("acceptance_thresholds", None)
        row.pop("would_accept_with_current_thresholds", None)
        row["acceptance_rule"] = "ranked_rescue_after_source_exhaustion"
        row.setdefault("metadata", {})[
            "acceptance_rule"
        ] = "ranked_rescue_after_source_exhaustion"
    accepted.extend(promoted)
    write_jsonl(accepted, args.accepted)
    write_jsonl(
        [row for row in skipped if key(row) not in promoted_keys],
        args.skipped,
    )
    report = {
        "accepted_before": len(accepted) - len(promoted),
        "eligible_rescue_candidates": len(candidates),
        "promoted": len(promoted),
        "accepted_after": len(accepted),
        "target_records": args.target_records,
        "thresholds": {
            "min_chosen": args.min_chosen,
            "max_rejected": args.max_rejected,
            "min_gap": args.min_gap,
        },
        "source_exhaustion_only": True,
    }
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if len(accepted) < args.target_records:
        raise RuntimeError(
            "全個人健康相談候補と順位救済を使ってもBASiS DPOが不足しました: "
            f"{len(accepted)}/{args.target_records}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
