"""assistant応答を見ず、患者側の情報収集機会でWildChat health候補を優先する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DETAIL_PATTERN = re.compile(
    r"\b(?:started|for (?:a |\d+ )?(?:day|days|week|weeks|month|months|year|years)|"
    r"worse|better|severe|mild|constant|intermittent|taking|medication|history|test|"
    r"fever|pain|cough|rash|nausea|dizzy|breath|sputum)\b",
    re.IGNORECASE,
)
UNCERTAINTY_PATTERN = re.compile(
    r"\b(?:worried|concerned|not sure|should i|what should|is this serious|help)\b",
    re.IGNORECASE,
)
BUCKET_PATTERNS = (
    ("medication_test", re.compile(r"\b(?:medication|medicine|drug|taking|test|scan|blood work)\b", re.I)),
    ("background", re.compile(r"\b(?:history|family|smoke|alcohol|travel|exposure|work)\b", re.I)),
    ("respiratory", re.compile(r"\b(?:cough|breath|sputum|wheeze|throat|congestion)\b", re.I)),
    ("pain", re.compile(r"\b(?:pain|ache|sore|severity)\b", re.I)),
)


def patient_context(record: dict[str, Any]) -> tuple[str, str, int]:
    users = [
        str(turn.get("text", ""))
        for turn in record.get("history", [])
        if isinstance(turn, dict) and turn.get("role") == "user"
    ]
    return "\n".join(users), str(record.get("next_user_turn") or ""), len(users)


def opportunity_score(record: dict[str, Any]) -> tuple[float, list[str]]:
    history, next_user, user_count = patient_context(record)
    latest = history.rsplit("\n", 1)[-1]
    reasons = []
    score = min(user_count, 6) * 0.6
    details = len(DETAIL_PATTERN.findall(latest))
    if details:
        score += min(details, 5) * 1.2
        reasons.append("patient_details")
    if UNCERTAINTY_PATTERN.search(latest):
        score += 2.0
        reasons.append("patient_concern")
    if DETAIL_PATTERN.search(next_user):
        score += 2.0
        reasons.append("observed_followup_information")
    if len(latest.split()) >= 12:
        score += 1.0
        reasons.append("substantive_latest_turn")
    if user_count >= 3:
        reasons.append("multi_turn_history")
    return score, reasons


def stage_bucket(record: dict[str, Any]) -> str:
    history, next_user, _ = patient_context(record)
    text = f"{history}\n{next_user}"
    for name, pattern in BUCKET_PATTERNS:
        if pattern.search(text):
            return name
    return "general_symptom"


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def prioritize_jsonl(input_path: Path, output_path: Path, report_path: Path, *, seed: int) -> dict[str, Any]:
    """会話単位spoolにより大規模JSONLをメモリへ全載せせず優先順へ並べる。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_stat = input_path.stat()
    if output_path.exists() and report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            previous.get("seed") == seed
            and previous.get("input_size_bytes") == input_stat.st_size
            and previous.get("input_mtime_ns") == input_stat.st_mtime_ns
            and previous.get("output_size_bytes") == output_path.stat().st_size
            and previous.get("assistant_response_used_for_priority") is False
        ):
            print(f"[health candidate priority] verified existing records={previous['records']}", flush=True)
            return previous
    records = conversations = 0
    bucket_counts: Counter[str] = Counter()
    tier_counts: Counter[int] = Counter()
    with tempfile.TemporaryDirectory(prefix="health_priority_", dir=output_path.parent) as temporary_name:
        temporary = Path(temporary_name)
        current_id = None
        current_rows: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal records, conversations, current_rows
            if not current_rows:
                return
            scored = [(opportunity_score(row), row) for row in current_rows]
            best_score, reasons = max((item[0] for item in scored), key=lambda item: item[0])
            best_row = max(scored, key=lambda item: item[0][0])[1]
            bucket = stage_bucket(best_row)
            score = best_score + min(len(current_rows), 5) * 0.2
            tier = int(score)
            path = temporary / f"tier_{tier:03d}_{bucket}.jsonl"
            payload = {
                "conversation_id": current_rows[0]["conversation_id"],
                "score": score,
                "reasons": reasons,
                "bucket": bucket,
                "rows": sorted(current_rows, key=lambda row: int(row["turn_index"])),
            }
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            records += len(current_rows)
            conversations += 1
            bucket_counts[bucket] += 1
            tier_counts[tier] += 1
            current_rows = []

        with input_path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                conversation_id = str(row["conversation_id"])
                if current_id is not None and conversation_id != current_id:
                    flush()
                current_id = conversation_id
                current_rows.append(row)
            flush()
        rank = 0
        with output_path.open("w", encoding="utf-8") as output:
            tiers = sorted({int(path.name.split("_", 2)[1]) for path in temporary.glob("tier_*.jsonl")}, reverse=True)
            for tier in tiers:
                files = [path.open(encoding="utf-8") for path in sorted(temporary.glob(f"tier_{tier:03d}_*.jsonl"))]
                try:
                    active = list(files)
                    while active:
                        next_active = []
                        for file in active:
                            line = file.readline()
                            if not line:
                                continue
                            next_active.append(file)
                            conversation = json.loads(line)
                            rank += 1
                            for row in sorted(conversation["rows"], key=lambda item: int(item["turn_index"])):
                                metadata = dict(row.get("metadata", {}))
                                metadata["candidate_priority"] = {
                                    "rank": rank,
                                    "opportunity_score": conversation["score"],
                                    "stage_bucket": conversation["bucket"],
                                    "reasons": conversation["reasons"],
                                    "uses_assistant_response": False,
                                }
                                output.write(json.dumps({**row, "metadata": metadata}, ensure_ascii=False) + "\n")
                        active = next_active
                finally:
                    for file in files:
                        file.close()
    report = {
        "records": records,
        "conversations": conversations,
        "stage_bucket_distribution": dict(bucket_counts),
        "score_tier_distribution": {str(key): value for key, value in sorted(tier_counts.items(), reverse=True)},
        "assistant_response_used_for_priority": False,
        "seed": seed,
        "input_size_bytes": input_stat.st_size,
        "input_mtime_ns": input_stat.st_mtime_ns,
        "output_size_bytes": output_path.stat().st_size,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="WildChat health候補を患者情報だけで優先順位付け")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = prioritize_jsonl(Path(args.input), Path(args.output), Path(args.report), seed=args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
