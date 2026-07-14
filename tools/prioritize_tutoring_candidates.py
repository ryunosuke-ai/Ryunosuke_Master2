"""assistant応答を見ず、user側の指導機会だけで候補順を決める。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


CONFUSION_PATTERN = re.compile(
    r"\b(don't understand|do not understand|confus(?:ed|ing)|stuck|wrong|"
    r"mistake|error|not sure|why (?:does|is|do|did)|doesn't work|cannot figure)\b",
    re.IGNORECASE,
)
ATTEMPT_PATTERN = re.compile(
    r"\b(i tried|my answer|my solution|i got|i think|here is my|feedback|"
    r"correct me|revise|check my|what did i do wrong)\b|=",
    re.IGNORECASE,
)
LEARNING_PATTERN = re.compile(
    r"\b(learn|study|homework|lesson|tutor|teach|practice|exercise|problem|"
    r"explain|understand|code|programming|equation|grammar|paragraph|essay|"
    r"pronunciation|physics|science|math)\b",
    re.IGNORECASE,
)

SUBJECT_PATTERNS = (
    ("math", re.compile(r"\b(math|algebra|geometry|equation|calculate|fraction|calculus)\b", re.I)),
    ("programming", re.compile(r"\b(code|programming|python|java|javascript|bug|algorithm)\b", re.I)),
    ("language", re.compile(r"\b(grammar|vocabulary|pronunciation|english|language|translate)\b", re.I)),
    ("writing", re.compile(r"\b(essay|paragraph|writing|revise|thesis|sentence|feedback)\b", re.I)),
    ("science", re.compile(r"\b(physics|chemistry|biology|science)\b", re.I)),
)


def user_context(record: dict[str, Any]) -> tuple[str, str, int]:
    """assistant応答を除いたuser履歴、次user反応、user発話数を返す。"""
    history = record.get("history", [])
    user_turns = [
        str(turn.get("text", ""))
        for turn in history
        if isinstance(turn, dict) and turn.get("role") == "user"
    ]
    return "\n".join(user_turns), str(record.get("next_user_turn") or ""), len(user_turns)


def opportunity_score(record: dict[str, Any]) -> tuple[float, list[str]]:
    """user側に診断・足場かけが必要な機会があるほど高くする。"""
    history, next_user, user_turns = user_context(record)
    latest_user = history.rsplit("\n", 1)[-1]
    reasons: list[str] = []
    score = 0.0
    if CONFUSION_PATTERN.search(latest_user):
        score += 5.0
        reasons.append("explicit_confusion")
    if ATTEMPT_PATTERN.search(latest_user):
        score += 3.0
        reasons.append("learner_attempt")
    if LEARNING_PATTERN.search(latest_user):
        score += 2.0
        reasons.append("learning_request")
    if "?" in latest_user:
        score += 1.0
        reasons.append("learner_question")
    if CONFUSION_PATTERN.search(next_user) or ATTEMPT_PATTERN.search(next_user):
        score += 3.0
        reasons.append("followup_learning_signal")
    score += min(user_turns, 4) * 0.5
    if user_turns >= 2:
        reasons.append("multi_turn_context")
    return score, reasons


def subject_bucket(record: dict[str, Any]) -> str:
    history, next_user, _ = user_context(record)
    text = f"{history}\n{next_user}"
    for name, pattern in SUBJECT_PATTERNS:
        if pattern.search(text):
            return name
    return "general"


def stable_key(conversation_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{conversation_id}".encode()).hexdigest()


def prioritize_records(
    records: list[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """会話を分断せず、score tier内はsubjectをround-robinする。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["conversation_id"])].append(record)
    conversations: list[dict[str, Any]] = []
    for conversation_id, rows in grouped.items():
        scored = [(opportunity_score(row), row) for row in rows]
        best_score, best_reasons = max(
            (item[0] for item in scored), key=lambda item: item[0]
        )
        bucket = subject_bucket(max(scored, key=lambda item: item[0][0])[1])
        conversations.append(
            {
                "conversation_id": conversation_id,
                "rows": sorted(rows, key=lambda row: int(row["turn_index"])),
                "score": best_score + min(len(rows), 5) * 0.2,
                "reasons": best_reasons,
                "bucket": bucket,
            }
        )
    tiers: dict[int, dict[str, deque[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(deque)
    )
    for conversation in conversations:
        tier = int(conversation["score"])
        tiers[tier][conversation["bucket"]].append(conversation)
    ordered_conversations: list[dict[str, Any]] = []
    for tier in sorted(tiers, reverse=True):
        buckets = tiers[tier]
        for queue in buckets.values():
            ordered = sorted(
                queue,
                key=lambda item: stable_key(item["conversation_id"], seed),
            )
            queue.clear()
            queue.extend(ordered)
        bucket_names = sorted(buckets)
        while any(buckets.values()):
            for bucket in bucket_names:
                if buckets[bucket]:
                    ordered_conversations.append(buckets[bucket].popleft())
    output: list[dict[str, Any]] = []
    for rank, conversation in enumerate(ordered_conversations, start=1):
        for row in conversation["rows"]:
            enriched = dict(row)
            metadata = dict(enriched.get("metadata", {}))
            metadata["candidate_priority"] = {
                "rank": rank,
                "opportunity_score": conversation["score"],
                "subject_bucket": conversation["bucket"],
                "reasons": conversation["reasons"],
                "uses_assistant_response": False,
            }
            enriched["metadata"] = metadata
            output.append(enriched)
    report = {
        "records": len(output),
        "conversations": len(ordered_conversations),
        "subject_distribution": dict(
            sorted(Counter(row["bucket"] for row in conversations).items())
        ),
        "score_tier_distribution": dict(
            sorted(
                Counter(int(row["score"]) for row in conversations).items(),
                reverse=True,
            )
        ),
        "assistant_response_used_for_priority": False,
        "seed": seed,
    }
    return output, report


def _write_spooled_conversation(
    directory: Path,
    rows: list[dict[str, Any]],
) -> tuple[int, str, float]:
    scored = [(opportunity_score(row), row) for row in rows]
    best_score, best_reasons = max(
        (item[0] for item in scored), key=lambda item: item[0]
    )
    bucket = subject_bucket(max(scored, key=lambda item: item[0][0])[1])
    score = best_score + min(len(rows), 5) * 0.2
    tier = int(score)
    path = directory / f"tier_{tier:03d}_{bucket}.jsonl"
    payload = {
        "score": score,
        "reasons": best_reasons,
        "bucket": bucket,
        "rows": sorted(rows, key=lambda row: int(row["turn_index"])),
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return tier, bucket, score


def prioritize_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    report_path: Path,
    seed: int,
) -> dict[str, Any]:
    """大規模JSONLを会話単位でspoolし、全件をメモリへ載せずに並べ替える。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_stat = input_path.stat()
    if output_path.is_file() and report_path.is_file():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
        if (
            previous.get("seed") == seed
            and previous.get("input_size_bytes") == input_stat.st_size
            and previous.get("input_mtime_ns") == input_stat.st_mtime_ns
            and previous.get("output_size_bytes") == output_path.stat().st_size
            and previous.get("assistant_response_used_for_priority") is False
        ):
            print(
                "[candidate priority] verified existing prioritized input: "
                f"records={previous.get('records', 0)}",
                flush=True,
            )
            return previous
    conversation_count = 0
    record_count = 0
    subject_counts: Counter[str] = Counter()
    tier_counts: Counter[int] = Counter()
    with tempfile.TemporaryDirectory(
        prefix="candidate_priority_", dir=output_path.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        current_id: str | None = None
        current_rows: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal conversation_count, record_count, current_rows
            if not current_rows:
                return
            tier, bucket, _ = _write_spooled_conversation(temporary, current_rows)
            conversation_count += 1
            record_count += len(current_rows)
            subject_counts[bucket] += 1
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
            tiers = sorted(
                {
                    int(path.name.split("_", 2)[1])
                    for path in temporary.glob("tier_*.jsonl")
                },
                reverse=True,
            )
            for tier in tiers:
                paths = sorted(temporary.glob(f"tier_{tier:03d}_*.jsonl"))
                files = [path.open(encoding="utf-8") for path in paths]
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
                            for row in conversation["rows"]:
                                enriched = dict(row)
                                metadata = dict(enriched.get("metadata", {}))
                                metadata["candidate_priority"] = {
                                    "rank": rank,
                                    "opportunity_score": conversation["score"],
                                    "subject_bucket": conversation["bucket"],
                                    "reasons": conversation["reasons"],
                                    "uses_assistant_response": False,
                                }
                                enriched["metadata"] = metadata
                                output.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                        active = next_active
                finally:
                    for file in files:
                        file.close()
    report = {
        "records": record_count,
        "conversations": conversation_count,
        "subject_distribution": dict(sorted(subject_counts.items())),
        "score_tier_distribution": {
            str(key): value for key, value in sorted(tier_counts.items(), reverse=True)
        },
        "assistant_response_used_for_priority": False,
        "seed": seed,
        "input_size_bytes": input_stat.st_size,
        "input_mtime_ns": input_stat.st_mtime_ns,
        "output_size_bytes": output_path.stat().st_size,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="WildChat tutoring候補の構造的優先順位付け")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = prioritize_jsonl(
        Path(args.input),
        Path(args.output),
        report_path=Path(args.report),
        seed=args.seed,
    )
    print(
        f"[candidate priority] records={report['records']} conversations={report['conversations']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
