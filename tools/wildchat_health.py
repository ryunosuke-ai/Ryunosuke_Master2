"""WildChat-1Mから健康相談の高再現率マルチターン候補をstreaming抽出する。"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from core.dialogue_schema import build_assistant_samples
from core.mathdial_basis import load_yaml
from tools.wildchat_tutoring import (
    NearDuplicateIndex,
    normalize_wildchat_row,
    sample_to_scoring_record,
    tokenize,
    write_jsonl,
)

HEALTH_FILTER_VERSION = "wildchat_health_broad.v4"

KNOWLEDGE_PREFIXES = (
    "what is ",
    "what are ",
    "explain ",
    "define ",
    "tell me about ",
    "how does ",
)
NON_CONSULTATION_PATTERN = re.compile(
    r"\b(?:"
    r"(?:summari[sz]e|summery|summerize)\s+(?:this|the following)\b"
    r"|(?:summari[sz]e|summery|summerize|draw a summary|rewrite|translate|"
    r"proofread|paraphrase|"
    r"improve|shorten|rephrase).{0,60}"
    r"(?:passage|text|sentence|email|article|abstract|paper|resume|paragraph)"
    r"|(?:write|writing|working on|create|make|generate|draft|prepare|edit|"
    r"help (?:me )?"
    r"(?:write|make|create|draw)).{0,80}"
    r"(?:script|screenplay|comic|manga|novel|article|essay|fiction|story|"
    r"lyrics|poem|website|slides?|presentation|powerpoint|resume|email|"
    r"character|faqs?|report|summary|expression|prompt|text|letter|content)"
    r"|(?:exam|quiz|multiple[- ]choice|group of answer choices|homework)"
    r"|(?:text[- ]based adventure|role ?play|dating expert|content writer|"
    r"sci[- ]?fi|fictional|flowchart image|find (?:a )?(?:city|neighbou?rhood)|"
    r"professional summarizer|pretend to be (?:a )?(?:famous )?author|"
    r"risk of bias|check grammar|correct (?:the )?grammar|youtube shorts?|"
    r"ai agent|doctor who|write (?:a )?(?:follow[- ]?up|sequel))"
    r"|(?:weekly meal plan|meal planning|workout plan|fitness plan)"
    r")\b",
    re.IGNORECASE,
)
CONTENT_DOMAIN_PATTERN = re.compile(
    r"\b(?:research (?:paper|study|article)|journal article|abstract|passage|"
    r"chapter|book i'm writing|comic|novel|story|video game|horror game|"
    r"zombie outbreak|employee survey|presentation|powerpoint|flowchart|"
    r"case study|student nurse|clinical placement|nursing documentation|"
    r"competency assessment programme|fabric features?|product features?)\b",
    re.IGNORECASE,
)
PERSONAL_PRONOUN_PATTERN = re.compile(
    r"\b(?:i|i've|i'm|i'd|me|my|mine|we|we've|we're|our|"
    r"my (?:child|son|daughter|mother|father|partner|husband|wife))\b",
    re.IGNORECASE,
)
PERSONAL_HEALTH_ACTION_PATTERN = re.compile(
    r"\b(?:have|had|feel|feeling|felt|experience|experiencing|suffer|"
    r"suffering|diagnosed|taking|take|developed|started|worsening|"
    r"getting worse|tested positive|tested negative|hurt|hurts)\b",
    re.IGNORECASE,
)
PERSONAL_CLINICAL_TERM_PATTERN = re.compile(
    r"\b(?:symptoms?|pain|ache|cough|fever|headache|nausea|dizz(?:y|iness)|"
    r"medications?|medicine|breath(?:ing|lessness)?|rash|infection|bleeding|"
    r"vomit(?:ing)?|diarrh(?:ea|oea)|swelling|allerg(?:y|ies|ic)|"
    r"blood pressure|heart rate|diagnosed|diagnosis|sore|burning|hurts?|"
    r"anxiety|depression|ocd|autism)\b",
    re.IGNORECASE,
)
DIRECT_CONSULTATION_PATTERN = re.compile(
    r"\b(?:should (?:i|we) (?:see|contact|visit)|"
    r"what (?:could|might) (?:this|these symptoms) be|"
    r"(?:my|our) (?:doctor|medication|medicine|symptoms?|pain)|"
    r"give (?:your|me) (?:a )?view on my symptoms|"
    r"what kind of illness (?:i|we) (?:have|might have))\b",
    re.IGNORECASE,
)


def _has_personal_clinical_signal(text: str) -> bool:
    """同一の短い範囲に本人の状態変化と臨床語があるか判定する。"""
    pronoun = PERSONAL_PRONOUN_PATTERN.search(text)
    action = PERSONAL_HEALTH_ACTION_PATTERN.search(text)
    clinical = PERSONAL_CLINICAL_TERM_PATTERN.search(text)
    if not pronoun or not action or not clinical:
        return False
    positions = (pronoun.start(), action.start(), clinical.start())
    return max(positions) - min(positions) <= 180
MEDICATION_PATTERN = re.compile(
    r"\b(?:aspirin|ibuprofen|acetaminophen|paracetamol|amoxicillin|azithromycin|"
    r"metformin|insulin|warfarin|lisinopril|prednisone|albuterol|omeprazole|"
    r"[a-z]{4,}(?:cillin|mycin|cycline|pril|sartan|olol|statin|prazole|formin))\b",
    re.IGNORECASE,
)
UNSAFE_MEDICAL_ADVICE_PATTERNS = (
    re.compile(r"\b(?:stop|discontinue|double|triple)\s+(?:taking\s+)?(?:your\s+)?(?:medication|medicine|dose|insulin|warfarin)\b", re.I),
    re.compile(
        r"(?:^|[.!?]\s*)(?:please\s+)?take\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml)\b"
        r"|\b(?:you\s+should|you\s+must)\s+take\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml)\b",
        re.I,
    ),
    re.compile(r"\b(?:definitely|certainly)\s+(?:have|is)\b", re.I),
    re.compile(r"\b(?:no need|do not need|don't need)\s+to\s+(?:see|contact|visit)\s+(?:a\s+)?(?:doctor|clinic|hospital)\b", re.I),
    re.compile(r"(?:薬|服用|投薬).{0,12}(?:中止|倍量|2倍|３倍|3倍)", re.I),
    re.compile(
        r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml).{0,12}"
        r"(?:(?:飲んで|服用して|投与して)(?:ください|下さい)|"
        r"(?:飲む|服用する|投与する)べき)",
        re.I,
    ),
    re.compile(r"(?:間違いなく|確実に).{0,8}(?:です|病気|疾患)", re.I),
    re.compile(r"(?:受診|医師|病院).{0,10}(?:必要ありません|不要です|行かなくて)", re.I),
)


@lru_cache(maxsize=256)
def _keyword_pattern(value: str) -> re.Pattern[str]:
    """英語keywordを単語境界付きで照合し、Spain内のpain等を除外する。"""
    escaped = re.escape(str(value).strip()).replace(r"\ ", r"\s+")
    prefix = r"(?<![A-Za-z0-9_])" if value and value[0].isalnum() else ""
    suffix = r"(?![A-Za-z0-9_])" if value and value[-1].isalnum() else ""
    return re.compile(prefix + escaped + suffix, re.IGNORECASE)


def _contains_any(text: str, values: Iterable[str]) -> bool:
    return any(_keyword_pattern(str(value)).search(text) for value in values)


def is_personal_health_consultation(
    record: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    """本人の健康問題について情報を追加しながら相談する会話か判定する。"""
    user_turns = [
        unicodedata.normalize("NFKC", turn["text"][:2_000])
        for turn in record["turns"]
        if turn["role"] == "user"
    ]
    if not user_turns:
        return False
    # 長いWildChat会話は途中で別用途へ移ることがある。文章作成・課題・創作が
    # 混ざる会話は、後続turnを医療相談と誤認しないよう会話単位で除外する。
    if any(
        NON_CONSULTATION_PATTERN.search(text[:2_000])
        or CONTENT_DOMAIN_PATTERN.search(text[:2_000])
        for text in user_turns
    ):
        return False
    substantive = [text.strip() for text in user_turns if len(tokenize(text)) >= 4]
    if substantive:
        first = substantive[0][:2_000]
        if (
            len(substantive[0]) > 1_200
            and not DIRECT_CONSULTATION_PATTERN.search(first)
        ):
            return False
        if first.startswith(('"', "“")) and not DIRECT_CONSULTATION_PATTERN.search(first):
            return False
    # 挨拶後に別用途へ遷移する長大セッションを除くため、相談開始は最初の
    # 実質的な3 user発話以内に現れることを要求する。
    for text in substantive[:3]:
        # 論文・記事の貼り付け本文を一人称相談と誤認しない。個人相談の
        # 発端は通常、短い依頼または長文でも冒頭に症状が現れる。
        candidate_text = text[:2_000]
        if not _contains_any(candidate_text, config["domain_keywords"]):
            continue
        if DIRECT_CONSULTATION_PATTERN.search(candidate_text):
            return True
        clinical = PERSONAL_CLINICAL_TERM_PATTERN.search(candidate_text)
        if not clinical:
            continue
        if _has_personal_clinical_signal(candidate_text):
            return True
    return False


def health_domain_flags(record: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    user_turns = [turn["text"] for turn in record["turns"] if turn["role"] == "user"]
    all_text = " ".join(turn["text"] for turn in record["turns"])
    # 粗判定で巨大な貼り付け文書全体を何度も走査しない。
    user_text = " ".join(text[:2_000] for text in user_turns)
    health = _contains_any(user_text, config["domain_keywords"])
    respiratory = health and _contains_any(user_text, config["respiratory_keywords"])
    personal = is_personal_health_consultation(record, config)
    followups = user_turns[1:]
    followup_information = any(
        len(tokenize(text)) >= 3
        and (
            _contains_any(text, config.get("patient_information_markers", []))
            or _contains_any(text, config["domain_keywords"])
            or bool(re.search(r"\b(?:i|my|me|mine)\b", text, re.IGNORECASE))
        )
        for text in followups
    )
    first = user_turns[0].strip().lower() if user_turns else ""
    single_knowledge = any(first.startswith(prefix) for prefix in KNOWLEDGE_PREFIXES) and not personal
    explicit_pii = any(re.search(pattern, all_text, flags=re.IGNORECASE) for pattern in config.get("pii_patterns", []))
    toxic_text = _contains_any(all_text, config.get("toxic_keywords", []))
    return {
        "health": health,
        "respiratory": respiratory,
        "personal": personal,
        "followup_information": followup_information,
        "single_knowledge": single_knowledge,
        "explicit_pii": explicit_pii,
        "toxic_text": toxic_text,
    }


def health_conversation_diagnostic_category(
    record: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """主実験の採否に使わない健康会話の診断カテゴリを返す。"""
    if is_personal_health_consultation(record, config):
        return "personal_consultation"
    user_text = " ".join(
        turn["text"][:2_000]
        for turn in record.get("turns", [])
        if turn.get("role") == "user"
    )
    if NON_CONSULTATION_PATTERN.search(user_text) or CONTENT_DOMAIN_PATTERN.search(
        user_text
    ):
        return "health_related_task"
    return "general_health_dialogue"


def protected_medical_terms(sample: dict[str, Any]) -> list[str]:
    """翻訳時に原語保持を要求する薬剤名だけを抽出する。"""
    text = " ".join(
        [str(turn.get("text", "")) for turn in sample.get("history", [])]
        + [str(sample.get("response", ""))]
    )
    return list(dict.fromkeys(match.group(0) for match in MEDICATION_PATTERN.finditer(text)))


def has_explicit_unsafe_medical_advice(text: str) -> bool:
    """明白な危険投薬・受診抑制・根拠のない断定だけを保守的に検知する。"""
    return any(pattern.search(text) for pattern in UNSAFE_MEDICAL_ADVICE_PATTERNS)


def sample_with_medical_metadata(
    sample: dict[str, Any],
    *,
    conversation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """共通scoring recordへ翻訳保持対象の薬剤名だけを追加する。"""
    record = sample_to_scoring_record(sample)
    record["metadata"] = {
        **record["metadata"],
        "protected_medical_terms": protected_medical_terms(sample),
        "personal_health_consultation": bool(
            (conversation_metadata or {}).get("personal_symptom_consultation")
        ),
        "health_filter_version": HEALTH_FILTER_VERSION,
    }
    return record


def extract_candidates(
    rows: Iterable[dict[str, Any]],
    config: dict[str, Any],
    limit: int | None = None,
    *,
    target_candidate_records: int | None = None,
    progress_every: int = 10_000,
    checkpoint_every: int = 100_000,
    initial_general: list[dict[str, Any]] | None = None,
    initial_respiratory: list[dict[str, Any]] | None = None,
    initial_counts: dict[str, Any] | None = None,
    on_checkpoint: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """health domainとマルチターン性だけで粗候補を抽出する。"""
    general = list(initial_general or [])
    respiratory = list(initial_respiratory or [])
    counts: Counter[str] = Counter(initial_counts or {})
    counts["stopped_by_candidate_target"] = 0
    counts["stopped_by_row_limit"] = 0
    exact_seen = {row["metadata"]["conversation_hash"] for row in general}
    near = NearDuplicateIndex(float(config.get("near_duplicate_jaccard", 0.9)))
    for existing in general:
        near.is_duplicate(tokenize(" ".join(turn["text"] for turn in existing["turns"])))
    for row in rows:
        if on_checkpoint and checkpoint_every > 0 and counts["stream_rows"] and counts["stream_rows"] % checkpoint_every == 0:
            on_checkpoint(general, respiratory, dict(counts), False)
        if limit is not None and counts["stream_rows"] >= limit:
            counts["stopped_by_row_limit"] = 1
            break
        counts["stream_rows"] += 1
        if progress_every > 0 and counts["stream_rows"] % progress_every == 0:
            print(
                "[extract_wildchat_health] "
                f"stream_rows={counts['stream_rows']} "
                f"general_conversations={len(general)} "
                f"candidate_records={counts['general_candidate_records']}",
                flush=True,
            )
        record, reason = normalize_wildchat_row(row, config)
        if record is None:
            counts[f"excluded_{reason}"] += 1
            continue
        user_turns = record["num_user_turns"]
        for threshold in range(2, 6):
            counts[f"at_least_{threshold}_user_turns"] += int(user_turns >= threshold)
        if user_turns < int(config.get("minimum_user_turns", 4)):
            counts["excluded_too_few_user_turns"] += 1
            continue
        flags = health_domain_flags(record, config)
        counts["health_domain"] += int(flags["health"])
        counts["personal_symptom_consultation"] += int(flags["personal"])
        counts["followup_information"] += int(flags["followup_information"])
        if not flags["health"]:
            counts["excluded_non_health"] += 1
            continue
        if config.get("require_personal_consultation", False) and not flags["personal"]:
            counts["excluded_non_personal_health"] += 1
            continue
        if flags["single_knowledge"] and config.get("exclude_single_turn_knowledge_questions", True):
            counts["excluded_medical_knowledge_only"] += 1
            continue
        if config.get("require_followup_information", True) and not flags["followup_information"]:
            counts["excluded_no_followup_information"] += 1
            continue
        if flags["explicit_pii"]:
            counts["excluded_explicit_pii"] += 1
            continue
        if flags["toxic_text"]:
            counts["excluded_toxic_text"] += 1
            continue
        digest = record["metadata"]["conversation_hash"]
        if digest in exact_seen:
            counts["excluded_exact_duplicate"] += 1
            continue
        token_set = tokenize(" ".join(turn["text"] for turn in record["turns"]))
        if near.is_duplicate(token_set):
            counts["excluded_near_duplicate"] += 1
            continue
        exact_seen.add(digest)
        record["metadata"].update(
            {
                "domain": "general_health_consultation",
                "has_followup_patient_information": True,
                "personal_symptom_consultation": flags["personal"],
                "health_filter_version": HEALTH_FILTER_VERSION,
                "pii_metadata_retained": False,
            }
        )
        general.append(record)
        eligible = [
            sample for sample in build_assistant_samples(record)
            if sample["metadata"]["dpo_eligible"] and sample.get("next_user_turn") is not None
        ]
        counts["general_candidate_records"] += len(eligible)
        if flags["respiratory"]:
            copied = json.loads(json.dumps(record))
            copied["metadata"]["domain"] = "respiratory_health"
            respiratory.append(copied)
            counts["respiratory_candidate_records"] += len(eligible)
        if target_candidate_records is not None and counts["general_candidate_records"] >= target_candidate_records:
            counts["stopped_by_candidate_target"] = 1
            break
    counts["general_conversations"] = len(general)
    counts["respiratory_conversations"] = len(respiratory)
    counts["target_candidate_records"] = target_candidate_records or 0
    counts["stream_exhausted"] = int(not counts["stopped_by_candidate_target"] and not counts["stopped_by_row_limit"])
    if on_checkpoint:
        on_checkpoint(general, respiratory, dict(counts), True)
    return general, respiratory, dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description="WildChat health候補をstreaming抽出")
    parser.add_argument("--config", default="configs/datasets/wildchat_health.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fixture")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target-candidate-records", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--heartbeat-file")
    parser.add_argument("--minimum-user-turns", type=int)
    parser.add_argument("--require-personal-consultation", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.minimum_user_turns is not None:
        config["minimum_user_turns"] = args.minimum_user_turns
    if args.require_personal_consultation:
        config["require_personal_consultation"] = True
    output = Path(args.output_dir)
    general_path = output / "general_health_consultation_conversations.jsonl"
    respiratory_path = output / "respiratory_health_conversations.jsonl"
    checkpoint_path = output / "stream_checkpoint.json"
    initial_general: list[dict[str, Any]] = []
    initial_respiratory: list[dict[str, Any]] = []
    initial_counts: dict[str, Any] = {}
    processed = 0
    if not args.no_resume and checkpoint_path.exists() and general_path.exists() and respiratory_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if int(checkpoint.get("seed", -1)) != args.seed:
            raise ValueError("WildChat health checkpointのseedが一致しません。")
        outputs = (
            output / "general_health_consultation_candidates.jsonl",
            output / "respiratory_health_candidates.jsonl",
            output / "statistics.json",
            output / "manifest.json",
        )
        if (
            checkpoint.get("completed") is True
            and checkpoint.get("health_filter_version") == HEALTH_FILTER_VERSION
            and all(path.is_file() for path in outputs)
        ):
            print("[extract_wildchat_health] completed checkpointを再利用", flush=True)
            return 0
        initial_general = [json.loads(line) for line in general_path.open(encoding="utf-8") if line.strip()]
        initial_respiratory = [json.loads(line) for line in respiratory_path.open(encoding="utf-8") if line.strip()]
        initial_counts = dict(checkpoint.get("statistics", {}))
        processed = int(initial_counts.get("stream_rows", 0))
    if args.fixture:
        source = (json.loads(line) for line in Path(args.fixture).open(encoding="utf-8") if line.strip())
        rows = islice(source, processed, None)
    else:
        from datasets import load_dataset

        rows = load_dataset(
            config["dataset_name"], split=config["split"], revision=config["revision"], streaming=True
        )
        buffer_size = int(config.get("stream_shuffle_buffer_size", 0))
        if buffer_size:
            rows = rows.shuffle(seed=args.seed, buffer_size=buffer_size)
        if processed:
            rows = rows.skip(processed)

    def checkpoint_callback(general: list[dict[str, Any]], respiratory: list[dict[str, Any]], counts: dict[str, Any], completed: bool) -> None:
        write_jsonl(general, general_path)
        write_jsonl(respiratory, respiratory_path)
        payload = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seed": args.seed,
            "completed": completed,
            "health_filter_version": HEALTH_FILTER_VERSION,
            "statistics": counts,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(checkpoint_path)
        if args.heartbeat_file:
            Path(args.heartbeat_file).write_text(json.dumps({"timestamp": payload["timestamp"], "stage": "extract_wildchat", "stream_rows": counts.get("stream_rows", 0)}, ensure_ascii=False) + "\n", encoding="utf-8")

    general, respiratory, stats = extract_candidates(
        rows,
        config,
        args.limit,
        target_candidate_records=args.target_candidate_records,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
        initial_general=initial_general,
        initial_respiratory=initial_respiratory,
        initial_counts=initial_counts,
        on_checkpoint=checkpoint_callback,
    )
    write_jsonl(general, general_path)
    write_jsonl(respiratory, respiratory_path)
    general_samples = [
        sample_with_medical_metadata(
            sample,
            conversation_metadata=record.get("metadata", {}),
        )
        for record in general
        for sample in build_assistant_samples(record)
        if sample["metadata"]["dpo_eligible"] and sample.get("next_user_turn") is not None
    ]
    respiratory_ids = {row["conversation_id"] for row in respiratory}
    write_jsonl(general_samples, output / "general_health_consultation_candidates.jsonl")
    write_jsonl([row for row in general_samples if row["conversation_id"] in respiratory_ids], output / "respiratory_health_candidates.jsonl")
    (output / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "dataset": config["dataset_name"],
        "revision": config["revision"],
        "config": config,
        "stream_shuffle_seed": args.seed,
        "target_candidate_records": args.target_candidate_records,
        "health_filter_version": HEALTH_FILTER_VERSION,
        "statistics": stats,
        "pii_policy": "retain conversation_hash and source_model only; discard source metadata",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
