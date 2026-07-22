"""MediTOD raw/canonicalデータの取得、正規化、分割、監査。"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from core.dialogue_schema import build_assistant_samples, canonical_json_hash, validate_conversation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets" / "meditod.yaml"
ATTRIBUTE_KEYS = {
    "when",
    "onset",
    "duration",
    "progression",
    "severity",
    "characteristics",
    "characteristics check",
    "progression check",
    "when check",
    "severity check",
    "location",
    "location check",
    "volume",
    "volume check",
    "color",
    "frequency",
    "ana factors check",
    "aggravating factors",
    "alleviating factors",
}


@dataclass(frozen=True)
class SourceDialogue:
    """1つのMediTOD source recordと注釈。"""

    source_key: str
    dialogue_id: str
    utterances: list[dict[str, Any]]
    annotations: list[list[dict[str, Any]]]


def load_yaml(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"MediTOD設定のrootはobjectである必要があります: {path}")
    return payload


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)


def _read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def download_public_raw(config: dict[str, Any], output_dir: Path | str) -> tuple[Path, Path, dict[str, Any]]:
    """revision固定URLからpublic rawを取得する。既存hash一致ファイルは再利用する。"""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for key in ("dialogs", "annotations"):
        path = target / f"{key}.json"
        valid_existing = False
        if path.exists():
            try:
                valid_existing = isinstance(_read_json(path), dict)
            except (OSError, json.JSONDecodeError):
                valid_existing = False
        if not valid_existing:
            temporary = path.with_suffix(path.suffix + ".download")
            temporary.unlink(missing_ok=True)
            try:
                urllib.request.urlretrieve(
                    config["public_raw"][f"{key}_url"], temporary
                )
                if not isinstance(_read_json(temporary), dict):
                    raise ValueError(f"MediTOD取得JSONのrootがobjectではありません: {key}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        paths.append(path)
    metadata = {
        "official_repository": config["official_repository"],
        "requested_revision": config["revision"],
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": config.get("license", {}),
        "source_files": {
            path.name: {"path": str(path), "sha256": file_sha256(path)} for path in paths
        },
    }
    return paths[0], paths[1], metadata


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _clinical_utterances(utterances: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        utterance
        for utterance in utterances
        if str(utterance.get("speaker", "")).strip().lower() in {"doctor", "patient"}
    ]


def content_hash(utterances: Iterable[dict[str, Any]]) -> str:
    value = [
        {
            "speaker": str(item.get("speaker", "")).strip().lower(),
            "text": _normalize_text(item.get("text", "")),
        }
        for item in _clinical_utterances(utterances)
    ]
    return canonical_json_hash(value)


def load_public_raw(dialogs_path: Path | str, annotations_path: Path | str) -> list[SourceDialogue]:
    dialogs = _read_json(dialogs_path)
    annotations = _read_json(annotations_path)
    if not isinstance(dialogs, dict) or not isinstance(annotations, dict):
        raise ValueError("MediTOD rawのdialogs/annotationsはobjectである必要があります。")
    if set(map(str, dialogs)) != set(map(str, annotations)):
        raise ValueError("MediTOD rawのdialogsとannotationsのID集合が一致しません。")
    records: list[SourceDialogue] = []
    for raw_key in sorted(dialogs, key=lambda value: str(value)):
        key = str(raw_key)
        row = dialogs[raw_key]
        annotation_rows = annotations[raw_key]
        utterances = row.get("utterances")
        if not isinstance(utterances, list) or not isinstance(annotation_rows, list):
            raise ValueError(f"MediTOD raw schemaが不正です: {key}")
        if len(utterances) != len(annotation_rows):
            raise ValueError(
                f"発話とannotationの長さが一致しません: {key} "
                f"{len(utterances)} != {len(annotation_rows)}"
            )
        normalized_annotations: list[list[dict[str, Any]]] = []
        for index, values in enumerate(annotation_rows):
            if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
                raise ValueError(f"annotation[{key}][{index}]がobject配列ではありません。")
            normalized_annotations.append([dict(item) for item in values])
        records.append(
            SourceDialogue(
                source_key=key,
                dialogue_id=str(row.get("dlg_id") or key),
                utterances=[dict(item) for item in utterances],
                annotations=normalized_annotations,
            )
        )
    return records


def _seed_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def assign_public_splits(
    grouped: dict[str, list[SourceDialogue]],
    *,
    seed: int,
    train_count: int,
    validation_count: int,
    test_count: int,
    ood_prefix: str,
) -> dict[str, str]:
    """会話hash単位でpublic raw専用splitを固定する。"""
    ood = []
    in_domain = []
    for digest, variants in grouped.items():
        prefixes = {row.dialogue_id.startswith(ood_prefix) for row in variants}
        if len(prefixes) != 1:
            raise ValueError(f"重複group内でOOD属性が一致しません: {digest}")
        (ood if True in prefixes else in_domain).append(digest)
    in_domain.sort(key=lambda value: _seed_rank(seed, value))
    ood.sort(key=lambda value: _seed_rank(seed, value))
    required = train_count + validation_count + test_count
    if len(in_domain) != required:
        raise ValueError(
            "MediTOD public rawのin-domain重複統合後件数が固定splitと一致しません: "
            f"{len(in_domain)} != {required}"
        )
    output = {}
    boundaries = (train_count, train_count + validation_count)
    for index, digest in enumerate(in_domain):
        output[digest] = "train" if index < boundaries[0] else "validation" if index < boundaries[1] else "test"
    output.update({digest: "test" for digest in ood})
    return output


def _annotation_for_source_turn(record: SourceDialogue, source_turn_index: int) -> list[dict[str, Any]]:
    if source_turn_index >= len(record.annotations):
        return []
    return [dict(item) for item in record.annotations[source_turn_index]]


def convert_raw_group(
    digest: str,
    variants: list[SourceDialogue],
    *,
    split: str,
    mode: str,
    ood_prefix: str,
) -> dict[str, Any]:
    """本文重複groupを1会話に変換し、annotation variantを保持する。"""
    variants = sorted(variants, key=lambda row: row.dialogue_id)
    representative = variants[0]
    clinical = _clinical_utterances(representative.utterances)
    source_index_by_uid = {
        str(item.get("uttr_id", index)): index
        for index, item in enumerate(representative.utterances)
    }
    turns: list[dict[str, Any]] = []
    for normalized_index, utterance in enumerate(clinical):
        speaker = str(utterance.get("speaker", "")).strip().lower()
        text = _normalize_text(utterance.get("text", ""))
        if not text:
            raise ValueError(f"空のMediTOD臨床発話があります: {representative.dialogue_id}")
        uid = str(utterance.get("uttr_id", normalized_index))
        source_index = source_index_by_uid[uid]
        annotation_variants = []
        for variant in variants:
            variant_uid_map = {
                str(item.get("uttr_id", index)): index
                for index, item in enumerate(variant.utterances)
            }
            if uid not in variant_uid_map:
                raise ValueError(f"重複annotation variantにuidがありません: {variant.dialogue_id}/{uid}")
            annotation_variants.append(
                {
                    "source_dialogue_id": variant.dialogue_id,
                    "annotations": _annotation_for_source_turn(variant, variant_uid_map[uid]),
                }
            )
        turns.append(
            {
                "role": "assistant" if speaker == "doctor" else "user",
                "text": text,
                "metadata": {
                    "source_turn_indices": [source_index],
                    "source_utterance_id": uid,
                    "source_speaker": speaker,
                    "keywords": list(utterance.get("keywords") or []),
                    "annotation_variants": annotation_variants,
                },
            }
        )
    control_annotations = []
    for variant in variants:
        for index, utterance in enumerate(variant.utterances):
            if str(utterance.get("speaker", "")).strip().lower() == "control":
                control_annotations.append(
                    {
                        "source_dialogue_id": variant.dialogue_id,
                        "source_turn_index": index,
                        "text": _normalize_text(utterance.get("text", "")),
                        "annotations": _annotation_for_source_turn(variant, index),
                    }
                )
    record = {
        "conversation_id": f"meditod_{mode}_{digest[:16]}",
        "source_dataset": "meditod",
        "split": split,
        "turns": turns,
        "num_messages": len(turns),
        "num_user_turns": sum(turn["role"] == "user" for turn in turns),
        "num_assistant_turns": sum(turn["role"] == "assistant" for turn in turns),
        "language": "English",
        "metadata": {
            "dataset_mode": mode,
            "split_policy": "meditod_public_raw_v1",
            "source_dialogue_ids": [row.dialogue_id for row in variants],
            "source_record_keys": [row.source_key for row in variants],
            "conversation_content_sha256": digest,
            "annotation_variant_count": len(variants),
            "duplicate_records_merged": len(variants) - 1,
            "control_annotations": control_annotations,
            "ood": all(row.dialogue_id.startswith(ood_prefix) for row in variants),
            "eligible_for_training": split == "train",
        },
    }
    return validate_conversation(record)


def _flatten_annotations(turn: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        annotation
        for variant in turn.get("metadata", {}).get("annotation_variants", [])
        for annotation in variant.get("annotations", [])
        if isinstance(annotation, dict)
    ]


def build_meditod_assistant_samples(record: dict[str, Any]) -> list[dict[str, Any]]:
    """共通assistant sampleへMediTODの前後annotationを付与する。"""
    samples = build_assistant_samples(record)
    by_turn = {sample["metadata"]["assistant_turn_index"]: sample for sample in samples}
    for turn_index, turn in enumerate(record["turns"]):
        if turn["role"] != "assistant":
            continue
        sample = by_turn[turn_index]
        sample["metadata"].update(
            {
                "response_annotation_variants": turn["metadata"].get("annotation_variants", []),
                "response_intents": sorted(
                    {str(item.get("intent", "")).strip().lower() for item in _flatten_annotations(turn) if item.get("intent")}
                ),
                "response_slots": sorted(
                    {str(item.get("slot", "")).strip().lower() for item in _flatten_annotations(turn) if item.get("slot")}
                ),
                "response_attributes": sorted(
                    {key for item in _flatten_annotations(turn) for key in item if key in ATTRIBUTE_KEYS}
                ),
                "next_user_annotation_variants": (
                    record["turns"][turn_index + 1]["metadata"].get("annotation_variants", [])
                    if turn_index + 1 < len(record["turns"])
                    and record["turns"][turn_index + 1]["role"] == "user"
                    else []
                ),
                "ood": bool(record["metadata"].get("ood")),
            }
        )
    return samples


def _annotation_signature(annotations: list[dict[str, Any]]) -> str:
    return canonical_json_hash(annotations)


def summarize(
    source_records: list[SourceDialogue],
    conversations: list[dict[str, Any]],
) -> dict[str, Any]:
    intents: Counter[str] = Counter()
    slots: Counter[str] = Counter()
    attributes: Counter[str] = Counter()
    disagreements = 0
    doctor_turns = patient_turns = controls = 0
    for record in source_records:
        for utterance in record.utterances:
            speaker = str(utterance.get("speaker", "")).lower()
            doctor_turns += speaker == "doctor"
            patient_turns += speaker == "patient"
            controls += speaker == "control"
    for conversation in conversations:
        for turn in conversation["turns"]:
            variants = turn["metadata"].get("annotation_variants", [])
            signatures = {
                _annotation_signature(variant.get("annotations", [])) for variant in variants
            }
            disagreements += len(signatures) > 1
            for annotation in _flatten_annotations(turn):
                if annotation.get("intent"):
                    intents[str(annotation["intent"]).strip().lower()] += 1
                if annotation.get("slot"):
                    slots[str(annotation["slot"]).strip().lower()] += 1
                for key in annotation:
                    if key in ATTRIBUTE_KEYS:
                        attributes[key] += 1
    return {
        "source": {
            "records": len(source_records),
            "doctor_turns": doctor_turns,
            "patient_turns": patient_turns,
            "control_turns_removed_from_text": controls,
        },
        "normalized": {
            "conversations": len(conversations),
            "messages": sum(row["num_messages"] for row in conversations),
            "assistant_turns": sum(row["num_assistant_turns"] for row in conversations),
            "user_turns": sum(row["num_user_turns"] for row in conversations),
            "mean_messages": (
                sum(row["num_messages"] for row in conversations) / len(conversations)
                if conversations else 0.0
            ),
            "split_conversations": dict(Counter(row["split"] for row in conversations)),
            "ood_conversations": sum(bool(row["metadata"].get("ood")) for row in conversations),
            "source_duplicates_merged": len(source_records) - len(conversations),
            "duplicate_groups": sum(len(row["metadata"]["source_dialogue_ids"]) > 1 for row in conversations),
            "annotation_disagreement_turns": disagreements,
            "intent_distribution": dict(sorted(intents.items())),
            "slot_distribution": dict(sorted(slots.items())),
            "attribute_distribution": dict(sorted(attributes.items())),
        },
    }


def assert_no_leakage(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    by_hash: dict[str, set[str]] = defaultdict(set)
    ids = set()
    for row in conversations:
        if row["conversation_id"] in ids:
            raise ValueError(f"conversation_idが重複しています: {row['conversation_id']}")
        ids.add(row["conversation_id"])
        bucket = "ood" if row["metadata"].get("ood") else row["split"]
        by_hash[row["metadata"]["conversation_content_sha256"]].add(bucket)
    leaked = {digest: sorted(values) for digest, values in by_hash.items() if len(values) > 1}
    if leaked:
        raise ValueError(f"MediTOD split間に本文hash leakageがあります: {len(leaked)}")
    return {
        "status": "passed",
        "dimensions": ["conversation_id", "conversation_content_sha256"],
        "unique_conversations": len(ids),
    }


def prepare_public_raw(
    dialogs_path: Path | str,
    annotations_path: Path | str,
    *,
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_records = load_public_raw(dialogs_path, annotations_path)
    grouped: dict[str, list[SourceDialogue]] = defaultdict(list)
    for record in source_records:
        grouped[content_hash(record.utterances)].append(record)
    raw_config = config["public_raw"]
    split_by_hash = assign_public_splits(
        grouped,
        seed=seed,
        train_count=int(raw_config["train_conversations"]),
        validation_count=int(raw_config["validation_conversations"]),
        test_count=int(raw_config["test_conversations"]),
        ood_prefix=str(raw_config["ood_prefix"]),
    )
    conversations = [
        convert_raw_group(
            digest,
            variants,
            split=split_by_hash[digest],
            mode="public_raw",
            ood_prefix=str(raw_config["ood_prefix"]),
        )
        for digest, variants in sorted(grouped.items())
    ]
    samples = [sample for row in conversations for sample in build_meditod_assistant_samples(row)]
    report = summarize(source_records, conversations)
    report["leakage_check"] = assert_no_leakage(conversations)
    report["split_policy"] = {
        "name": "meditod_public_raw_v1",
        "seed": seed,
        "train": int(raw_config["train_conversations"]),
        "validation": int(raw_config["validation_conversations"]),
        "test": int(raw_config["test_conversations"]),
        "ood": sum(row["metadata"].get("ood", False) for row in conversations),
    }
    report["samples"] = {
        "total": len(samples),
        "dpo_eligible": sum(row["metadata"]["dpo_eligible"] for row in samples),
        "after_state_observed": sum(row["metadata"]["after_state_observed"] for row in samples),
    }
    return conversations, samples, report


def _canonical_turn_annotations(utterance: dict[str, Any]) -> list[dict[str, Any]]:
    """canonical発話のactions/nluをraw互換annotation配列へまとめる。"""
    speaker = str(utterance.get("speaker", "")).strip().lower()
    values = utterance.get("actions") if speaker == "doctor" else utterance.get("nlu")
    return [dict(item) for item in (values or []) if isinstance(item, dict)]


def prepare_canonical_full(
    data_dir: Path | str,
    *,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """UMLS申請済みcanonical完全版を公式split IDで変換する。"""
    root = Path(data_dir)
    names = config["canonical_full"]
    dialogs_path = root / names["dialogs_filename"]
    dialogs = _read_json(dialogs_path)
    if not isinstance(dialogs, dict):
        raise ValueError("canonical dialogs.jsonはobjectである必要があります。")

    split_ids: dict[str, set[str]] = {}
    for split, config_key in (
        ("train", "train_ids_filename"),
        ("validation", "validation_ids_filename"),
        ("test", "test_ids_filename"),
    ):
        path = root / names[config_key]
        values = _read_json(path)
        if not isinstance(values, list):
            raise ValueError(f"canonical split IDは配列である必要があります: {path}")
        split_ids[split] = {str(value) for value in values}
    all_split_ids = set().union(*split_ids.values())
    if sum(len(values) for values in split_ids.values()) != len(all_split_ids):
        raise ValueError("canonical公式split間でdialogue IDが重複しています。")

    ood_ids_path = root / "ood_ids.json"
    explicit_ood = set(map(str, _read_json(ood_ids_path))) if ood_ids_path.exists() else set()
    unknown_split_ids = all_split_ids - set(map(str, dialogs))
    if unknown_split_ids:
        raise ValueError(f"canonical dialogsに存在しないsplit IDがあります: {sorted(unknown_split_ids)[:5]}")
    remaining = set(map(str, dialogs)) - all_split_ids
    if explicit_ood and explicit_ood != remaining:
        raise ValueError("ood_ids.jsonと公式split外dialogue IDが一致しません。")
    ood_ids = explicit_ood or remaining

    conversations: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    source_doctor = source_patient = controls = 0
    for raw_id in sorted(dialogs, key=lambda value: str(value)):
        dialogue_id = str(raw_id)
        row = dialogs[raw_id]
        utterances = row.get("utterances") if isinstance(row, dict) else None
        if not isinstance(utterances, list):
            raise ValueError(f"canonical dialogue schemaが不正です: {dialogue_id}")
        digest = content_hash(utterances)
        if digest in seen_hashes:
            raise ValueError(
                "canonical完全版に本文重複があります。公式splitを壊さず監査するため停止します: "
                f"{dialogue_id} / {seen_hashes[digest]}"
            )
        seen_hashes[digest] = dialogue_id
        split = next(
            (name for name, values in split_ids.items() if dialogue_id in values),
            "test" if dialogue_id in ood_ids else "",
        )
        if not split:
            raise ValueError(f"canonical dialogueのsplitが決まりません: {dialogue_id}")
        turns = []
        control_annotations = []
        for index, utterance in enumerate(utterances):
            speaker = str(utterance.get("speaker", "")).strip().lower()
            if speaker == "control":
                controls += 1
                control_annotations.append(
                    {
                        "source_turn_index": index,
                        "text": _normalize_text(utterance.get("text", "")),
                        "annotations": _canonical_turn_annotations(utterance),
                    }
                )
                continue
            if speaker not in {"doctor", "patient"}:
                raise ValueError(f"canonical dialogueの未知話者です: {dialogue_id}/{speaker}")
            source_doctor += speaker == "doctor"
            source_patient += speaker == "patient"
            text = _normalize_text(utterance.get("text", ""))
            if not text:
                raise ValueError(f"canonical dialogueに空発話があります: {dialogue_id}/{index}")
            turns.append(
                {
                    "role": "assistant" if speaker == "doctor" else "user",
                    "text": text,
                    "metadata": {
                        "source_turn_indices": [index],
                        "source_utterance_id": str(utterance.get("uttr_id", index)),
                        "source_speaker": speaker,
                        "keywords": list(utterance.get("keywords") or []),
                        "annotation_variants": [
                            {
                                "source_dialogue_id": dialogue_id,
                                "annotations": _canonical_turn_annotations(utterance),
                            }
                        ],
                        "dialog_state": utterance.get("dialog_state"),
                    },
                }
            )
        conversation = {
            "conversation_id": f"meditod_canonical_{digest[:16]}",
            "source_dataset": "meditod",
            "split": split,
            "turns": turns,
            "num_messages": len(turns),
            "num_user_turns": sum(turn["role"] == "user" for turn in turns),
            "num_assistant_turns": sum(turn["role"] == "assistant" for turn in turns),
            "language": "English",
            "metadata": {
                "dataset_mode": "canonical_full",
                "split_policy": "meditod_official_canonical",
                "source_dialogue_ids": [dialogue_id],
                "conversation_content_sha256": digest,
                "annotation_variant_count": 1,
                "duplicate_records_merged": 0,
                "control_annotations": control_annotations,
                "ood": dialogue_id in ood_ids,
                "eligible_for_training": split == "train" and dialogue_id not in ood_ids,
            },
        }
        conversations.append(validate_conversation(conversation))
    samples = [sample for row in conversations for sample in build_meditod_assistant_samples(row)]
    synthetic_sources = [
        SourceDialogue(
            source_key=str(key),
            dialogue_id=str(key),
            utterances=list(dialogs[key]["utterances"]),
            annotations=[
                _canonical_turn_annotations(turn) for turn in dialogs[key]["utterances"]
            ],
        )
        for key in dialogs
    ]
    report = summarize(synthetic_sources, conversations)
    report["source"].update(
        {
            "doctor_turns": source_doctor,
            "patient_turns": source_patient,
            "control_turns_removed_from_text": controls,
        }
    )
    report["leakage_check"] = assert_no_leakage(conversations)
    report["split_policy"] = {
        "name": "meditod_official_canonical",
        "split_id_files": {
            split: names[f"{split if split != 'validation' else 'validation'}_ids_filename"]
            for split in ("train", "validation", "test")
        },
        "ood": len(ood_ids),
    }
    report["samples"] = {
        "total": len(samples),
        "dpo_eligible": sum(row["metadata"]["dpo_eligible"] for row in samples),
        "after_state_observed": sum(row["metadata"]["after_state_observed"] for row in samples),
    }
    report["canonical_source_sha256"] = file_sha256(dialogs_path)
    return conversations, samples, report
