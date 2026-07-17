"""完成済みMathDial DPOデータのpromptだけを中立形式へ変換する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from core.dpo_prompting import (
    CONTEXT_ONLY_DPO_PROMPT_TEMPLATE_VERSION,
    DPO_PROMPT_TEMPLATE_VERSION,
    NEUTRAL_CONVERSATION_DPO_PROMPT_TEMPLATE_VERSION,
    convert_mathdial_instruction_prompt_to_context_only,
    convert_mathdial_instruction_prompt_to_neutral_conversation,
)


PROMPT_REWRITE_VERSION = "mathdial_prompt_only_rewrite.v2"
PROMPT_MODES = ("context_only", "neutral_conversation")
PROMPT_METADATA_KEYS = {
    "source_dpo_prompt_template",
    "dpo_prompt_template",
    "source_prompt_sha256",
    "context_only_prompt_sha256",
    "rewritten_prompt_sha256",
    "local_prompt_mode",
    "frozen_chosen_sha256",
    "frozen_rejected_sha256",
    "prompt_rewrite_version",
}
PROMPT_TOP_LEVEL_KEYS = {
    "prompt",
    "source_dpo_prompt_template_version",
    "dpo_prompt_template_version",
    "prompt_rewrite_version",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_ordered_hash(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def record_identity(record: dict[str, Any], line_number: int) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{line_number}行目のmetadataがobjectではありません。")
    payload = {
        "line_number": line_number,
        "source_dataset": record.get("source_dataset"),
        "source_dialogue_id": record.get("source_dialogue_id"),
        "turn_index": record.get("turn_index"),
        "metadata_source_dataset": metadata.get("source_dataset"),
        "metadata_source_hash": metadata.get("source_hash"),
        "metadata_gold": bool(metadata.get("gold")),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def immutable_record_payload(record: dict[str, Any]) -> dict[str, Any]:
    """prompt変換で変更を許可しない全フィールドを返す。"""
    payload = {
        key: value
        for key, value in record.items()
        if key not in PROMPT_TOP_LEVEL_KEYS
    }
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadataがobjectではありません。")
    payload["metadata"] = {
        key: value
        for key, value in metadata.items()
        if key not in PROMPT_METADATA_KEYS
    }
    return payload


def is_gold_record(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata", {})
    return bool(metadata.get("gold")) or str(record.get("source_dataset", "")).lower() == "mathdial"


def prompt_template_for_mode(prompt_mode: str) -> str:
    """指定modeのtemplate versionを返す。"""
    if prompt_mode == "context_only":
        return CONTEXT_ONLY_DPO_PROMPT_TEMPLATE_VERSION
    if prompt_mode == "neutral_conversation":
        return NEUTRAL_CONVERSATION_DPO_PROMPT_TEMPLATE_VERSION
    raise ValueError(f"未知のprompt_modeです: {prompt_mode}")


def rewrite_prompt(prompt: str, *, prompt_mode: str) -> tuple[str, str]:
    """指定modeのprompt本文とtemplate versionを返す。"""
    output_template = prompt_template_for_mode(prompt_mode)
    if prompt_mode == "context_only":
        rewritten = convert_mathdial_instruction_prompt_to_context_only(prompt)
    else:
        rewritten = convert_mathdial_instruction_prompt_to_neutral_conversation(
            prompt
        )
    return rewritten, output_template


def rewrite_record(
    record: dict[str, Any],
    *,
    line_number: int,
    prompt_mode: str = "context_only",
) -> dict[str, Any]:
    for key in ("prompt", "chosen", "rejected", "metadata"):
        if key not in record:
            raise ValueError(f"{line_number}行目に`{key}`がありません。")
    prompt = str(record["prompt"])
    chosen = str(record["chosen"])
    rejected = str(record["rejected"])
    if not prompt.strip() or not chosen.strip() or not rejected.strip():
        raise ValueError(f"{line_number}行目のprompt/chosen/rejectedが空です。")
    source_template = record.get("dpo_prompt_template_version")
    if source_template != DPO_PROMPT_TEMPLATE_VERSION:
        raise ValueError(
            f"{line_number}行目の旧prompt templateが想定外です: {source_template!r}"
        )

    rewritten_prompt, output_template = rewrite_prompt(
        prompt,
        prompt_mode=prompt_mode,
    )
    metadata = dict(record["metadata"])
    metadata.update(
        {
            "source_dpo_prompt_template": source_template,
            "dpo_prompt_template": output_template,
            "source_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "rewritten_prompt_sha256": hashlib.sha256(
                rewritten_prompt.encode("utf-8")
            ).hexdigest(),
            "frozen_chosen_sha256": hashlib.sha256(
                chosen.encode("utf-8")
            ).hexdigest(),
            "frozen_rejected_sha256": hashlib.sha256(
                rejected.encode("utf-8")
            ).hexdigest(),
            "local_prompt_mode": prompt_mode,
            "prompt_rewrite_version": PROMPT_REWRITE_VERSION,
        }
    )
    output = dict(record)
    output.update(
        {
            "prompt": rewritten_prompt,
            "metadata": metadata,
            "source_dpo_prompt_template_version": source_template,
            "dpo_prompt_template_version": output_template,
            "prompt_rewrite_version": PROMPT_REWRITE_VERSION,
        }
    )
    if output["chosen"] != chosen or output["rejected"] != rejected:
        raise AssertionError("prompt変換中にchosen/rejectedが変更されました。")
    if record_identity(output, line_number) != record_identity(record, line_number):
        raise AssertionError("prompt変換中にsource識別子または順序情報が変更されました。")
    if immutable_record_payload(output) != immutable_record_payload(record):
        raise AssertionError(
            "prompt変換中にscore・採択条件・source情報などが変更されました。"
        )
    return output


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}をJSONとして読めません: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}はJSON objectではありません。")
            yield line_number, payload


def rewrite_file(
    source: Path,
    output: Path,
    *,
    expected_records: int,
    expected_gold: int,
    prompt_mode: str = "context_only",
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    digests = {
        name: hashlib.sha256()
        for name in (
            "identity",
            "immutable_record",
            "chosen",
            "rejected",
            "source_prompt",
            "output_prompt",
        )
    }
    count = 0
    gold_count = 0
    try:
        with temporary.open("w", encoding="utf-8") as file:
            for line_number, record in _read_jsonl(source):
                rewritten = rewrite_record(
                    record,
                    line_number=line_number,
                    prompt_mode=prompt_mode,
                )
                count += 1
                gold_count += int(is_gold_record(record))
                for name, value in (
                    ("identity", record_identity(record, line_number)),
                    (
                        "immutable_record",
                        json.dumps(
                            immutable_record_payload(record),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                    ("chosen", str(record["chosen"])),
                    ("rejected", str(record["rejected"])),
                    ("source_prompt", str(record["prompt"])),
                    ("output_prompt", str(rewritten["prompt"])),
                ):
                    _update_ordered_hash(digests[name], value)
                file.write(json.dumps(rewritten, ensure_ascii=False) + "\n")
        if count != expected_records:
            raise ValueError(f"{source}の件数が想定外です: {count}/{expected_records}")
        if gold_count != expected_gold:
            raise ValueError(f"{source}のgold件数が想定外です: {gold_count}/{expected_gold}")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    output_template = prompt_template_for_mode(prompt_mode)
    return {
        "input": str(source),
        "output": str(output),
        "input_sha256": sha256_file(source),
        "output_sha256": sha256_file(output),
        "records": count,
        "gold_records": gold_count,
        "non_gold_records": count - gold_count,
        "ordered_identity_sha256": digests["identity"].hexdigest(),
        "ordered_immutable_record_sha256": digests[
            "immutable_record"
        ].hexdigest(),
        "ordered_chosen_sha256": digests["chosen"].hexdigest(),
        "ordered_rejected_sha256": digests["rejected"].hexdigest(),
        "ordered_source_prompt_sha256": digests["source_prompt"].hexdigest(),
        "ordered_output_prompt_sha256": digests["output_prompt"].hexdigest(),
        "source_template": DPO_PROMPT_TEMPLATE_VERSION,
        "output_template": output_template,
        "local_prompt_mode": prompt_mode,
        "prompt_rewrite_version": PROMPT_REWRITE_VERSION,
        "chosen_rejected_unchanged": True,
        "record_order_unchanged": True,
        "scores_acceptance_and_source_fields_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MathDial DPO学習データのpromptだけを中立形式へ変換"
    )
    parser.add_argument("--basis-input", required=True)
    parser.add_argument("--random-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--records-per-arm", type=int, default=2500)
    parser.add_argument("--basis-gold-records", type=int, default=500)
    parser.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODES,
        default="context_only",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    basis_summary = rewrite_file(
        Path(args.basis_input),
        output_dir / "mathdial_basis_train.jsonl",
        expected_records=args.records_per_arm,
        expected_gold=args.basis_gold_records,
        prompt_mode=args.prompt_mode,
    )
    random_summary = rewrite_file(
        Path(args.random_input),
        output_dir / "mathdial_random_train.jsonl",
        expected_records=args.records_per_arm,
        expected_gold=0,
        prompt_mode=args.prompt_mode,
    )
    manifest = {
        "version": PROMPT_REWRITE_VERSION,
        "local_prompt_mode": args.prompt_mode,
        "output_template": basis_summary["output_template"],
        "invariants": {
            "only_prompt_and_prompt_metadata_changed": True,
            "chosen_rejected_unchanged": True,
            "record_order_unchanged": True,
            "basis_records": args.records_per_arm,
            "basis_selected_records": args.records_per_arm - args.basis_gold_records,
            "basis_gold_records": args.basis_gold_records,
            "random_records": args.records_per_arm,
            "random_gold_records": 0,
        },
        "arms": {"basis": basis_summary, "random": random_summary},
    }
    manifest_path = (
        Path(args.manifest) if args.manifest else output_dir / "rewrite_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
