"""旧runの検証済みMathDial BASiS-DPO採択結果を安全に再利用する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from core.dpo_prompting import DPO_PROMPT_TEMPLATE_VERSION
from core.transition_bayes_model import load_transition_bayes_model
from tools.jsonl_utils import read_jsonl_records
from tools.translate_and_generate_dpo import (
    PROMPT_TEMPLATE_VERSION,
    bayes_model_version,
    dpo_record_key,
    passes_thresholds,
    source_record_key,
)


def parse_args() -> argparse.Namespace:
    """CLI引数を読む。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", required=True)
    parser.add_argument("--current-selection", required=True)
    parser.add_argument("--bayes-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-skipped")
    parser.add_argument("--skipped-output")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--generation-model", required=True)
    parser.add_argument("--scoring-model", required=True)
    parser.add_argument("--style-preset", default="mathdial_tutoring")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument(
        "--source-candidates",
        type=int,
        action="append",
        default=[],
        help="再利用を許可する旧runのrejected候補数。未指定時は現在値だけを許可します。",
    )
    parser.add_argument(
        "--reuse-threshold-skips",
        action="store_true",
        help="low_chosen/high_rejected/small_gapの判定済みskipも継承します。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-characters", type=int, default=16000)
    parser.add_argument("--min-score-gap", type=float, default=0.20)
    parser.add_argument("--min-chosen-posterior", type=float, default=0.70)
    parser.add_argument("--max-rejected-posterior", type=float, default=0.55)
    return parser.parse_args()


def _read(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    records, malformed = read_jsonl_records(
        path,
        missing_ok=missing_ok,
        strict=False,
        label=str(path),
    )
    if malformed:
        raise ValueError(f"再利用元または現在出力に壊れたJSONLがあります: {path} malformed={malformed}")
    return [record for record in records if isinstance(record, dict)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_length(record: dict[str, Any]) -> int:
    return len(str(record.get("prompt", ""))) + len(str(record.get("response", "")))


def _validate_record(
    record: dict[str, Any],
    selection: dict[str, Any],
    *,
    model_version: str,
    generation_model: str,
    scoring_model: str,
    style_preset: str,
    candidates: int,
    allowed_source_candidates: set[int] | None,
    seed: int,
    max_source_characters: int,
    min_score_gap: float,
    min_chosen_posterior: float,
    max_rejected_posterior: float,
) -> str | None:
    """再利用不可なら理由、利用可能ならNoneを返す。"""
    source_prompt = str(selection.get("prompt", ""))
    source_response = str(selection.get("response", ""))
    if _source_length(selection) > max_source_characters:
        return "source_too_long"
    if record.get("source_prompt_en") != source_prompt:
        return "source_prompt_mismatch"
    if record.get("source_chosen_en") != source_response:
        return "source_response_mismatch"
    metadata = record.get("metadata") or {}
    expected_prompt_hash = hashlib.sha256(source_prompt.encode("utf-8")).hexdigest()
    if metadata.get("source_prompt_hash") != expected_prompt_hash:
        return "source_prompt_hash_mismatch"
    if metadata.get("translated_prompt_hash") != metadata.get("rejected_prompt_hash"):
        return "translated_context_hash_mismatch"
    if record.get("model_used_for_translation") != generation_model:
        return "generation_model_mismatch"
    if record.get("model_used_for_rejected_generation") != generation_model:
        return "rejected_generation_model_mismatch"
    if record.get("model_used_for_scoring") != scoring_model:
        return "scoring_model_mismatch"
    if record.get("bayesian_model_version") != model_version:
        return "bayes_model_mismatch"
    if record.get("prompt_template_version") != PROMPT_TEMPLATE_VERSION:
        return "prompt_template_mismatch"
    if record.get("dpo_prompt_template_version") != DPO_PROMPT_TEMPLATE_VERSION:
        return "dpo_prompt_template_mismatch"
    if metadata.get("style_preset") != style_preset:
        return "style_preset_mismatch"
    actual_candidates = int(metadata.get("rejected_candidates", -1))
    allowed_candidates = allowed_source_candidates or {candidates}
    if actual_candidates not in allowed_candidates:
        return "candidate_count_mismatch"
    if int(metadata.get("seed", -1)) != seed:
        return "seed_mismatch"
    if passes_thresholds(
        record,
        min_score_gap=min_score_gap,
        min_chosen_posterior=min_chosen_posterior,
        max_rejected_posterior=max_rejected_posterior,
    ) != "strict":
        return "threshold_mismatch"
    return None


def _validate_skipped_record(
    record: dict[str, Any],
    selection: dict[str, Any],
    **kwargs: Any,
) -> str | None:
    """候補数以外が一致する確定済みthreshold skipだけを再利用する。"""
    skip_reason = str(record.get("skip_reason", ""))
    if skip_reason not in {"low_chosen", "high_rejected", "small_gap"}:
        return "non_threshold_skip"
    # 共通条件を検証するため、一時的に採択条件を満たす値へ差し替える。
    probe = dict(record)
    probe["score_chosen"] = 1.0
    probe["score_rejected"] = 0.0
    probe["score_gap"] = 1.0
    return _validate_record(probe, selection, **kwargs)


def reuse_accepted_records(
    *,
    source_output: Path,
    current_selection: Path,
    bayes_model: Path,
    output: Path,
    source_skipped: Path | None = None,
    skipped_output: Path | None = None,
    manifest: Path,
    generation_model: str,
    scoring_model: str,
    style_preset: str,
    candidates: int,
    allowed_source_candidates: set[int] | None = None,
    reuse_threshold_skips: bool = False,
    seed: int,
    max_source_characters: int,
    min_score_gap: float,
    min_chosen_posterior: float,
    max_rejected_posterior: float,
) -> dict[str, Any]:
    """条件一致した採択済みレコードだけを現在runへ統合する。"""
    load_transition_bayes_model(bayes_model)
    model_version = bayes_model_version(bayes_model)
    selections = {source_record_key(row): row for row in _read(current_selection)}
    existing = _read(output, missing_ok=True)
    best = {dpo_record_key(row): row for row in existing}
    reasons: Counter[str] = Counter()
    inherited = 0

    for record in _read(source_output):
        try:
            key = dpo_record_key(record)
        except (KeyError, TypeError, ValueError):
            reasons["invalid_source_key"] += 1
            continue
        selection = selections.get(key)
        if selection is None:
            reasons["not_in_current_selection"] += 1
            continue
        reason = _validate_record(
            record,
            selection,
            model_version=model_version,
            generation_model=generation_model,
            scoring_model=scoring_model,
            style_preset=style_preset,
            candidates=candidates,
            allowed_source_candidates=allowed_source_candidates,
            seed=seed,
            max_source_characters=max_source_characters,
            min_score_gap=min_score_gap,
            min_chosen_posterior=min_chosen_posterior,
            max_rejected_posterior=max_rejected_posterior,
        )
        if reason is not None:
            reasons[reason] += 1
            continue
        current = best.get(key)
        if current is None:
            copied = dict(record)
            copied_metadata = copied.setdefault("metadata", {})
            copied_metadata["reused_from_dpo_output"] = str(source_output)
            copied_metadata["continued_into_rejected_candidates"] = candidates
            copied_metadata["mixed_candidate_continuation"] = (
                int(copied_metadata.get("rejected_candidates", -1)) != candidates
            )
            best[key] = copied
            inherited += 1
        else:
            reasons["already_present"] += 1

    reused_skips = 0
    skip_reasons: Counter[str] = Counter()
    if reuse_threshold_skips:
        if source_skipped is None or skipped_output is None:
            raise ValueError("threshold skip再利用にはsource_skippedとskipped_outputが必要です。")
        existing_skips = _read(skipped_output, missing_ok=True)
        best_skips = {
            dpo_record_key(row): row
            for row in existing_skips
            if "source_dialogue_id" in row and "turn_index" in row
        }
        for record in _read(source_skipped, missing_ok=True):
            try:
                key = dpo_record_key(record)
            except (KeyError, TypeError, ValueError):
                skip_reasons["invalid_source_key"] += 1
                continue
            if key in best:
                skip_reasons["already_accepted"] += 1
                continue
            selection = selections.get(key)
            if selection is None:
                skip_reasons["not_in_current_selection"] += 1
                continue
            reason = _validate_skipped_record(
                record,
                selection,
                model_version=model_version,
                generation_model=generation_model,
                scoring_model=scoring_model,
                style_preset=style_preset,
                candidates=candidates,
                allowed_source_candidates=allowed_source_candidates,
                seed=seed,
                max_source_characters=max_source_characters,
                min_score_gap=min_score_gap,
                min_chosen_posterior=min_chosen_posterior,
                max_rejected_posterior=max_rejected_posterior,
            )
            if reason is not None:
                skip_reasons[reason] += 1
                continue
            if key in best_skips:
                skip_reasons["already_present"] += 1
                continue
            copied = dict(record)
            copied_metadata = copied.setdefault("metadata", {})
            copied_metadata["reused_from_dpo_skipped"] = str(source_skipped)
            copied_metadata["continued_into_rejected_candidates"] = candidates
            copied_metadata["mixed_candidate_continuation"] = (
                int(copied_metadata.get("rejected_candidates", -1)) != candidates
            )
            best_skips[key] = copied
            reused_skips += 1
        skipped_output.parent.mkdir(parents=True, exist_ok=True)
        with skipped_output.open("w", encoding="utf-8") as file:
            for record in sorted(best_skips.values(), key=dpo_record_key):
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(best.values(), key=lambda row: dpo_record_key(row))
    with output.open("w", encoding="utf-8") as file:
        for record in ordered:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    payload = {
        "source_output": str(source_output),
        "source_output_sha256": _sha256(source_output),
        "current_selection": str(current_selection),
        "current_selection_sha256": _sha256(current_selection),
        "bayes_model": str(bayes_model),
        "bayes_model_version": model_version,
        "source_accepted_records": sum(1 for _ in source_output.open(encoding="utf-8")),
        "existing_current_records": len(existing),
        "inherited_records": inherited,
        "output_records": len(ordered),
        "rejected_reasons": dict(sorted(reasons.items())),
        "skipped_records_reused": reused_skips,
        "skipped_rejected_reasons": dict(sorted(skip_reasons.items())),
        "source_skipped": str(source_skipped) if source_skipped else None,
        "source_skipped_sha256": (
            _sha256(source_skipped)
            if source_skipped is not None and source_skipped.is_file()
            else None
        ),
        "conditions": {
            "generation_model": generation_model,
            "scoring_model": scoring_model,
            "style_preset": style_preset,
            "candidates": candidates,
            "allowed_source_candidates": sorted(allowed_source_candidates or {candidates}),
            "mixed_candidate_continuation": bool(
                allowed_source_candidates and allowed_source_candidates != {candidates}
            ),
            "seed": seed,
            "max_source_characters": max_source_characters,
            "min_score_gap": min_score_gap,
            "min_chosen_posterior": min_chosen_posterior,
            "max_rejected_posterior": max_rejected_posterior,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "dpo_prompt_template_version": DPO_PROMPT_TEMPLATE_VERSION,
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    args = parse_args()
    payload = reuse_accepted_records(
        source_output=Path(args.source_output),
        current_selection=Path(args.current_selection),
        bayes_model=Path(args.bayes_model),
        output=Path(args.output),
        source_skipped=Path(args.source_skipped) if args.source_skipped else None,
        skipped_output=Path(args.skipped_output) if args.skipped_output else None,
        manifest=Path(args.manifest),
        generation_model=args.generation_model,
        scoring_model=args.scoring_model,
        style_preset=args.style_preset,
        candidates=args.candidates,
        allowed_source_candidates=set(args.source_candidates) or {args.candidates},
        reuse_threshold_skips=args.reuse_threshold_skips,
        seed=args.seed,
        max_source_characters=args.max_source_characters,
        min_score_gap=args.min_score_gap,
        min_chosen_posterior=args.min_chosen_posterior,
        max_rejected_posterior=args.max_rejected_posterior,
    )
    print(
        "[reuse dpo] "
        f"inherited={payload['inherited_records']} "
        f"existing={payload['existing_current_records']} "
        f"output={payload['output_records']} "
        f"skips={payload['skipped_records_reused']} "
        f"rejected={payload['rejected_reasons']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
