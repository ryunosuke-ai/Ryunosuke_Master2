"""MathDial特徴抽出、品質検証、ベイズモデル構築CLI。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from core.mathdial_basis import (
    build_basis_model,
    build_extraction_instructions,
    build_transition_compat_model,
    build_validation_instructions,
    canonical_hash,
    format_extraction_input,
    load_yaml,
    validate_extraction,
    validate_ontology,
)
from core.transition_bayes_model import parse_transition_bayes_model
from tools.analyze_small_corpus import OpenAIResponsesGenerator, extract_json_object, resolve_analysis_model


DEFAULT_ONTOLOGY = "configs/ontologies/mathdial_v1.yaml"


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}をJSONとして読めません: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}はobjectである必要があります。")
            rows.append(row)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(row: dict[str, Any], path: Path | str) -> None:
    """中断時にもresumeできるよう1件を直ちに追記する。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def normalize_extraction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """意味を変えない範囲でLLMのJSON型揺れを正規化する。"""
    normalized = dict(payload)
    confidence = normalized.get("confidence")
    if isinstance(confidence, str):
        try:
            normalized["confidence"] = float(confidence.strip())
        except ValueError:
            pass
    return normalized


def _mock_extraction(sample: dict[str, Any], ontology: dict[str, Any]) -> dict[str, Any]:
    text = (" ".join(turn["text"] for turn in sample["history"][-2:]) + " " + sample["response"]).lower()
    strategy = "probing_question" if "?" in sample["response"] else "explanation"
    if any(token in text for token in ("wrong", "incorrect", "not quite", "check again")):
        before = "misconception"
    elif any(token in text for token in ("not sure", "confused", "don't understand")):
        before = "uncertainty"
    else:
        before = "partial_understanding"
    after = "unobserved" if sample.get("next_user_turn") is None else "partial_understanding"
    return validate_extraction({
        "student_state_before": before,
        "tutor_strategy": strategy,
        "student_state_after": after,
        "conversation_stage": "guided_reasoning",
        "style_features": ["student_state_grounding", "elicits_reasoning"] if strategy == "probing_question" else ["targeted_feedback"],
        "confidence": 0.8,
        "short_reason": "Deterministic mock extraction for pipeline validation.",
    }, ontology)


def extract_features(
    samples: list[dict[str, Any]], conversations: list[dict[str, Any]], ontology: dict[str, Any], *,
    generator: Any | None, model: str, limit: int | None, resume_rows: list[dict[str, Any]], mock: bool,
    max_attempts: int = 3, workers: int = 1,
    on_success: Any | None = None, on_error: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """resume可能な特徴抽出を行う。"""
    conversations_by_id = {row["conversation_id"]: row for row in conversations}
    existing = {row["sample_id"]: row for row in resume_rows}
    output = list(resume_rows)
    errors: list[dict[str, Any]] = []
    pending = [sample for sample in samples if sample.get("metadata", {}).get("split") == "train" and sample["sample_id"] not in existing]
    if limit is not None:
        pending = pending[:limit]
    instructions = build_extraction_instructions(ontology)
    total = len(pending)
    def extract_one(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        index, sample = item
        last_error: Exception | None = None
        for attempt in range(1, max(1, max_attempts) + 1):
            try:
                if mock:
                    extracted = _mock_extraction(sample, ontology)
                else:
                    raw = generator.generate(instructions=instructions, input_text=format_extraction_input(sample, conversations_by_id.get(sample["conversation_id"])), model=model, max_output_tokens=2000, response_text_format={"type": "json_object"})
                    payload = normalize_extraction_payload(extract_json_object(raw))
                    extracted = validate_extraction(payload, ontology)
                break
            except Exception as exc:
                last_error = exc
                if attempt < max(1, max_attempts):
                    print(f"[extract_features] retry {index}/{total} attempt={attempt + 1}: {type(exc).__name__}: {exc}", flush=True)
        else:
            error = {"sample_id": sample.get("sample_id"), "error_type": type(last_error).__name__, "error": str(last_error), "attempts": max(1, max_attempts)}
            return index, sample, None, error

        try:
            extracted.update({
                "sample_id": sample["sample_id"],
                "conversation_id": sample["conversation_id"],
                "split": sample["metadata"]["split"],
                "assistant_turn_index": sample["metadata"]["assistant_turn_index"],
                "teacher_moves": sample["metadata"].get("teacher_moves", []),
                "validation_status": "unvalidated",
                "ontology_version": ontology["version"],
            })
            return index, sample, extracted, None
        except Exception as exc:
            error = {"sample_id": sample.get("sample_id"), "error_type": type(exc).__name__, "error": str(exc), "attempts": 1}
            return index, sample, None, error

    indexed_pending = list(enumerate(pending, 1))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        # mapは入力順で結果を返すため、並列数を変えてもJSONL順序は一定になる。
        for index, sample, extracted, error in executor.map(extract_one, indexed_pending):
            if extracted is not None:
                output.append(extracted)
                if on_success:
                    on_success(extracted)
                print(f"[extract_features] completed {index}/{total} sample={sample['sample_id']}", flush=True)
                continue
            assert error is not None
            errors.append(error)
            if on_error:
                on_error(error)
            print(f"[extract_features] error {index}/{total} sample={sample.get('sample_id')}: {error['error']}", flush=True)
    return output, errors


def validate_with_llm(
    rows: list[dict[str, Any]], samples: list[dict[str, Any]], conversations: list[dict[str, Any]],
    ontology: dict[str, Any], *, generator: Any | None, model: str, mock: bool, mode: str,
    workers: int = 1, max_attempts: int = 3,
    resume_rows: list[dict[str, Any]] | None = None,
    on_success: Any | None = None, on_error: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """特徴抽出とは独立したpromptで抽出結果を検証する。"""
    samples_by_id = {row["sample_id"]: row for row in samples}
    conversations_by_id = {row["conversation_id"]: row for row in conversations}
    instructions = build_validation_instructions(ontology)
    output, errors = list(resume_rows or []), []
    done_ids = {row["sample_id"] for row in output}
    pending_rows = [row for row in rows if row["sample_id"] not in done_ids]
    threshold = float(ontology.get("confidence_threshold", 0.6))
    total = len(pending_rows)

    def validate_one(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
        index, row = item
        if mode == "low_confidence" and float(row["confidence"]) >= threshold:
            return index, {**row, "validation_status": "valid", "validation_method": "confidence_bypass"}, None
        sample = samples_by_id.get(row["sample_id"])
        if sample is None:
            return index, None, {"sample_id": row["sample_id"], "error_type": "missing_sample", "attempts": 1}
        last_error: Exception | None = None
        for attempt in range(1, max(1, max_attempts) + 1):
            try:
                if mock:
                    payload = {"valid": True, "corrected_extraction": {key: row[key] for key in ("student_state_before", "tutor_strategy", "student_state_after", "conversation_stage", "style_features", "confidence", "short_reason")}, "confidence": row["confidence"], "short_reason": "mock validation"}
                else:
                    validator_input = {
                        "transcript": json.loads(format_extraction_input(sample, conversations_by_id.get(row["conversation_id"]))),
                        "proposed_extraction": {key: row[key] for key in ("student_state_before", "tutor_strategy", "student_state_after", "conversation_stage", "style_features", "confidence", "short_reason")},
                    }
                    raw = generator.generate(instructions=instructions, input_text=json.dumps(validator_input, ensure_ascii=False, indent=2), model=model, max_output_tokens=2500, response_text_format={"type": "json_object"})
                    payload = extract_json_object(raw)
                if not isinstance(payload.get("valid"), bool):
                    raise ValueError("validatorの`valid`はbooleanである必要があります。")
                corrected_payload = payload.get("corrected_extraction", {})
                if not isinstance(corrected_payload, dict):
                    raise ValueError("validatorの`corrected_extraction`はobjectである必要があります。")
                corrected = validate_extraction(normalize_extraction_payload(corrected_payload), ontology)
                validated = {**row, **corrected, "validation_status": "valid", "validation_method": "secondary_llm", "validator_original_valid": payload["valid"], "validator_confidence": float(payload.get("confidence", 0.0)), "validator_reason": str(payload.get("short_reason", ""))}
                return index, validated, None
            except Exception as exc:
                last_error = exc
                if attempt < max(1, max_attempts):
                    print(f"[validate_extraction] retry {index}/{total} attempt={attempt + 1}: {type(exc).__name__}: {exc}", flush=True)
        return index, None, {"sample_id": row["sample_id"], "error_type": type(last_error).__name__, "error": str(last_error), "attempts": max(1, max_attempts)}

    indexed_rows = list(enumerate(pending_rows, 1))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, validated, error in executor.map(validate_one, indexed_rows):
            if validated is not None:
                output.append(validated)
                if on_success:
                    on_success(validated)
                print(f"[validate_extraction] completed {index}/{total} sample={validated['sample_id']}", flush=True)
                continue
            assert error is not None
            errors.append(error)
            if on_error:
                on_error(error)
            print(f"[validate_extraction] error {index}/{total} sample={error['sample_id']}: {error.get('error', error['error_type'])}", flush=True)
    return output, errors


def stratified_quality_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Teacher move別に均等な品質検証標本を作る。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        moves = row.get("teacher_moves", [])
        for move in moves or ["unlabeled"]:
            groups[str(move)].append(row)
    rng = random.Random(seed)
    per_group = max(1, count // max(1, len(groups)))
    chosen: dict[str, dict[str, Any]] = {}
    for move in sorted(groups):
        candidates = list(groups[move])
        rng.shuffle(candidates)
        for row in candidates[:per_group]:
            chosen[row["sample_id"]] = row
    remainder = [row for row in rows if row["sample_id"] not in chosen]
    rng.shuffle(remainder)
    for row in remainder:
        if len(chosen) >= count:
            break
        chosen[row["sample_id"]] = row
    return list(chosen.values())[:count]


def strategy_to_move(strategy: str, ontology: dict[str, Any]) -> str:
    for move, strategies in ontology["teacher_move_mapping"].items():
        if strategy in strategies:
            return move
    raise ValueError(f"Teacher moveへ対応しないstrategyです: {strategy}")


def quality_metrics(rows: list[dict[str, Any]], ontology: dict[str, Any]) -> dict[str, Any]:
    """単一/多値Teacher moveに対する外部一致を集計する。"""
    labels = ["probing", "focus", "telling", "generic"]
    confusion = {gold: {pred: 0.0 for pred in labels} for gold in labels}
    strict = []
    aware_correct = 0
    evaluated = 0
    disagreements = []
    ambiguous = []
    for row in rows:
        gold = sorted(set(row.get("teacher_moves", [])))
        if not gold:
            continue
        pred = strategy_to_move(row["tutor_strategy"], ontology)
        evaluated += 1
        aware_correct += int(pred in gold)
        for label in gold:
            confusion[label][pred] += 1.0 / len(gold)
        if len(gold) == 1:
            strict.append((gold[0], pred))
        else:
            ambiguous.append({"sample_id": row["sample_id"], "gold": gold, "predicted": pred})
        if pred not in gold:
            disagreements.append({"sample_id": row["sample_id"], "gold": gold, "predicted": pred, "strategy": row["tutor_strategy"], "reason": row["short_reason"]})
    per_label = {}
    f1_values = []
    supports = []
    strict_correct = sum(gold == pred for gold, pred in strict)
    for label in labels:
        tp = sum(g == label and p == label for g, p in strict)
        fp = sum(g != label and p == label for g, p in strict)
        fn = sum(g == label and p != label for g, p in strict)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(g == label for g, _ in strict)
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1_values.append(f1)
        supports.append(support)
    weighted = sum(f * s for f, s in zip(f1_values, supports)) / sum(supports) if sum(supports) else 0.0
    return {
        "evaluated": evaluated,
        "single_label_evaluated": len(strict),
        "accuracy": strict_correct / len(strict) if strict else 0.0,
        "ambiguity_aware_accuracy": aware_correct / evaluated if evaluated else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values),
        "weighted_f1": weighted,
        "per_label": per_label,
        "confusion_matrix": confusion,
        "confidence": [row["confidence"] for row in rows],
        "disagreements": disagreements,
        "ambiguous_examples": ambiguous,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MathDial特徴抽出・検証・ベイズモデル構築")
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--samples", required=True)
    extract.add_argument("--conversations", required=True)
    extract.add_argument("--ontology", default=DEFAULT_ONTOLOGY)
    extract.add_argument("--output", required=True)
    extract.add_argument("--errors", required=True)
    extract.add_argument("--model", default=resolve_analysis_model())
    extract.add_argument("--limit", type=int)
    extract.add_argument("--resume", action="store_true")
    extract.add_argument("--mock", action="store_true")
    extract.add_argument("--max-attempts", type=int, default=3)
    extract.add_argument("--workers", type=int, default=1)
    llm_validate = sub.add_parser("llm-validate")
    llm_validate.add_argument("--input", required=True)
    llm_validate.add_argument("--samples", required=True)
    llm_validate.add_argument("--conversations", required=True)
    llm_validate.add_argument("--ontology", default=DEFAULT_ONTOLOGY)
    llm_validate.add_argument("--output", required=True)
    llm_validate.add_argument("--errors", required=True)
    llm_validate.add_argument("--model", default=resolve_analysis_model())
    llm_validate.add_argument("--mode", choices=("all", "low_confidence"), default="all")
    llm_validate.add_argument("--mock", action="store_true")
    llm_validate.add_argument("--workers", type=int, default=max(1, int(os.getenv("WORKERS", "1"))))
    llm_validate.add_argument("--max-attempts", type=int, default=3)
    llm_validate.add_argument("--resume", action="store_true", default=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--input", required=True)
    validate.add_argument("--ontology", default=DEFAULT_ONTOLOGY)
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--sample-size", type=int, default=160)
    validate.add_argument("--seed", type=int, default=42)
    build = sub.add_parser("build-model")
    build.add_argument("--input", required=True)
    build.add_argument("--ontology", default=DEFAULT_ONTOLOGY)
    build.add_argument("--output", required=True)
    build.add_argument("--compat-output", required=True)
    args = parser.parse_args()
    ontology = validate_ontology(load_yaml(args.ontology))
    if args.command == "extract":
        existing = read_jsonl(args.output) if args.resume and Path(args.output).exists() else []
        output_path = Path(args.output)
        error_path = Path(args.errors)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not (args.resume and output_path.exists()):
            output_path.write_text("", encoding="utf-8")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text("", encoding="utf-8")
        rows, errors = extract_features(
            read_jsonl(args.samples), read_jsonl(args.conversations), ontology,
            generator=None if args.mock else OpenAIResponsesGenerator(), model=args.model,
            limit=args.limit, resume_rows=existing, mock=args.mock, max_attempts=args.max_attempts,
            workers=args.workers,
            on_success=lambda row: append_jsonl(row, output_path),
            on_error=lambda row: append_jsonl(row, error_path),
        )
        new_successes = len(rows) - len(existing)
        print(f"[extract_features] summary successes={new_successes} errors={len(errors)} resumed={len(existing)}", flush=True)
        if new_successes == 0 and not existing:
            raise RuntimeError("特徴抽出の成功件数が0件です。errors.jsonlを確認してください。")
        return 0
    if args.command == "llm-validate":
        output_path = Path(args.output)
        error_path = Path(args.errors)
        existing = read_jsonl(output_path) if args.resume and output_path.exists() else []
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not (args.resume and output_path.exists()):
            output_path.write_text("", encoding="utf-8")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text("", encoding="utf-8")
        rows, errors = validate_with_llm(
            read_jsonl(args.input), read_jsonl(args.samples), read_jsonl(args.conversations), ontology,
            generator=None if args.mock else OpenAIResponsesGenerator(), model=args.model, mock=args.mock,
            mode=args.mode, workers=args.workers, max_attempts=args.max_attempts, resume_rows=existing,
            on_success=lambda row: append_jsonl(row, output_path),
            on_error=lambda row: append_jsonl(row, error_path),
        )
        new_successes = len(rows) - len(existing)
        print(f"[validate_extraction] summary successes={new_successes} errors={len(errors)} resumed={len(existing)}", flush=True)
        if new_successes == 0 and not existing:
            raise RuntimeError("特徴抽出validationの成功件数が0件です。validation_errors.jsonlを確認してください。")
        return 0
    if args.command == "validate":
        rows = stratified_quality_sample(read_jsonl(args.input), args.sample_size, args.seed)
        metrics = quality_metrics(rows, ontology)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_jsonl(rows, output / "sample.jsonl")
        markdown = ["# MathDial extraction quality", "", f"- evaluated: {metrics['evaluated']}", f"- accuracy: {metrics['accuracy']:.4f}", f"- ambiguity-aware accuracy: {metrics['ambiguity_aware_accuracy']:.4f}", f"- macro-F1: {metrics['macro_f1']:.4f}", "", "## Disagreements", ""]
        markdown.extend(f"- `{row['sample_id']}` gold={row['gold']} predicted={row['predicted']}: {row['reason']}" for row in metrics["disagreements"][:30])
        (output / "review.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
        return 0
    rows = read_jsonl(args.input)
    fine = build_basis_model(rows, ontology)
    compat = build_transition_compat_model(rows, ontology)
    parse_transition_bayes_model(compat)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(fine, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.compat_output).write_text(json.dumps(compat, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
