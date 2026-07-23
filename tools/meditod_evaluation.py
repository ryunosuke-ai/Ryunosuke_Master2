"""MediTOD held-out評価promptの固定、日本語化、3モデル応答生成。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from core.adaptive_request_pacer import AdaptiveRequestPacer
from core.dpo_prompting import (
    DEFAULT_MAX_HISTORY_TURNS,
    MEDITOD_DPO_PROMPT_TEMPLATE_VERSION,
    build_meditod_dpo_prompt,
)
from tools.analyze_small_corpus import OpenAIResponsesGenerator, extract_json_object
from tools.run_oracle_evaluation_lora_pair import (
    BASE_ADAPTER_NAME,
    DPO_ADAPTER_NAME,
    generate_reply_with_adapter,
    load_lora_pair_bundle,
)
from tools.score_dialogue_with_bayes_model import resolve_scoring_model
from tools.translate_and_generate_dpo import missing_meditod_numeric_tokens
from tools.wildchat_health import MEDICATION_PATTERN


TRANSLATION_VERSION = "meditod_eval_translation_v1"
TRANSLATION_FIDELITY_VERSION = "meditod_eval_medical_fidelity_v2"
STRATA = (
    "symptom_attributes",
    "associated_symptoms",
    "medical_history",
    "medication_or_tests",
    "lifestyle_or_exposure",
    "stage_transition",
)
ENGLISH_NEGATION = re.compile(
    r"\b(?:no|not|never|without|denies?|negative for|doesn't|don't|isn't|hasn't|haven't)\b",
    re.IGNORECASE,
)
JAPANESE_NEGATION = re.compile(
    r"(?:ない|なく|なかっ|ません|いません|ず|否定|認め(?:ない|ません)|なし|no|not|never|without)",
    re.IGNORECASE,
)
TEMPORAL_SOURCE = re.compile(
    r"\b(?:today|yesterday|day|days|week|weeks|month|months|year|years|hour|hours|"
    r"minute|minutes|since|ago|before|after|morning|night)\b",
    re.IGNORECASE,
)
TEMPORAL_TRANSLATION = re.compile(
    r"(?:今日|昨日|日|週間?|週|か月|ヶ月|月間|年|時間|分|以来|前|後|朝|夜|"
    r"today|yesterday|days?|weeks?|months?|years?|hours?|minutes?|since|ago|before|after)",
    re.IGNORECASE,
)
UNIT_SOURCE = re.compile(r"\b(?:mg|mcg|g|kg|ml|l|cm|mm|°?c|bpm|mmhg)\b", re.IGNORECASE)
SYMPTOM_CONCEPTS = {
    "cough": ("cough", "咳", "せき"),
    "fever": ("fever", "発熱", "熱"),
    "pain": ("pain", "ache", "痛"),
    "rash": ("rash", "発疹", "皮疹"),
    "vomiting": ("vomit", "vomiting", "嘔吐", "吐き"),
    "diarrhea": ("diarrhea", "下痢"),
    "headache": ("headache", "頭痛"),
    "dizziness": ("dizzy", "dizziness", "めまい"),
    "breathlessness": ("breathless", "shortness of breath", "息苦", "呼吸困難"),
    "nausea": ("nausea", "吐き気", "悪心"),
    "sputum": ("sputum", "phlegm", "痰"),
    "bleeding": ("bleed", "blood", "出血", "血"),
    "swelling": ("swelling", "swollen", "腫れ", "腫脹"),
    "fatigue": ("fatigue", "tired", "倦怠", "疲れ"),
}


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def sample_stratum(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata", {})
    slots = " ".join(map(str, metadata.get("response_slots", []))).lower()
    attributes = set(map(str, metadata.get("response_attributes", [])))
    intents = set(map(str, metadata.get("response_intents", [])))
    if any(value in slots for value in ("medication", "medical test", "test")):
        return "medication_or_tests"
    if any(value in slots for value in ("habit", "exposure", "travel", "occupation", "residence")):
        return "lifestyle_or_exposure"
    if "history" in slots:
        return "medical_history"
    if "symptom" in slots and attributes:
        return "symptom_attributes"
    if "symptom" in slots:
        return "associated_symptoms"
    if intents.intersection({"diagnosis", "diagnose", "inform", "other"}):
        return "stage_transition"
    return "stage_transition"


def select_eval_prompts(
    samples: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    ood: bool,
    max_per_consultation: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [
        sample for sample in samples
        if sample.get("metadata", {}).get("split") == "test"
        and bool(sample.get("metadata", {}).get("ood")) is ood
        and sample.get("metadata", {}).get("history_ends_with_user")
        and sample.get("next_user_turn") is not None
    ]
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in candidates:
        by_stratum[sample_stratum(sample)].append(sample)
    for values in by_stratum.values():
        values.sort(key=lambda row: _rank(seed, row["sample_id"]))
    quota = {name: count // len(STRATA) for name in STRATA}
    for name in STRATA[: count % len(STRATA)]:
        quota[name] += 1
    selected: list[dict[str, Any]] = []
    per_consultation: Counter[str] = Counter()
    selected_ids = set()
    for stratum in STRATA:
        for sample in by_stratum[stratum]:
            if sum(sample_stratum(row) == stratum for row in selected) >= quota[stratum]:
                break
            if per_consultation[sample["conversation_id"]] >= max_per_consultation:
                continue
            selected.append(sample)
            selected_ids.add(sample["sample_id"])
            per_consultation[sample["conversation_id"]] += 1
    if len(selected) < count:
        remaining = sorted(
            (sample for sample in candidates if sample["sample_id"] not in selected_ids),
            key=lambda row: _rank(seed + 1, row["sample_id"]),
        )
        for sample in remaining:
            if per_consultation[sample["conversation_id"]] >= max_per_consultation:
                continue
            selected.append(sample)
            per_consultation[sample["conversation_id"]] += 1
            if len(selected) >= count:
                break
    if len(selected) != count:
        raise ValueError(f"MediTOD評価promptが不足しています: {len(selected)}/{count}")
    selected.sort(key=lambda row: _rank(seed + 2, row["sample_id"]))
    rows = []
    for index, sample in enumerate(selected):
        rows.append(
            {
                "prompt_id": f"meditod_{'ood' if ood else 'eval'}_{index:03d}",
                "sample_id": sample["sample_id"],
                "conversation_id": sample["conversation_id"],
                "history_en": sample["history"],
                "reference_response_en": sample["response"],
                "next_patient_turn_en": sample.get("next_user_turn"),
                "selection_stratum": sample_stratum(sample),
                "source_response_intents": sample["metadata"].get("response_intents", []),
                "source_response_slots": sample["metadata"].get("response_slots", []),
                "source_response_attributes": sample["metadata"].get("response_attributes", []),
                "ood": ood,
            }
        )
    manifest = {
        "sampling_version": "meditod_eval_stratified_v1",
        "count": count,
        "seed": seed,
        "ood": ood,
        "max_per_consultation": max_per_consultation,
        "consultations": len(per_consultation),
        "stratum_counts": dict(Counter(row["selection_stratum"] for row in rows)),
        "sample_ids_sha256": hashlib.sha256("\n".join(row["sample_id"] for row in rows).encode()).hexdigest(),
        "selection_uses_model_outputs": False,
        "selection_uses_oracle_scores": False,
    }
    return rows, manifest


def translation_instructions() -> str:
    return (
        "医療相談の会話履歴、参照医療者応答、次の患者発話を自然な日本語へ忠実に翻訳してください。"
        "話者roleと発話数、否定、時期、数値、単位、薬剤名、症状名を保持し、診断や助言を追加しないでください。"
        "JSONのみでhistory_ja, reference_response_ja, next_patient_turn_jaを返してください。"
    )


def validate_translation(source: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    history = payload.get("history_ja")
    if not isinstance(history, list) or len(history) != len(source["history_en"]):
        raise ValueError("MediTOD評価履歴の翻訳前後で発話数が一致しません。")
    normalized = []
    for original, translated in zip(source["history_en"], history):
        if not isinstance(translated, dict) or translated.get("role") != original["role"] or not str(translated.get("text", "")).strip():
            raise ValueError("MediTOD評価履歴のrole順または本文が不正です。")
        normalized.append({"role": original["role"], "text": str(translated["text"]).strip()})
    reference = str(payload.get("reference_response_ja", "")).strip()
    next_patient = str(payload.get("next_patient_turn_ja", "")).strip()
    if not reference or not next_patient:
        raise ValueError("MediTOD評価の参照応答または次患者発話が空です。")
    return {"history_ja": normalized, "reference_response_ja": reference, "next_patient_turn_ja": next_patient}


def evaluation_translation_fidelity_errors(
    source: dict[str, Any], translated: dict[str, Any]
) -> dict[str, list[str]]:
    """評価翻訳で医学的に重要な表現が欠落・反転していないか調べる。"""
    source_text = " ".join(
        [str(turn.get("text", "")) for turn in source["history_en"]]
        + [str(source["reference_response_en"]), str(source["next_patient_turn_en"])]
    )
    translated_text = " ".join(
        [str(turn.get("text", "")) for turn in translated["history_ja"]]
        + [
            str(translated["reference_response_ja"]),
            str(translated["next_patient_turn_ja"]),
        ]
    )
    errors: dict[str, list[str]] = {}
    # MediTOD本体は患者・医療者対話なので、引用番号を除く数値を厳格に保持する。
    missing_numbers = missing_meditod_numeric_tokens(
        source_text,
        translated_text,
        strict_personal=True,
    )
    if missing_numbers:
        errors["numbers"] = missing_numbers
    if ENGLISH_NEGATION.search(source_text) and not JAPANESE_NEGATION.search(translated_text):
        errors["negation"] = ["negation"]
    if TEMPORAL_SOURCE.search(source_text) and not TEMPORAL_TRANSLATION.search(translated_text):
        errors["temporal_expression"] = ["time"]
    units = list(dict.fromkeys(match.group(0).casefold() for match in UNIT_SOURCE.finditer(source_text)))
    missing_units = [unit for unit in units if unit not in translated_text.casefold()]
    if missing_units:
        errors["units"] = missing_units
    medications = list(
        dict.fromkeys(match.group(0) for match in MEDICATION_PATTERN.finditer(source_text))
    )
    missing_medications = [
        value for value in medications if value.casefold() not in translated_text.casefold()
    ]
    if missing_medications:
        errors["medications"] = missing_medications
    lowered_source = source_text.casefold()
    lowered_translation = translated_text.casefold()
    missing_symptoms = []
    for concept, variants in SYMPTOM_CONCEPTS.items():
        if any(value.casefold() in lowered_source for value in variants) and not any(
            value.casefold() in lowered_translation for value in variants
        ):
            missing_symptoms.append(concept)
    if missing_symptoms:
        errors["symptoms"] = missing_symptoms
    return errors


def _generate_translation_payload(
    row: dict[str, Any],
    *,
    generator: OpenAIResponsesGenerator,
    model: str,
    instructions: str,
    pacer: AdaptiveRequestPacer,
) -> dict[str, Any]:
    """一時API失敗を再試行し、JSONを取得する。"""
    last_error: Exception | None = None
    for attempt, delay in enumerate((15, 30, 60), start=1):
        try:
            pacer.wait()
            raw = generator.generate(
                instructions=instructions,
                input_text=json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "history_en",
                            "reference_response_en",
                            "next_patient_turn_en",
                        )
                    },
                    ensure_ascii=False,
                ),
                model=model,
                max_output_tokens=8000,
                response_text_format={"type": "json_object"},
            )
            pacer.record_success()
            return extract_json_object(raw)
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
            print(
                f"[meditod_eval_translate] retry {row['prompt_id']} "
                f"attempt={attempt}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            if "ratelimit" in type(exc).__name__.casefold() or "429" in str(exc):
                pacer.record_rate_limit(float(delay))
            else:
                time.sleep(delay)
    assert last_error is not None
    raise last_error


def _translate_one(
    row: dict[str, Any],
    *,
    model: str,
    mock: bool,
    pacer: AdaptiveRequestPacer,
) -> dict[str, Any]:
    """1評価promptを翻訳し、重要情報欠落時は1回だけ修復する。"""
    if mock:
        payload = {
            "history_ja": [
                {"role": turn["role"], "text": f"[日本語訳] {turn['text']}"}
                for turn in row["history_en"]
            ],
            "reference_response_ja": f"[日本語訳] {row['reference_response_en']}",
            "next_patient_turn_ja": f"[日本語訳] {row['next_patient_turn_en']}",
        }
    else:
        generator = OpenAIResponsesGenerator()
        payload = _generate_translation_payload(
            row,
            generator=generator,
            model=model,
            instructions=translation_instructions(),
            pacer=pacer,
        )
    translated = validate_translation(row, payload)
    errors = evaluation_translation_fidelity_errors(row, translated)
    repaired = False
    if errors and not mock:
        repair_instructions = (
            translation_instructions()
            + "\n前回の翻訳で次の重要情報が欠落した可能性があります。省略や意味の反転をせず、"
            "元の対応箇所へ忠実に保持してください。診断や助言は追加しないでください。\n"
            + json.dumps(errors, ensure_ascii=False)
        )
        payload = _generate_translation_payload(
            row,
            generator=generator,
            model=model,
            instructions=repair_instructions,
            pacer=pacer,
        )
        translated = validate_translation(row, payload)
        repaired = True
        errors = evaluation_translation_fidelity_errors(row, translated)
    if errors:
        raise ValueError(f"MediTOD評価翻訳で医療情報が失われました: {errors}")
    prepared = {
        **row,
        **translated,
        "history": translated["history_ja"],
        "translation_model": model,
        "translation_version": TRANSLATION_VERSION,
        "translation_fidelity_version": TRANSLATION_FIDELITY_VERSION,
        "translation_repaired": repaired,
    }
    prepared["model_prompt"] = build_model_prompt(prepared)
    prepared["model_prompt_template_version"] = MEDITOD_DPO_PROMPT_TEMPLATE_VERSION
    return prepared


def build_model_prompt(row: dict[str, Any]) -> str:
    history_turns = [
        {
            "speaker": "AI" if turn["role"] == "assistant" else "User",
            "text": turn["text"],
        }
        for turn in row["history_ja"]
    ]
    return build_meditod_dpo_prompt(
        history_turns=history_turns[-DEFAULT_MAX_HISTORY_TURNS:],
        max_history_turns=min(len(history_turns), DEFAULT_MAX_HISTORY_TURNS),
    )


def append_error(path: Path | str | None, row: dict[str, Any], exc: Exception) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"prompt_id": row.get("prompt_id"), "sample_id": row.get("sample_id"), "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False) + "\n")


def translate_prompts(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    errors_path: Path | None,
    model: str,
    mock: bool,
    resume: bool,
    workers: int,
    requests_per_minute: float,
) -> list[dict[str, Any]]:
    existing_rows = read_jsonl(output_path) if resume and output_path.exists() else []
    existing = {row["prompt_id"]: row for row in existing_rows}
    output = list(existing.values())
    pending = [row for row in rows if row["prompt_id"] not in existing]
    pacer = AdaptiveRequestPacer(
        requests_per_minute=max(0.0, requests_per_minute),
        initial_backoff_seconds=15.0,
    )
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _translate_one,
                row,
                model=model,
                mock=mock,
                pacer=pacer,
            ): row
            for row in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                output.append(future.result())
                output.sort(key=lambda value: value["prompt_id"])
                write_jsonl(output, output_path)
                print(
                    f"[meditod_eval_translate] completed {completed}/{len(pending)} "
                    f"{row['prompt_id']}",
                    flush=True,
                )
            except Exception as exc:
                append_error(errors_path, row, exc)
                print(f"[meditod_eval_translate] skip {row['prompt_id']}: {exc}", flush=True)
    return sorted(output, key=lambda row: row["prompt_id"])


def generate_three_model_responses(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    oracle_path: Path,
    errors_path: Path | None,
    base_model: str,
    basis_lora: str,
    random_lora: str,
    seed: int,
    mock: bool,
) -> list[dict[str, Any]]:
    existing_rows = read_jsonl(output_path) if output_path.exists() else []
    existing = {row["prompt_id"]: row for row in existing_rows}
    output = list(existing.values())
    bundle = None if mock else load_lora_pair_bundle(
        base_model, base_lora_path=basis_lora, dpo_lora_path=random_lora, use_4bit=False
    )
    for index, row in enumerate(rows):
        if row["prompt_id"] in existing:
            continue
        try:
            prompt = build_model_prompt(row)
            if mock:
                responses = {
                    "base_response": "症状についてもう少し教えてください。",
                    "basis_response": "その症状はいつ始まり、時間とともに変化していますか。",
                    "random_dpo_response": "心配なら病院へ行ってください。",
                }
            else:
                assert bundle is not None
                disable = getattr(bundle.model, "disable_adapter", None)
                if disable is None:
                    raise RuntimeError("Base応答生成に必要なdisable_adapterがありません。")
                with disable():
                    base = generate_reply_with_adapter(bundle, prompt, adapter_name=None, max_new_tokens=256, temperature=0.7, top_p=0.9, repetition_penalty=1.05, seed=seed + index)
                basis = generate_reply_with_adapter(bundle, prompt, adapter_name=BASE_ADAPTER_NAME, max_new_tokens=256, temperature=0.7, top_p=0.9, repetition_penalty=1.05, seed=seed + index)
                random_response = generate_reply_with_adapter(bundle, prompt, adapter_name=DPO_ADAPTER_NAME, max_new_tokens=256, temperature=0.7, top_p=0.9, repetition_penalty=1.05, seed=seed + index)
                responses = {"base_response": base, "basis_response": basis, "random_dpo_response": random_response}
            if any(not str(value).strip() for value in responses.values()):
                raise ValueError("3モデルのいずれかが空応答です。")
            order = ["base", "basis", "random_dpo"]
            random.Random(f"{seed}:{row['prompt_id']}").shuffle(order)
            output.append(
                {
                    **row,
                    **responses,
                    "model_prompt": prompt,
                    "response_order": order,
                    "generation_seed": seed + index,
                    "base_model_id": base_model,
                    "basis_lora_path": basis_lora,
                    "random_lora_path": random_lora,
                }
            )
            write_jsonl(output, output_path)
        except Exception as exc:
            append_error(errors_path, row, exc)
            print(f"[meditod_eval_generate] skip {row['prompt_id']}: {exc}", flush=True)
    output.sort(key=lambda row: row["prompt_id"])
    write_jsonl(output, output_path)
    write_jsonl(blind_oracle_rows(output), oracle_path)
    return output


def blind_oracle_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response_keys = {"base": "base_response", "basis": "basis_response", "random_dpo": "random_dpo_response"}
    output = []
    for row in rows:
        oracle_context = (
            f"参照医療者応答: {row['reference_response_ja']}\n"
            f"その後の患者発話: {row['next_patient_turn_ja']}\n"
            f"Reference clinician response: {row['reference_response_en']}\n"
            f"Following patient turn: {row['next_patient_turn_en']}"
        )
        for position, model in enumerate(row["response_order"]):
            output.append(
                {
                    "sample_id": row["sample_id"],
                    "model_name": model,
                    "prompt": oracle_context,
                    "history": row["history_ja"],
                    "response": row[response_keys[model]],
                    "metadata": {
                        "blind_position": position,
                        "prompt_version": "meditod_eval_prompt_v1",
                        "model_prompt_template_version": MEDITOD_DPO_PROMPT_TEMPLATE_VERSION,
                        "conversation_id": row["conversation_id"],
                        "selection_stratum": row["selection_stratum"],
                        "source_response_intents": row["source_response_intents"],
                        "source_response_slots": row["source_response_slots"],
                        "source_response_attributes": row["source_response_attributes"],
                        "ood": row["ood"],
                    },
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="MediTOD日本語評価データ・3モデル応答生成")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--samples", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--errors-output")
    prepare.add_argument("--count", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--ood", action="store_true")
    prepare.add_argument("--max-per-consultation", type=int, default=6)
    prepare.add_argument("--model", default=resolve_scoring_model())
    prepare.add_argument("--workers", type=int, default=4)
    prepare.add_argument("--requests-per-minute", type=float, default=120.0)
    prepare.add_argument("--resume", action="store_true")
    prepare.add_argument("--mock", action="store_true")
    generate = sub.add_parser("generate")
    generate.add_argument("--input", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--oracle-output", required=True)
    generate.add_argument("--errors-output")
    generate.add_argument("--base-model", required=True)
    generate.add_argument("--basis-lora", required=True)
    generate.add_argument("--random-lora", required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        selected, manifest = select_eval_prompts(
            read_jsonl(args.samples),
            count=args.count,
            seed=args.seed,
            ood=args.ood,
            max_per_consultation=args.max_per_consultation,
        )
        translated = translate_prompts(
            selected,
            output_path=Path(args.output),
            errors_path=Path(args.errors_output) if args.errors_output else None,
            model=args.model,
            mock=args.mock,
            resume=args.resume,
            workers=args.workers,
            requests_per_minute=args.requests_per_minute,
        )
        if len(translated) != args.count:
            raise RuntimeError(f"MediTOD評価翻訳が不足しています: {len(translated)}/{args.count}")
        Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    rows = generate_three_model_responses(
        read_jsonl(args.input),
        output_path=Path(args.output),
        oracle_path=Path(args.oracle_output),
        errors_path=Path(args.errors_output) if args.errors_output else None,
        base_model=args.base_model,
        basis_lora=args.basis_lora,
        random_lora=args.random_lora,
        seed=args.seed,
        mock=args.mock,
    )
    if len(rows) != len(read_jsonl(args.input)):
        raise RuntimeError("MediTOD 3モデル応答が不足しています。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
