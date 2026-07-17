"""MathDial held-out評価promptの日本語化と3モデル応答生成。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from core.dpo_prompting import (
    CONTEXT_ONLY_DPO_PROMPT_TEMPLATE_VERSION,
    DEFAULT_MAX_HISTORY_TURNS,
    DPO_PROMPT_TEMPLATE_VERSION,
    NEUTRAL_CONVERSATION_DPO_PROMPT_TEMPLATE_VERSION,
    build_context_only_dpo_prompt,
    build_mathdial_dpo_prompt,
    build_neutral_conversation_dpo_prompt,
)
from tools.analyze_small_corpus import OpenAIResponsesGenerator, extract_json_object
from tools.score_dialogue_with_bayes_model import resolve_scoring_model
from tools.run_oracle_evaluation_lora_pair import (
    BASE_ADAPTER_NAME,
    DPO_ADAPTER_NAME,
    generate_reply_with_adapter,
    load_lora_pair_bundle,
)


TRANSLATION_VERSION = "mathdial_eval_translation_v1"
LOCAL_PROMPT_MODES = (
    "mathdial_instruction",
    "context_only",
    "neutral_conversation",
)


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


def append_error(path: Path | str | None, row: dict[str, Any], exc: Exception) -> None:
    """個別失敗を再試行可能な形で隔離する。"""
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt_id": row.get("prompt_id"),
        "sample_id": row.get("sample_id"),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    with output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def select_test_prompts(
    samples: list[dict[str, Any]],
    conversations: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    excluded_sample_ids: set[str] | None = None,
    excluded_qids: set[str] | None = None,
    stratify_teacher_moves: bool = False,
    prompt_id_prefix: str = "mathdial_eval",
) -> list[dict[str, Any]]:
    """qidと会話を重複させずheld-out test promptを固定する。"""
    conversation_by_id = {row["conversation_id"]: row for row in conversations if row["split"] == "test"}
    excluded_sample_ids = excluded_sample_ids or set()
    excluded_qids = excluded_qids or set()
    candidates = []
    for sample in samples:
        if sample.get("metadata", {}).get("split") != "test" or not sample.get("metadata", {}).get("history_ends_with_user"):
            continue
        if str(sample.get("sample_id", "")) in excluded_sample_ids:
            continue
        conversation = conversation_by_id.get(sample["conversation_id"])
        if not conversation:
            continue
        qid = str(conversation.get("metadata", {}).get("qid", ""))
        if qid in excluded_qids:
            continue
        candidates.append((sample, conversation))
    rng = random.Random(seed)
    rng.shuffle(candidates)

    if stratify_teacher_moves:
        move_order = ("probing", "focus", "telling", "generic")
        buckets = {move: [] for move in move_order}
        remainder = []
        for pair in candidates:
            moves = [
                str(move).strip().lower()
                for move in pair[0].get("metadata", {}).get("teacher_moves", [])
            ]
            assigned = next((move for move in move_order if move in moves), None)
            if assigned is None:
                remainder.append(pair)
            else:
                buckets[assigned].append(pair)
        interleaved = []
        while any(buckets.values()):
            for move in move_order:
                if buckets[move]:
                    interleaved.append(buckets[move].pop())
        candidates = interleaved + remainder

    selected, seen_qids, seen_conversations = [], set(), set()
    for sample, conversation in candidates:
        qid = str(conversation.get("metadata", {}).get("qid", ""))
        if qid in seen_qids or sample["conversation_id"] in seen_conversations:
            continue
        selected.append({
            "prompt_id": f"{prompt_id_prefix}_{len(selected):03d}",
            "sample_id": sample["sample_id"],
            "conversation_id": sample["conversation_id"],
            "qid": qid,
            "problem_en": conversation.get("metadata", {}).get("question", ""),
            "ground_truth_en": conversation.get("metadata", {}).get("ground_truth", ""),
            "history_en": sample["history"],
            "source_teacher_moves": sample["metadata"].get("teacher_moves", []),
            "split": "test",
        })
        seen_qids.add(qid)
        seen_conversations.add(sample["conversation_id"])
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise ValueError(f"qid一意なtest評価promptが不足しています: {len(selected)}/{count}")
    return selected


def exclusion_ids_from_prompts(paths: list[Path | str]) -> tuple[set[str], set[str]]:
    """既存評価promptから再利用禁止sample idとqidを読む。"""
    sample_ids: set[str] = set()
    qids: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id", "")).strip()
            qid = str(row.get("qid", "")).strip()
            if sample_id:
                sample_ids.add(sample_id)
            if qid:
                qids.add(qid)
    return sample_ids, qids


def translation_instructions() -> str:
    return (
        "Translate the structured math tutoring context into natural Japanese for a Japanese learner. "
        "Preserve roles, turn count, numbers, formulas, units, and the learner's mistakes. Do not solve or "
        "correct the problem, add hints, add tutor responses, or change the learner's level of understanding. "
        "Return JSON only with problem_ja, ground_truth_ja, and history_ja. history_ja must be an array with "
        "the exact same role sequence and number of turns as history_en."
    )


def validate_translation(source: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    history = payload.get("history_ja")
    if not isinstance(history, list) or len(history) != len(source["history_en"]):
        raise ValueError("評価履歴の翻訳前後で発話数が一致しません。")
    normalized = []
    for original, translated in zip(source["history_en"], history):
        if not isinstance(translated, dict) or translated.get("role") != original["role"] or not str(translated.get("text", "")).strip():
            raise ValueError("評価履歴のrole順または本文が不正です。")
        normalized.append({"role": original["role"], "text": str(translated["text"]).strip()})
    problem = str(payload.get("problem_ja", "")).strip()
    ground_truth = str(payload.get("ground_truth_ja", "")).strip()
    if not problem:
        raise ValueError("problem_jaが空です。")
    return {"problem_ja": problem, "ground_truth_ja": ground_truth, "history_ja": normalized}


def build_mathdial_model_prompt(
    row: dict[str, Any],
    *,
    local_prompt_mode: str,
) -> tuple[str, str]:
    """評価応答生成用promptとtemplate versionを返す。"""
    if local_prompt_mode not in LOCAL_PROMPT_MODES:
        raise ValueError(f"未知のlocal_prompt_modeです: {local_prompt_mode}")
    problem = str(row.get("problem_ja", "")).strip()
    if not problem:
        raise ValueError("評価応答生成に必要なproblem_jaが空です。")
    history = row.get("history_ja", row.get("history", []))
    if not isinstance(history, list):
        raise ValueError("評価応答生成のhistoryが配列ではありません。")
    history_turns = [
        {
            "speaker": "AI" if turn.get("role") == "assistant" else "User",
            "text": str(turn.get("text", "")),
        }
        for turn in history
        if isinstance(turn, dict) and str(turn.get("text", "")).strip()
    ]
    if local_prompt_mode == "mathdial_instruction":
        model_history = [
            {"speaker": "User", "text": f"数学問題: {problem}"},
            *history_turns,
        ]
        return build_mathdial_dpo_prompt(
            history_turns=model_history
        ), DPO_PROMPT_TEMPLATE_VERSION

    # 問題文は必ず保持し、instructionを加えず直近履歴だけを続ける。
    context_turns = [
        {"speaker": "User", "text": problem},
        *history_turns[-DEFAULT_MAX_HISTORY_TURNS:],
    ]
    builder = (
        build_context_only_dpo_prompt
        if local_prompt_mode == "context_only"
        else build_neutral_conversation_dpo_prompt
    )
    template_version = (
        CONTEXT_ONLY_DPO_PROMPT_TEMPLATE_VERSION
        if local_prompt_mode == "context_only"
        else NEUTRAL_CONVERSATION_DPO_PROMPT_TEMPLATE_VERSION
    )
    return (
        builder(
            history_turns=context_turns,
            max_history_turns=len(context_turns),
        ),
        template_version,
    )


def translate_prompts(rows: list[dict[str, Any]], *, generator: Any | None, model: str, mock: bool, existing: list[dict[str, Any]], output_path: Path | str | None = None, errors_path: Path | str | None = None, skip_sample_errors: bool = False, local_prompt_mode: str = "mathdial_instruction", target_count: int | None = None) -> list[dict[str, Any]]:
    for existing_row in existing:
        existing_mode = existing_row.get(
            "local_prompt_mode", "mathdial_instruction"
        )
        if existing_mode != local_prompt_mode:
            raise ValueError(
                "既存評価promptのlocal_prompt_modeが今回の条件と一致しません: "
                f"{existing_mode} != {local_prompt_mode}"
            )
    done = {row["prompt_id"]: row for row in existing}
    output = list(existing)
    for row in rows:
        if target_count is not None and len(output) >= target_count:
            break
        if row["prompt_id"] in done:
            continue
        try:
            if mock:
                payload = {
                    "problem_ja": f"[日本語訳] {row['problem_en']}",
                    "ground_truth_ja": f"[日本語訳] {row['ground_truth_en']}",
                    "history_ja": [{"role": turn["role"], "text": f"[日本語訳] {turn['text']}"} for turn in row["history_en"]],
                }
            else:
                raw = generator.generate(instructions=translation_instructions(), input_text=json.dumps({"problem_en": row["problem_en"], "ground_truth_en": row["ground_truth_en"], "history_en": row["history_en"]}, ensure_ascii=False), model=model, max_output_tokens=6000, response_text_format={"type": "json_object"})
                payload = extract_json_object(raw)
            translated = validate_translation(row, payload)
        except Exception as exc:
            if not skip_sample_errors:
                raise
            append_error(errors_path, row, exc)
            print(f"[mathdial_eval_translate] skip {row['prompt_id']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        prepared = {
            **row,
            **translated,
            "prompt": f"数学問題: {translated['problem_ja']}",
            "history": translated["history_ja"],
            "translation_model": model,
            "translation_version": TRANSLATION_VERSION,
        }
        model_prompt, template_version = build_mathdial_model_prompt(
            prepared,
            local_prompt_mode=local_prompt_mode,
        )
        prepared.update(
            {
                "model_prompt": model_prompt,
                "local_prompt_mode": local_prompt_mode,
                "model_prompt_template_version": template_version,
            }
        )
        output.append(prepared)
        if output_path:
            write_jsonl(output, output_path)
    return output


def generate_three_model_responses(rows: list[dict[str, Any]], *, base_model: str, basis_lora: str, random_lora: str, output_path: Path, mock: bool, seed: int, errors_path: Path | str | None = None, skip_sample_errors: bool = False, local_prompt_mode: str = "mathdial_instruction") -> list[dict[str, Any]]:
    existing_rows = read_jsonl(output_path) if output_path.exists() else []
    for existing_row in existing_rows:
        existing_mode = existing_row.get("local_prompt_mode", "mathdial_instruction")
        if existing_mode != local_prompt_mode:
            raise ValueError(
                "既存評価応答のlocal_prompt_modeが今回の条件と一致しません: "
                f"{existing_mode} != {local_prompt_mode}"
            )
    existing = {row["prompt_id"]: row for row in existing_rows}
    bundle = None if mock else load_lora_pair_bundle(base_model, base_lora_path=basis_lora, dpo_lora_path=random_lora, use_4bit=False)
    output = list(existing.values())
    for index, row in enumerate(rows):
        if row["prompt_id"] in existing:
            continue
        try:
            model_prompt, template_version = build_mathdial_model_prompt(
                row,
                local_prompt_mode=local_prompt_mode,
            )
            if mock:
                responses = {"base_response": "まず、どこまで分かったか教えてください。", "basis_response": "その考え方のどの段階で迷いましたか。次の一歩を一緒に確認しましょう。", "random_dpo_response": "問題をもう一度よく読んで計算してください。"}
            else:
                assert bundle is not None
                disable = getattr(bundle.model, "disable_adapter", None)
                if disable is None:
                    raise RuntimeError("Base応答生成に必要なdisable_adapterがありません。")
                with disable():
                    base_response = generate_reply_with_adapter(bundle, model_prompt, adapter_name=None, max_new_tokens=256, temperature=0.7, top_p=0.9, repetition_penalty=1.05, seed=seed + index)
                basis_response = generate_reply_with_adapter(bundle, model_prompt, adapter_name=BASE_ADAPTER_NAME, max_new_tokens=256, temperature=0.7, top_p=0.9, repetition_penalty=1.05, seed=seed + index)
                random_response = generate_reply_with_adapter(bundle, model_prompt, adapter_name=DPO_ADAPTER_NAME, max_new_tokens=256, temperature=0.7, top_p=0.9, repetition_penalty=1.05, seed=seed + index)
                responses = {"base_response": base_response, "basis_response": basis_response, "random_dpo_response": random_response}
        except Exception as exc:
            if not skip_sample_errors:
                raise
            append_error(errors_path, row, exc)
            print(f"[mathdial_eval_generate] skip {row['prompt_id']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        positions = ["base", "basis", "random_dpo"]
        random.Random(f"{seed}:{row['prompt_id']}").shuffle(positions)
        record = {
            **row,
            **responses,
            "model_prompt": model_prompt,
            "local_prompt_mode": local_prompt_mode,
            "model_prompt_template_version": template_version,
            "response_order": positions,
            "generation_seed": seed + index,
            "base_model_id": base_model,
            "basis_lora_path": basis_lora,
            "random_lora_path": random_lora,
        }
        output.append(record)
        write_jsonl(output, output_path)
    return output


def blind_oracle_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """response orderに従い、モデル名をpromptへ出さないOracle入力へ展開する。"""
    output = []
    response_key = {"base": "base_response", "basis": "basis_response", "random_dpo": "random_dpo_response"}
    for row in rows:
        for model in row["response_order"]:
            oracle_prompt = (
                f"数学問題（日本語）: {row['problem_ja']}\n"
                f"正解参照（日本語）: {row['ground_truth_ja']}\n"
                f"Original problem: {row['problem_en']}\n"
                f"Reference answer: {row['ground_truth_en']}"
            )
            output.append({"sample_id": row["sample_id"], "model_name": model, "prompt": oracle_prompt, "history": row["history"], "response": row[response_key[model]], "metadata": {"blind_position": row["response_order"].index(model), "prompt_version": "mathdial_eval_prompt_v1", "local_prompt_mode": row.get("local_prompt_mode", "mathdial_instruction"), "model_prompt_template_version": row.get("model_prompt_template_version", DPO_PROMPT_TEMPLATE_VERSION)}})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="MathDial日本語評価データ・応答生成")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--samples", required=True)
    prepare.add_argument("--conversations", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--count", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--exclude-prompts", action="append", default=[])
    prepare.add_argument("--stratify-teacher-moves", action="store_true")
    prepare.add_argument("--prompt-id-prefix", default="mathdial_eval")
    prepare.add_argument("--model", default=resolve_scoring_model())
    prepare.add_argument("--resume", action="store_true")
    prepare.add_argument("--mock", action="store_true")
    prepare.add_argument("--errors-output")
    prepare.add_argument("--skip-sample-errors", action="store_true")
    prepare.add_argument(
        "--candidate-reserve",
        type=int,
        default=0,
        help="翻訳失敗時の補欠として追加選定する未使用test qid数。",
    )
    prepare.add_argument(
        "--local-prompt-mode",
        choices=LOCAL_PROMPT_MODES,
        default="mathdial_instruction",
    )
    generate = sub.add_parser("generate")
    generate.add_argument("--input", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--oracle-output")
    generate.add_argument("--base-model", default="Qwen/Qwen3.5-27B")
    generate.add_argument("--basis-lora", required=True)
    generate.add_argument("--random-lora", required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--mock", action="store_true")
    generate.add_argument("--errors-output")
    generate.add_argument("--skip-sample-errors", action="store_true")
    generate.add_argument(
        "--local-prompt-mode",
        choices=LOCAL_PROMPT_MODES,
        default="mathdial_instruction",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        if args.candidate_reserve < 0:
            parser.error("--candidate-reserveは0以上で指定してください。")
        excluded_sample_ids, excluded_qids = exclusion_ids_from_prompts(
            [Path(path) for path in args.exclude_prompts]
        )
        selected = select_test_prompts(
            read_jsonl(args.samples),
            read_jsonl(args.conversations),
            count=args.count + args.candidate_reserve,
            seed=args.seed,
            excluded_sample_ids=excluded_sample_ids,
            excluded_qids=excluded_qids,
            stratify_teacher_moves=args.stratify_teacher_moves,
            prompt_id_prefix=args.prompt_id_prefix,
        )
        existing = read_jsonl(args.output) if args.resume and Path(args.output).exists() else []
        translated = translate_prompts(selected, generator=None if args.mock else OpenAIResponsesGenerator(), model=args.model, mock=args.mock, existing=existing, output_path=args.output, errors_path=args.errors_output, skip_sample_errors=args.skip_sample_errors, local_prompt_mode=args.local_prompt_mode, target_count=args.count)
        translated = translated[:args.count]
        write_jsonl(translated, args.output)
        return 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_three_model_responses(read_jsonl(args.input), base_model=args.base_model, basis_lora=args.basis_lora, random_lora=args.random_lora, output_path=output, mock=args.mock, seed=args.seed, errors_path=args.errors_output, skip_sample_errors=args.skip_sample_errors, local_prompt_mode=args.local_prompt_mode)
    if args.oracle_output:
        write_jsonl(blind_oracle_rows(rows), args.oracle_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
