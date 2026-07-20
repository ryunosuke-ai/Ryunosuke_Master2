"""MathDial held-out評価promptの日本語化と3モデル応答生成。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import yaml

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
DISCRIMINATIVE_SAMPLING_VERSION = "mathdial_discriminative_followup.v1"
LOCAL_PROMPT_MODES = (
    "mathdial_instruction",
    "context_only",
    "neutral_conversation",
)
SAMPLING_PRESETS = ("standard", "discriminative_followup")
DISCRIMINATIVE_MOVES = ("probing", "telling", "focus")
DISCRIMINATIVE_STAGES = ("initial", "guided", "advanced")


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


def discriminative_stage(history: list[dict[str, Any]]) -> str:
    """履歴長から、事前固定したMathDial評価段階を返す。"""
    count = len(history)
    if count == 2:
        return "initial"
    if 4 <= count <= 8:
        return "guided"
    if count >= 10:
        return "advanced"
    raise ValueError(f"識別力評価の段階へ割り当てられない履歴長です: {count}")


def has_substantive_last_user_turn(history: list[dict[str, Any]]) -> bool:
    """短い相づちを除き、推論または数式を含むuser発話か判定する。"""
    if not history or history[-1].get("role") != "user":
        return False
    text = str(history[-1].get("text", "")).strip()
    return len(text) >= 20 or (
        len(text) >= 8 and re.search(r"\d|[=+*/-]", text) is not None
    )


def load_discriminative_quota_config(
    path: Path | str,
) -> tuple[dict[tuple[str, str], int], int, dict[str, Any]]:
    """識別力評価の9層quotaを読み、固定schemaを検証する。"""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("識別力評価quota configはmappingである必要があります。")
    if config.get("version") != DISCRIMINATIVE_SAMPLING_VERSION:
        raise ValueError(
            "識別力評価quota configのversionが一致しません: "
            f"{config.get('version')}"
        )
    raw_quotas = config.get("quotas")
    if not isinstance(raw_quotas, dict):
        raise ValueError("識別力評価quota configにquotasがありません。")
    quotas: dict[tuple[str, str], int] = {}
    for move in DISCRIMINATIVE_MOVES:
        move_quotas = raw_quotas.get(move)
        if not isinstance(move_quotas, dict):
            raise ValueError(f"Teacher move quotaがありません: {move}")
        for stage in DISCRIMINATIVE_STAGES:
            value = move_quotas.get(stage)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"quotaは正数で指定してください: {move}/{stage}")
            quotas[(move, stage)] = value
    target_count = sum(quotas.values())
    if int(config.get("target_count", -1)) != target_count:
        raise ValueError(
            "target_countと9層quota合計が一致しません: "
            f"{config.get('target_count')} != {target_count}"
        )
    reserve = config.get("reserve_per_stratum")
    if not isinstance(reserve, int) or reserve < 0:
        raise ValueError("reserve_per_stratumは0以上の整数で指定してください。")
    return quotas, reserve, config


def _stable_key(seed: int, *values: Any) -> str:
    text = ":".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _assign_qids_to_strata(
    options: dict[str, dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]]],
    *,
    quotas: dict[tuple[str, str], int],
    seed: int,
) -> dict[str, tuple[str, str]]:
    """qidを一度だけ使い、最大流で全stratum quotaを満たす。"""
    source = ("source",)
    sink = ("sink",)
    adjacency: dict[tuple[Any, ...], list[tuple[Any, ...]]] = defaultdict(list)
    capacity: dict[tuple[tuple[Any, ...], tuple[Any, ...]], int] = {}

    def add_edge(left: tuple[Any, ...], right: tuple[Any, ...], value: int) -> None:
        adjacency[left].append(right)
        adjacency[right].append(left)
        capacity[(left, right)] = value
        capacity[(right, left)] = 0

    qids = sorted(options, key=lambda qid: _stable_key(seed, "qid", qid))
    for qid in qids:
        qid_node = ("qid", qid)
        add_edge(source, qid_node, 1)
        cells = sorted(
            options[qid],
            key=lambda cell: _stable_key(seed, "cell", qid, *cell),
        )
        for move, stage in cells:
            add_edge(qid_node, ("cell", move, stage), 1)
    for (move, stage), value in sorted(quotas.items()):
        add_edge(("cell", move, stage), sink, value)

    flow = 0
    while True:
        previous: dict[tuple[Any, ...], tuple[Any, ...] | None] = {source: None}
        queue = deque([source])
        while queue and sink not in previous:
            node = queue.popleft()
            for next_node in adjacency[node]:
                if next_node in previous or capacity[(node, next_node)] <= 0:
                    continue
                previous[next_node] = node
                queue.append(next_node)
        if sink not in previous:
            break
        node = sink
        while previous[node] is not None:
            parent = previous[node]
            assert parent is not None
            capacity[(parent, node)] -= 1
            capacity[(node, parent)] += 1
            node = parent
        flow += 1

    required = sum(quotas.values())
    if flow != required:
        availability = {
            f"{move}/{stage}": sum(
                (move, stage) in cells for cells in options.values()
            )
            for move, stage in quotas
        }
        raise ValueError(
            "qid一意条件の下で識別力評価quotaを満たせません: "
            f"{flow}/{required}, availability={availability}"
        )

    assigned: dict[str, tuple[str, str]] = {}
    for qid in qids:
        qid_node = ("qid", qid)
        for move, stage in options[qid]:
            cell_node = ("cell", move, stage)
            if capacity.get((cell_node, qid_node), 0) > 0:
                assigned[qid] = (move, stage)
                break
    if len(assigned) != required:
        raise RuntimeError("最大流の識別力評価割当を復元できません。")
    return assigned


def select_discriminative_followup_prompts(
    samples: list[dict[str, Any]],
    conversations: list[dict[str, Any]],
    *,
    quotas: dict[tuple[str, str], int],
    reserve_per_stratum: int,
    seed: int,
    excluded_sample_ids: set[str] | None = None,
    excluded_qids: set[str] | None = None,
    prompt_id_prefix: str = "mathdial_discriminative",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """未使用test qidから、事前固定した9層と補欠を選ぶ。"""
    conversation_by_id = {
        row["conversation_id"]: row
        for row in conversations
        if row.get("split") == "test"
    }
    excluded_sample_ids = excluded_sample_ids or set()
    excluded_qids = excluded_qids or set()
    options: dict[
        str,
        dict[
            tuple[str, str],
            list[tuple[dict[str, Any], dict[str, Any]]],
        ],
    ] = defaultdict(lambda: defaultdict(list))
    eligible_sample_count = 0
    for sample in samples:
        metadata = sample.get("metadata", {})
        if metadata.get("split") != "test" or not metadata.get("history_ends_with_user"):
            continue
        if str(sample.get("sample_id", "")) in excluded_sample_ids:
            continue
        if sample.get("next_user_turn") is None:
            continue
        history = sample.get("history")
        if not isinstance(history, list) or not has_substantive_last_user_turn(history):
            continue
        try:
            stage = discriminative_stage(history)
        except ValueError:
            continue
        conversation = conversation_by_id.get(str(sample.get("conversation_id", "")))
        if conversation is None:
            continue
        qid = str(conversation.get("metadata", {}).get("qid", "")).strip()
        if not qid or qid in excluded_qids:
            continue
        moves = {
            str(move).strip().lower()
            for move in metadata.get("teacher_moves", [])
        }
        valid_moves = moves & set(DISCRIMINATIVE_MOVES)
        if not valid_moves:
            continue
        eligible_sample_count += 1
        for move in valid_moves:
            options[qid][(move, stage)].append((sample, conversation))

    combined_quotas = {
        cell: value + reserve_per_stratum for cell, value in quotas.items()
    }
    assigned = _assign_qids_to_strata(
        options,
        quotas=combined_quotas,
        seed=seed,
    )
    selected: list[dict[str, Any]] = []
    stratum_counts: dict[str, dict[str, int]] = {}
    for cell in quotas:
        move, stage = cell
        qids = sorted(
            [qid for qid, assigned_cell in assigned.items() if assigned_cell == cell],
            key=lambda qid: _stable_key(seed, "assigned", move, stage, qid),
        )
        target = quotas[cell]
        stratum_counts[f"{move}/{stage}"] = {
            "primary": target,
            "reserve": reserve_per_stratum,
            "total": len(qids),
        }
        for cell_index, qid in enumerate(qids):
            candidates = sorted(
                options[qid][cell],
                key=lambda pair: _stable_key(
                    seed,
                    "sample",
                    qid,
                    pair[0].get("sample_id"),
                ),
            )
            sample, conversation = candidates[0]
            selected.append(
                {
                    "sample_id": sample["sample_id"],
                    "conversation_id": sample["conversation_id"],
                    "qid": qid,
                    "problem_en": conversation.get("metadata", {}).get(
                        "question", ""
                    ),
                    "ground_truth_en": conversation.get("metadata", {}).get(
                        "ground_truth", ""
                    ),
                    "history_en": sample["history"],
                    "source_teacher_moves": sample["metadata"].get(
                        "teacher_moves", []
                    ),
                    "split": "test",
                    "selection_teacher_move": move,
                    "selection_stage": stage,
                    "selection_role": (
                        "primary" if cell_index < target else "reserve"
                    ),
                    "selection_rank_in_stratum": cell_index,
                }
            )
    selected.sort(
        key=lambda row: (
            row["selection_role"] != "primary",
            _stable_key(seed, "output", row["sample_id"]),
        )
    )
    for index, row in enumerate(selected):
        row["prompt_id"] = f"{prompt_id_prefix}_{index:03d}"

    excluded_qid_hash = hashlib.sha256(
        "\n".join(sorted(excluded_qids)).encode("utf-8")
    ).hexdigest()
    manifest = {
        "version": DISCRIMINATIVE_SAMPLING_VERSION,
        "status": "prospective_targeted_followup_after_subgroup_analysis",
        "seed": seed,
        "target_count": sum(quotas.values()),
        "reserve_per_stratum": reserve_per_stratum,
        "candidate_count": len(selected),
        "eligible_sample_count": eligible_sample_count,
        "eligible_qid_count": len(options),
        "excluded_sample_count": len(excluded_sample_ids),
        "excluded_qid_count": len(excluded_qids),
        "excluded_qids_sha256": excluded_qid_hash,
        "filters": {
            "split": "test",
            "history_ends_with_user": True,
            "next_user_turn_observed": True,
            "teacher_moves": list(DISCRIMINATIVE_MOVES),
            "generic_only_excluded": True,
            "substantive_last_user_turn": (
                "20+ characters, or 8+ characters with number/formula"
            ),
            "qid_unique": True,
            "conversation_unique": True,
            "model_outputs_used_for_selection": False,
        },
        "strata": stratum_counts,
    }
    return selected, manifest


def finalize_discriminative_translations(
    translated: list[dict[str, Any]],
    *,
    quotas: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    """翻訳成功済み候補から、各層のprimary不足を補欠で補う。"""
    selected: list[dict[str, Any]] = []
    seen_qids: set[str] = set()
    for move, stage in quotas:
        candidates = [
            row
            for row in translated
            if row.get("selection_teacher_move") == move
            and row.get("selection_stage") == stage
        ]
        candidates.sort(
            key=lambda row: (
                row.get("selection_role") != "primary",
                int(row.get("selection_rank_in_stratum", 10**9)),
                str(row.get("sample_id", "")),
            )
        )
        accepted = []
        for row in candidates:
            qid = str(row.get("qid", ""))
            if not qid or qid in seen_qids:
                continue
            accepted.append(row)
            seen_qids.add(qid)
            if len(accepted) == quotas[(move, stage)]:
                break
        if len(accepted) != quotas[(move, stage)]:
            raise ValueError(
                "翻訳成功候補が識別力評価quotaに不足しています: "
                f"{move}/{stage}={len(accepted)}/{quotas[(move, stage)]}"
            )
        selected.extend(accepted)
    selected.sort(key=lambda row: str(row.get("prompt_id", "")))
    return selected


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
            *history_turns[-DEFAULT_MAX_HISTORY_TURNS:],
        ]
        return build_mathdial_dpo_prompt(
            history_turns=model_history,
            max_history_turns=len(model_history),
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
        "--sampling-preset",
        choices=SAMPLING_PRESETS,
        default="standard",
    )
    prepare.add_argument("--sampling-quota-config")
    prepare.add_argument("--selection-manifest")
    prepare.add_argument(
        "--candidate-output",
        help="識別力評価でprimaryと補欠の翻訳結果を保存するJSONL。",
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
        if args.sampling_preset == "discriminative_followup":
            if not args.sampling_quota_config:
                parser.error(
                    "discriminative_followupには"
                    "--sampling-quota-configが必要です。"
                )
            if not args.selection_manifest:
                parser.error(
                    "discriminative_followupには"
                    "--selection-manifestが必要です。"
                )
            quotas, reserve_per_stratum, quota_config = (
                load_discriminative_quota_config(args.sampling_quota_config)
            )
            if sum(quotas.values()) != args.count:
                parser.error(
                    "--countと識別力評価quota合計が一致しません: "
                    f"{args.count} != {sum(quotas.values())}"
                )
            candidates, manifest = select_discriminative_followup_prompts(
                read_jsonl(args.samples),
                read_jsonl(args.conversations),
                quotas=quotas,
                reserve_per_stratum=reserve_per_stratum,
                seed=args.seed,
                excluded_sample_ids=excluded_sample_ids,
                excluded_qids=excluded_qids,
                prompt_id_prefix=args.prompt_id_prefix,
            )
            candidate_output = Path(
                args.candidate_output
                or str(
                    Path(args.output).with_name(
                        "prompt_candidates_ja.jsonl"
                    )
                )
            )
            existing_candidates = (
                read_jsonl(candidate_output)
                if args.resume and candidate_output.exists()
                else []
            )
            translated_candidates = translate_prompts(
                candidates,
                generator=(
                    None if args.mock else OpenAIResponsesGenerator()
                ),
                model=args.model,
                mock=args.mock,
                existing=existing_candidates,
                output_path=candidate_output,
                errors_path=args.errors_output,
                skip_sample_errors=args.skip_sample_errors,
                local_prompt_mode=args.local_prompt_mode,
                target_count=None,
            )
            translated = finalize_discriminative_translations(
                translated_candidates,
                quotas=quotas,
            )
            write_jsonl(translated, args.output)
            selected_counts: dict[str, int] = defaultdict(int)
            reserve_used: dict[str, int] = defaultdict(int)
            for row in translated:
                key = (
                    f"{row['selection_teacher_move']}/"
                    f"{row['selection_stage']}"
                )
                selected_counts[key] += 1
                if row.get("selection_role") == "reserve":
                    reserve_used[key] += 1
            template_version = DPO_PROMPT_TEMPLATE_VERSION
            if args.local_prompt_mode == "context_only":
                template_version = CONTEXT_ONLY_DPO_PROMPT_TEMPLATE_VERSION
            elif args.local_prompt_mode == "neutral_conversation":
                template_version = (
                    NEUTRAL_CONVERSATION_DPO_PROMPT_TEMPLATE_VERSION
                )
            manifest.update(
                {
                    "quota_config": quota_config,
                    "quota_config_path": str(args.sampling_quota_config),
                    "translated_candidate_count": len(
                        translated_candidates
                    ),
                    "final_count": len(translated),
                    "selected_strata": dict(
                        sorted(selected_counts.items())
                    ),
                    "reserve_used": dict(sorted(reserve_used.items())),
                    "candidate_output": str(candidate_output),
                    "output": str(args.output),
                    "local_prompt_mode": args.local_prompt_mode,
                    "model_prompt_template_version": template_version,
                }
            )
            selection_manifest = Path(args.selection_manifest)
            selection_manifest.parent.mkdir(parents=True, exist_ok=True)
            selection_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return 0
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
