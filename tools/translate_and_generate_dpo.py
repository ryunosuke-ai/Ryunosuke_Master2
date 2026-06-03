"""抽出済み対話を自然な日本語DPOデータへ変換する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from core.transition_bayes_model import (
    TransitionBayesModel,
    load_transition_bayes_model,
)
from tools.analyze_small_corpus import (
    TextGenerator,
    extract_json_object,
    resolve_analysis_model,
    write_json,
)
from tools.audit_logging import DEFAULT_AUDIT_LOG_PATH, append_audit_log
from tools.score_dialogue_with_bayes_model import (
    OpenAIResponsesGenerator,
    load_env_file,
    resolve_scoring_model,
)
from tools.score_dialogue_with_transition_bayes_model import (
    build_transition_scoring_instructions,
    is_content_filter_error,
    score_single_record,
)


DEFAULT_INPUT_PATH = "artifacts/datasets/dailydialog_selected_en.jsonl"
DEFAULT_BAYES_MODEL_PATH = "artifacts/bayes_models/generated_transition_bayes_model.json"
DEFAULT_OUTPUT_PATH = "artifacts/datasets/dailydialog_ja_dpo_preferences.jsonl"
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_CANDIDATES = 4
DEFAULT_MIN_SCORE_GAP = 0.25
DEFAULT_MIN_CHOSEN_POSTERIOR = 0.70
DEFAULT_MAX_REJECTED_POSTERIOR = 0.55
DEFAULT_SEED = 42
PROMPT_TEMPLATE_VERSION = "translate_and_generate_dpo.v2"


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    load_env_file()
    parser = argparse.ArgumentParser(description="抽出済み英語応答から日本語DPO JSONLを作成します。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help=f"入力JSONL（既定: {DEFAULT_INPUT_PATH}）。")
    parser.add_argument("--bayes-model", default=DEFAULT_BAYES_MODEL_PATH, help=f"状態遷移ベイズモデルJSON（既定: {DEFAULT_BAYES_MODEL_PATH}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"出力DPO JSONL（既定: {DEFAULT_OUTPUT_PATH}）。")
    parser.add_argument("--model", default=resolve_scoring_model(), help="翻訳・rejected生成モデル。大量処理ではgpt-5.4を推奨。")
    parser.add_argument("--audit-model", default=resolve_analysis_model(), help="必要時の品質監査モデル。既定はgpt-5.4-pro系。")
    parser.add_argument("--score-model", default=resolve_scoring_model(), help="再スコアリングモデル。")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="最大出力トークン数。")
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES, help="rejected候補の生成数。")
    parser.add_argument("--min-score-gap", type=float, default=DEFAULT_MIN_SCORE_GAP, help="採用するscore_gapの下限。")
    parser.add_argument("--min-chosen-posterior", type=float, default=DEFAULT_MIN_CHOSEN_POSTERIOR, help="翻訳後chosenのposterior下限。")
    parser.add_argument("--max-rejected-posterior", type=float, default=DEFAULT_MAX_REJECTED_POSTERIOR, help="rejectedのposterior上限。")
    parser.add_argument("--max-records", type=int, default=None, help="処理件数の上限。")
    parser.add_argument("--target-records", type=int, default=None, help="accepted DPO件数の目標。達したら処理を終了します。")
    parser.add_argument("--workers", type=int, default=1, help="サンプル単位で並列生成するworker数。1なら逐次処理。")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="再現性記録用の乱数シード。")
    parser.add_argument("--audit-log", default=DEFAULT_AUDIT_LOG_PATH, help="重要操作の要約を追記するaudit_log.mdのパス。")
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、入力件数だけ確認します。")
    return parser.parse_args()


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """JSONLを読み込む。"""
    input_path = Path(path)
    records: list[dict[str, Any]] = []
    try:
        with input_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{line_number}行目をJSONとして読めません: {exc}") from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"入力JSONLが見つかりません: {input_path}") from exc
    if not records:
        raise ValueError("入力JSONLに有効なレコードがありません。")
    return records


def read_existing_dpo_records(path: Path | str) -> list[dict[str, Any]]:
    """既存DPO出力があれば読み込む。なければ空配列を返す。"""
    output_path = Path(path)
    if not output_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with output_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"既存DPO出力の{line_number}行目をJSONとして読めません: {exc}") from exc
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path | str) -> None:
    """JSONLを書き出す。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: dict[str, Any], path: Path | str) -> None:
    """JSONLへ1レコード追記する。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def dpo_record_key(record: dict[str, Any]) -> tuple[str, int]:
    """DPO再開判定用のキーを返す。"""
    return str(record["source_dialogue_id"]), int(record["turn_index"])


def source_record_key(record: dict[str, Any]) -> tuple[str, int]:
    """抽出元レコードのキーを返す。"""
    return str(record["conversation_id"]), int(record["turn_index"])


def bayes_model_version(path: Path | str) -> str:
    """ベイズモデルJSONの内容ハッシュを返す。"""
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def build_translation_rejected_instructions(model: TransitionBayesModel) -> str:
    """翻訳とrejected候補生成の指示を作る。"""
    state_lines = "\n".join(f"- {name}: {model.state_descriptions[name]}" for name in model.states)
    observation_lines = "\n".join(f"- {name}: {model.observation_descriptions[name]}" for name in model.observations)
    return (
        "あなたはDPO学習用の日本語対話データ作成者です。"
        "目的は、英語の文脈付き高スコア応答を、日本人同士の自然な会話として使えるDPOサンプルへ変換することです。"
        "このデータはローカルLLMに小コーパス由来の会話戦略を学習させるために使われます。\n\n"
        "翻訳方針:\n"
        "- 直訳ではなく、日本人同士の自然な会話として書き換えてください。\n"
        "- ただし、元のchosenが持つ意図、感情の受け止め、話題の深め方、会話状態の進み方は保ってください。\n"
        "- 特に、相手の過去の経験、思い出の情景、当時の気持ち、一緒にいた人、季節・場所・音・匂いなどの具体性を深める応答戦略を保持してください。\n"
        "- chosenが質問の場合は、尋問調ではなく、相手が思い出を話しやすい一問にしてください。\n"
        "- promptは過去の会話文脈として自然に読めるよう、話者ラベルも含めて日本語化してください。\n"
        "- chosenは、文脈を受けた理想的な次の応答として自然で、短すぎず、説明的すぎず、会話を続けやすい表現にしてください。\n\n"
        "rejected候補の生成方針:\n"
        "- rejectedは同じtranslated_promptに対する返答として作ってください。\n"
        "- 文法的に破綻した返答、攻撃的な返答、安全性に問題がある返答は作らないでください。\n"
        "- 一見自然に読めるが、推定されたデータセット目的・会話状態・観測ラベルに照らすと低評価になりやすい返答にしてください。\n"
        "- chosenの単なる短縮、同義表現、語尾だけの変更は禁止です。\n"
        "- 候補ごとに低評価になりやすい理由が少しずつ異なるようにしてください。例: 文脈を浅く流す、一般論に戻す、助言に逸れる、相手の具体的内容を拾わない、昔の経験や情景を深めない、会話を早く閉じる。\n"
        "- rejectedも日本語としては自然で、DPO学習で比較対象にできる品質にしてください。\n\n"
        "出力はJSONのみで、translated_prompt, translated_chosen, rejected_candidates, translation_quality_score を含めてください。"
        "rejected_candidatesは文字列配列です。translation_quality_scoreは、意図・感情・会話戦略を保持できている度合いを0.0〜1.0で付けてください。\n\n"
        f"推定されたデータセット目的:\n{model.dataset_hypothesis}\n\n"
        f"会話状態:\n{state_lines}\n\n"
        f"観測ラベル:\n{observation_lines}"
    )


def build_translation_rejected_input(record: dict[str, Any], *, candidates: int, seed: int) -> str:
    """翻訳・rejected生成用の入力を作る。"""
    return (
        f"json output only.\n"
        f"seed: {seed}\n"
        f"rejected_candidates_count: {candidates}\n"
        f"source_dialogue_id: {record.get('conversation_id')}\n"
        f"turn_index: {record.get('turn_index')}\n\n"
        f"english_prompt:\n{record.get('prompt', '')}\n\n"
        f"english_chosen_response:\n{record.get('response', '')}"
    )


def validate_translation_payload(payload: dict[str, Any], *, candidates: int) -> dict[str, Any]:
    """翻訳・rejected生成のJSONを検証する。"""
    translated_prompt = str(payload.get("translated_prompt", "")).strip()
    translated_chosen = str(payload.get("translated_chosen", "")).strip()
    rejected_candidates = payload.get("rejected_candidates")
    if not translated_prompt:
        raise ValueError("`translated_prompt` が空です。")
    if not translated_chosen:
        raise ValueError("`translated_chosen` が空です。")
    if not isinstance(rejected_candidates, list):
        raise ValueError("`rejected_candidates` は配列である必要があります。")
    rejected_texts = []
    seen = set()
    for item in rejected_candidates:
        text = str(item).strip()
        if not text or text == translated_chosen or text in seen:
            continue
        rejected_texts.append(text)
        seen.add(text)
    if len(rejected_texts) < candidates:
        raise ValueError("rejected候補数が不足しています。")
    quality = payload.get("translation_quality_score", 0.0)
    if not isinstance(quality, (int, float)):
        raise ValueError("`translation_quality_score` は数値である必要があります。")
    return {
        "translated_prompt": translated_prompt,
        "translated_chosen": translated_chosen,
        "rejected_candidates": rejected_texts,
        "translation_quality_score": max(0.0, min(1.0, float(quality))),
    }


def score_japanese_response(
    *,
    record: dict[str, Any],
    response: str,
    bayes_model: TransitionBayesModel,
    generator: TextGenerator,
    score_model: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    """日本語prompt/responseを観測ラベル化し、ベイズスコアを計算する。"""
    scoring_record = {
        "conversation_id": record["conversation_id"],
        "turn_index": record["turn_index"],
        "prompt": record["prompt"],
        "response": response,
    }
    return score_single_record(
        scoring_record,
        bayes_model=bayes_model,
        generator=generator,
        model=score_model,
        max_output_tokens=max_output_tokens,
        instructions=build_transition_scoring_instructions(bayes_model),
        prior_distribution=record.get("prior_state_distribution"),
        progress_label="[STEP 5/6] japanese rescore",
    )


def generate_translation_payload(
    *,
    source_record: dict[str, Any],
    index: int,
    generator: TextGenerator,
    instructions: str,
    model: str,
    max_output_tokens: int,
    candidates: int,
    seed: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """翻訳・rejected生成を行う。LLM由来の失敗はサンプル単位のskip理由へ変換する。"""
    try:
        output_text = generator.generate(
            instructions=instructions,
            input_text=build_translation_rejected_input(source_record, candidates=candidates, seed=seed + index),
            model=model,
            max_output_tokens=max_output_tokens,
            response_text_format={"type": "json_object"},
        )
    except Exception as exc:
        if not is_content_filter_error(exc):
            raise
        print(
            "[STEP 5/6] skip content_filter generation "
            f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}",
            flush=True,
        )
        return None, "content_filter_generation"
    try:
        return validate_translation_payload(extract_json_object(output_text), candidates=candidates), None
    except ValueError as exc:
        print(
            "[STEP 5/6] skip invalid generation "
            f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}: {exc}",
            flush=True,
        )
        return None, "invalid_generation"


def choose_rejected(
    *,
    base_record: dict[str, Any],
    rejected_candidates: list[str],
    chosen_score: dict[str, Any],
    bayes_model: TransitionBayesModel,
    generator: TextGenerator,
    score_model: str,
    max_output_tokens: int,
) -> tuple[str, dict[str, Any], float]:
    """複数rejected候補を再スコアリングし、score_gap最大の候補を選ぶ。"""
    best_text = ""
    best_score: dict[str, Any] | None = None
    best_gap = -1.0
    chosen_posterior = float(chosen_score["posterior"])
    for candidate in rejected_candidates:
        if candidate == base_record["response"]:
            continue
        rejected_score = score_japanese_response(
            record=base_record,
            response=candidate,
            bayes_model=bayes_model,
            generator=generator,
            score_model=score_model,
            max_output_tokens=max_output_tokens,
        )
        gap = chosen_posterior - float(rejected_score["posterior"])
        if gap > best_gap:
            best_text = candidate
            best_score = rejected_score
            best_gap = gap
    if best_score is None:
        raise ValueError("採用可能なrejected候補がありません。")
    return best_text, best_score, best_gap


def build_one_dpo_record(
    source_record: dict[str, Any],
    *,
    index: int,
    bayes_model: TransitionBayesModel,
    model_version: str,
    generator: TextGenerator,
    instructions: str,
    model: str,
    score_model: str,
    max_output_tokens: int,
    candidates: int,
    min_score_gap: float,
    min_chosen_posterior: float,
    max_rejected_posterior: float,
    seed: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """1件の英語候補からDPOレコードを作る。条件未満ならskip理由を返す。"""
    translation_payload, skip_reason = generate_translation_payload(
        source_record=source_record,
        index=index,
        generator=generator,
        instructions=instructions,
        model=model,
        max_output_tokens=max_output_tokens,
        candidates=candidates,
        seed=seed,
    )
    if translation_payload is None:
        return None, skip_reason
    japanese_record = {
        "conversation_id": source_record["conversation_id"],
        "turn_index": source_record["turn_index"],
        "prompt": translation_payload["translated_prompt"],
        "response": translation_payload["translated_chosen"],
        "prior_state_distribution": source_record.get("prior_state_distribution"),
    }
    chosen_score = score_japanese_response(
        record=japanese_record,
        response=translation_payload["translated_chosen"],
        bayes_model=bayes_model,
        generator=generator,
        score_model=score_model,
        max_output_tokens=max_output_tokens,
    )
    rejected_text, rejected_score, score_gap = choose_rejected(
        base_record=japanese_record,
        rejected_candidates=translation_payload["rejected_candidates"],
        chosen_score=chosen_score,
        bayes_model=bayes_model,
        generator=generator,
        score_model=score_model,
        max_output_tokens=max_output_tokens,
    )
    if float(chosen_score["posterior"]) < min_chosen_posterior:
        return None, "low_chosen"
    if float(rejected_score["posterior"]) > max_rejected_posterior:
        return None, "high_rejected"
    if score_gap < min_score_gap:
        return None, "small_gap"
    return {
        "prompt": translation_payload["translated_prompt"],
        "chosen": translation_payload["translated_chosen"],
        "rejected": rejected_text,
        "score_chosen": float(chosen_score["posterior"]),
        "score_rejected": float(rejected_score["posterior"]),
        "score_gap": score_gap,
        "source_dataset": source_record.get("metadata", {}).get("source_dataset", "DailyDialog"),
        "source_dialogue_id": source_record.get("conversation_id"),
        "turn_index": source_record.get("turn_index"),
        "history_turns": source_record.get("metadata", {}).get("context_turns"),
        "translated_chosen": translation_payload["translated_chosen"],
        "translated_rejected": rejected_text,
        "state_sequence": [
            {
                "role": "chosen",
                "most_likely_state": chosen_score.get("most_likely_state"),
                "state_posteriors": chosen_score.get("state_posteriors"),
            },
            {
                "role": "rejected",
                "most_likely_state": rejected_score.get("most_likely_state"),
                "state_posteriors": rejected_score.get("state_posteriors"),
            },
        ],
        "strategy_sequence": [
            {
                "role": "chosen",
                "observation": chosen_score.get("observation"),
                "observation_score": chosen_score.get("observation_score"),
            },
            {
                "role": "rejected",
                "observation": rejected_score.get("observation"),
                "observation_score": rejected_score.get("observation_score"),
            },
        ],
        "reward_breakdown": {
            "chosen": chosen_score,
            "rejected": rejected_score,
        },
        "translation_quality_score": translation_payload["translation_quality_score"],
        "model_used_for_scoring": score_model,
        "model_used_for_translation": model,
        "model_used_for_rejected_generation": model,
        "bayesian_model_version": model_version,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "source_prompt_en": source_record.get("prompt"),
        "source_chosen_en": source_record.get("response"),
        "metadata": {
            "source_dataset": source_record.get("metadata", {}).get("source_dataset", "DailyDialog"),
            "source_split": source_record.get("metadata", {}).get("source_split"),
            "history_turns": source_record.get("metadata", {}).get("context_turns"),
            "source_posterior_en": source_record.get("posterior"),
            "generation_model": model,
            "scoring_model": score_model,
            "seed": seed,
            "bayes_model_name": bayes_model.name,
            "bayes_model_version": model_version,
            "prompt_template": PROMPT_TEMPLATE_VERSION,
            "rejected_candidates": candidates,
        },
    }, None


def build_dpo_records(
    selected_records: list[dict[str, Any]],
    *,
    bayes_model: TransitionBayesModel,
    bayes_model_path: Path | str,
    generator: TextGenerator,
    model: str,
    score_model: str,
    max_output_tokens: int,
    candidates: int,
    min_score_gap: float,
    min_chosen_posterior: float,
    max_rejected_posterior: float,
    seed: int,
    max_records: int | None,
    target_records: int | None = None,
    workers: int = 1,
    output_path: Path | str | None = None,
    existing_records: list[dict[str, Any]] | None = None,
    audit_log_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """抽出済み英語レコードから日本語DPOレコードを作る。"""
    dpo_records: list[dict[str, Any]] = list(existing_records or [])
    instructions = build_translation_rejected_instructions(bayes_model)
    model_version = bayes_model_version(bayes_model_path)
    source_records = selected_records[:max_records] if max_records is not None else selected_records
    done_keys = {dpo_record_key(record) for record in dpo_records}
    pending_records = [record for record in source_records if source_record_key(record) not in done_keys]
    if done_keys:
        print(f"[STEP 5/6] dpo generation resume: skipped={len(done_keys)} pending={len(pending_records)}", flush=True)
    if target_records is not None and len(dpo_records) >= target_records:
        print(
            f"[STEP 5/6] target already reached: accepted={len(dpo_records)} target={target_records}",
            flush=True,
        )
        return sorted(dpo_records, key=lambda record: record["score_gap"], reverse=True)[:target_records]
    skipped_low_chosen = 0
    skipped_high_rejected = 0
    skipped_small_gap = 0
    skipped_content_filter_generation = 0
    skipped_invalid_generation = 0
    indexed_records = list(enumerate(pending_records, start=1))

    def handle_result(record: dict[str, Any] | None, skip_reason: str | None) -> None:
        nonlocal skipped_low_chosen, skipped_high_rejected, skipped_small_gap
        nonlocal skipped_content_filter_generation, skipped_invalid_generation
        if skip_reason == "low_chosen":
            skipped_low_chosen += 1
            print("[STEP 5/6] skip low chosen posterior", flush=True)
            return
        if skip_reason == "high_rejected":
            skipped_high_rejected += 1
            print("[STEP 5/6] skip high rejected posterior", flush=True)
            return
        if skip_reason == "small_gap":
            skipped_small_gap += 1
            print("[STEP 5/6] skip small score_gap", flush=True)
            return
        if skip_reason == "content_filter_generation":
            skipped_content_filter_generation += 1
            print("[STEP 5/6] skip content_filter generation", flush=True)
            return
        if skip_reason == "invalid_generation":
            skipped_invalid_generation += 1
            print("[STEP 5/6] skip invalid generation", flush=True)
            return
        if record is None:
            return
        if target_records is not None and len(dpo_records) >= target_records:
            return
        dpo_records.append(record)
        if output_path is not None:
            append_jsonl(record, output_path)
        print(
            f"[STEP 5/6] accepted score_gap={float(record['score_gap']):.3f} "
            f"chosen={float(record['score_chosen']):.3f} rejected={float(record['score_rejected']):.3f}",
            flush=True,
        )

    if workers <= 1:
        for index, source_record in indexed_records:
            if target_records is not None and len(dpo_records) >= target_records:
                print(f"[STEP 5/6] target reached: accepted={len(dpo_records)} target={target_records}", flush=True)
                break
            print(
                f"[STEP 5/6] dpo generation: {index}/{len(pending_records)} "
                f"accepted={len(dpo_records)} "
                f"skipped={skipped_low_chosen + skipped_high_rejected + skipped_small_gap + skipped_content_filter_generation + skipped_invalid_generation} "
                f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}",
                flush=True,
            )
            record, skip_reason = build_one_dpo_record(
                source_record,
                index=index,
                bayes_model=bayes_model,
                model_version=model_version,
                generator=generator,
                instructions=instructions,
                model=model,
                score_model=score_model,
                max_output_tokens=max_output_tokens,
                candidates=candidates,
                min_score_gap=min_score_gap,
                min_chosen_posterior=min_chosen_posterior,
                max_rejected_posterior=max_rejected_posterior,
                seed=seed,
            )
            handle_result(record, skip_reason)
    else:
        print(f"[STEP 5/6] dpo generation parallel workers={workers} pending={len(indexed_records)}", flush=True)
        completed = 0
        for chunk_start in range(0, len(indexed_records), workers):
            if target_records is not None and len(dpo_records) >= target_records:
                print(f"[STEP 5/6] target reached: accepted={len(dpo_records)} target={target_records}", flush=True)
                break
            chunk = indexed_records[chunk_start : chunk_start + workers]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        build_one_dpo_record,
                        source_record,
                        index=index,
                        bayes_model=bayes_model,
                        model_version=model_version,
                        generator=generator,
                        instructions=instructions,
                        model=model,
                        score_model=score_model,
                        max_output_tokens=max_output_tokens,
                        candidates=candidates,
                        min_score_gap=min_score_gap,
                        min_chosen_posterior=min_chosen_posterior,
                        max_rejected_posterior=max_rejected_posterior,
                        seed=seed,
                    ): source_record
                    for index, source_record in chunk
                }
                for future in as_completed(futures):
                    source_record = futures[future]
                    record, skip_reason = future.result()
                    completed += 1
                    progress = completed / len(indexed_records) * 100.0 if indexed_records else 100.0
                    print(
                        f"[STEP 5/6] dpo generation: {completed}/{len(indexed_records)} "
                        f"({progress:.1f}%) accepted={len(dpo_records)} "
                        f"skipped={skipped_low_chosen + skipped_high_rejected + skipped_small_gap + skipped_content_filter_generation + skipped_invalid_generation} "
                        f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}",
                        flush=True,
                    )
                    handle_result(record, skip_reason)
    print(
        f"[STEP 5/6] dpo generation complete: accepted={len(dpo_records)} "
        f"skipped_low_chosen={skipped_low_chosen} "
        f"skipped_high_rejected={skipped_high_rejected} "
        f"skipped_small_gap={skipped_small_gap} "
        f"skipped_content_filter_generation={skipped_content_filter_generation} "
        f"skipped_invalid_generation={skipped_invalid_generation}",
        flush=True,
    )
    if audit_log_path is not None:
        append_audit_log(
            title="日本語DPO preferenceデータ生成",
            target_files=[
                str(bayes_model_path),
                str(output_path) if output_path is not None else "(memory only)",
            ],
            operation="抽出済み高スコア英語応答を自然な日本語chosenへ翻訳し、自然だが低スコアなrejected候補を生成・再スコアリングした。",
            reason="QwenのDPO/LoRA学習に使うprompt/chosen/rejected形式の研究データを作成するため。",
            alternatives=[
                "rejectedを単純な悪文にする案はDPO品質が落ちるため採用しなかった。",
                "翻訳後chosenを再スコアリングしない案は、翻訳で会話戦略が崩れたサンプルを検出できないため採用しなかった。",
            ],
            command=(
                "python3 -m tools.translate_and_generate_dpo "
                f"--bayes-model {bayes_model_path} "
                f"--model {model} --score-model {score_model} "
                f"--candidates {candidates} --min-score-gap {min_score_gap} "
                f"--min-chosen-posterior {min_chosen_posterior} "
                f"--max-rejected-posterior {max_rejected_posterior} "
                f"--workers {workers} --seed {seed}"
            ),
            before_after=[
                f"入力候補数: {len(source_records)}",
                f"既存採用済み件数: {len(existing_records or [])}",
                f"最終採用件数: {len(dpo_records)}",
                f"target_records: {target_records}",
                f"skip low chosen posterior: {skipped_low_chosen}",
                f"skip high rejected posterior: {skipped_high_rejected}",
                f"skip small score_gap: {skipped_small_gap}",
                f"skip content_filter generation: {skipped_content_filter_generation}",
                f"skip invalid generation: {skipped_invalid_generation}",
            ],
            risks=[
                "content_filter_generationは該当サンプルをDPOデータから除外するため、件数不足時は候補数を増やす必要がある。",
                "skip low chosen posteriorが多い場合、翻訳または抽出条件が小コーパス由来の戦略を十分保持していない可能性がある。",
                "skip small score_gapが多い場合、chosen/rejectedの対比が弱くDPO効果が出にくい可能性がある。",
            ],
            audit_log_path=audit_log_path,
            details=[
                f"generation_model: {model}",
                f"score_model: {score_model}",
                f"prompt_template_version: {PROMPT_TEMPLATE_VERSION}",
                f"bayes_model_version: {model_version}",
            ],
        )
    sorted_records = sorted(dpo_records, key=lambda record: record["score_gap"], reverse=True)
    if target_records is not None:
        return sorted_records[:target_records]
    return sorted_records


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    selected_records = read_jsonl(args.input)
    bayes_model = load_transition_bayes_model(args.bayes_model)
    if args.dry_run:
        print("日本語DPO生成 dry-run")
        print(f"  input_records: {len(selected_records)}")
        print(f"  bayes_model: {bayes_model.name}")
        print(f"  generation_model: {args.model}")
        print(f"  score_model: {args.score_model}")
        return 0
    dpo_records = build_dpo_records(
        selected_records,
        bayes_model=bayes_model,
        bayes_model_path=args.bayes_model,
        generator=OpenAIResponsesGenerator(),
        model=args.model,
        score_model=args.score_model,
        max_output_tokens=args.max_output_tokens,
        candidates=args.candidates,
        min_score_gap=args.min_score_gap,
        min_chosen_posterior=args.min_chosen_posterior,
        max_rejected_posterior=args.max_rejected_posterior,
        seed=args.seed,
        max_records=args.max_records,
        target_records=args.target_records,
        workers=max(1, args.workers),
        output_path=args.output,
        existing_records=read_existing_dpo_records(args.output),
        audit_log_path=args.audit_log,
    )
    write_jsonl(dpo_records, args.output)
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    write_json(
        {
            "input": args.input,
            "output": args.output,
            "bayes_model": args.bayes_model,
            "bayes_model_version": bayes_model_version(args.bayes_model),
            "generation_model": args.model,
            "audit_model": args.audit_model,
            "score_model": args.score_model,
            "seed": args.seed,
            "prompt_template": PROMPT_TEMPLATE_VERSION,
            "workers": max(1, args.workers),
            "target_records": args.target_records,
            "records_written": len(dpo_records),
        },
        manifest_path,
    )
    print(f"日本語DPO JSONLを書き出しました: {args.output} ({len(dpo_records)} 件)")
    print(f"再現性manifestを書き出しました: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
