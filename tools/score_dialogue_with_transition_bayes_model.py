"""状態遷移ベイズモデルを使い、大きな対話データの各応答を評価する。"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from core.transition_bayes_model import (
    TransitionBayesModel,
    TransitionObservationScore,
    load_transition_bayes_model,
    score_transition_observation,
)
from tools.score_dialogue_with_bayes_model import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    TextGenerator,
    OpenAIResponsesGenerator,
    build_scoring_input,
    extract_json_object,
    load_env_file,
    read_dialogue_records,
    resolve_scoring_model,
    write_jsonl,
)
from tools.audit_logging import DEFAULT_AUDIT_LOG_PATH, append_audit_log
from tools.jsonl_utils import ensure_jsonl_append_boundary, read_jsonl_records


DEFAULT_INPUT_PATH = "data/large_dialogue.jsonl"
DEFAULT_MODEL_PATH = "artifacts/bayes_models/generated_transition_bayes_model.json"
DEFAULT_OUTPUT_PATH = "artifacts/scored_dialogues/transition_bayes_scored_dialogue.jsonl"
SCORING_PRESETS = ("legacy", "mathdial_tutoring")
CONTENT_FILTER_FALLBACK_REASON = (
    "LLM評価APIのcontent filterにより観測ラベルを直接判定できなかったため、"
    "大量処理継続用にnegative/off_style寄りの観測へフォールバックしました。"
)
ERROR_FALLBACK_REASON = (
    "LLM評価APIまたはJSON解析の一時的な失敗により観測ラベルを安定判定できなかったため、"
    "大量処理継続用にnegative/off_style寄りの観測へフォールバックしました。"
)
CONTENT_FILTER_RETRY_PREFIX = (
    "content filterの誤検出を避けるため、固有の年齢・日付・個人名・親密表現などを"
    "中立的なプレースホルダに置換した安全化版です。"
    "評価では、置換された具体情報そのものではなく、会話文脈への応答戦略だけを判定してください。\n\n"
)
SENSITIVE_WORD_PATTERN = re.compile(
    r"\b("
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"birthday|old|older|younger|child|children|kid|kids|boy|girl"
    r")\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    load_env_file()
    parser = argparse.ArgumentParser(description="対話データを状態遷移ベイズモデルでスコアリングします。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help=f"入力JSONL/CSV（既定: {DEFAULT_INPUT_PATH}）。")
    parser.add_argument("--bayes-model", default=DEFAULT_MODEL_PATH, help=f"状態遷移ベイズモデルJSON（既定: {DEFAULT_MODEL_PATH}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"出力JSONL（既定: {DEFAULT_OUTPUT_PATH}）。")
    default_model = resolve_scoring_model()
    parser.add_argument("--model", default=default_model, help=f"評価LLMモデルまたはAzure deployment名（既定: {default_model}）。")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="最大出力トークン数。")
    parser.add_argument("--workers", type=int, default=1, help="会話単位で並列評価するworker数。1なら逐次処理。")
    parser.add_argument(
        "--max-records",
        type=int,
        help="API評価する最大応答件数。会話内の状態遷移を壊さないよう会話境界まで含めます。",
    )
    parser.add_argument(
        "--include-crossing-conversation",
        action="store_true",
        help=(
            "max-recordsを初めて超える会話も丸ごと含めます。"
            "件数下限を満たす必要があるpilot専用で、既定動作は変更しません。"
        ),
    )
    parser.add_argument(
        "--fallback-on-errors",
        action="store_true",
        help="content_filter以外のAPI/JSON失敗もnegative寄り観測へフォールバックして処理を継続します。",
    )
    parser.add_argument(
        "--repair-retryable-fallbacks",
        action="store_true",
        help=(
            "既存出力中の429・timeout等を含む会話を丸ごと削除して再評価します。"
            "会話内posteriorの整合性を保つため、失敗発話だけの差し替えは行いません。"
        ),
    )
    parser.add_argument(
        "--scoring-preset",
        choices=SCORING_PRESETS,
        default="legacy",
        help="観測分類指示。既定legacyは既存ESConv互換、mathdial_tutoringはMathDial専用です。",
    )
    parser.add_argument(
        "--invalid-observation-retries",
        type=int,
        default=0,
        help="未知ラベル・JSON不正時に許可観測だけで再判定する回数。legacyの既定は0です。",
    )
    parser.add_argument("--audit-log", default=DEFAULT_AUDIT_LOG_PATH, help="重要操作の要約を追記するaudit_log.mdのパス。")
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、入力件数だけ確認します。")
    return parser.parse_args()


def limit_records_by_conversation(
    records: list[dict[str, Any]],
    max_records: int | None,
    *,
    include_crossing_conversation: bool = False,
) -> list[dict[str, Any]]:
    """入力順を維持し、会話を途中で分断せずに評価件数を制限する。"""
    if max_records is None or max_records <= 0 or len(records) <= max_records:
        return records
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["conversation_id"]), []).append(record)
    selected: list[dict[str, Any]] = []
    for conversation_records in grouped.values():
        if selected and len(selected) + len(conversation_records) > max_records:
            if include_crossing_conversation:
                selected.extend(conversation_records)
                break
            continue
        selected.extend(conversation_records)
        if len(selected) >= max_records:
            break
    return selected


def is_retryable_fallback(record: dict[str, Any]) -> bool:
    """再API評価で回復する可能性があるfallbackか判定する。"""
    error = str(record.get("llm_error", "")).lower()
    kind = str(record.get("llm_error_kind", ""))
    if not error:
        return False
    if kind == "invalid_observation":
        return True
    retryable_markers = (
        "429",
        "ratelimit",
        "rate limit",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "service unavailable",
        "internal server error",
        "server_error",
    )
    return any(marker in error for marker in retryable_markers)


def prepare_retryable_fallback_repair(
    records: list[dict[str, Any]],
    existing_results: list[dict[str, Any]],
    output_path: Path | str,
) -> tuple[list[dict[str, Any]], set[str]]:
    """retryable fallbackを含む会話全体を出力から除き、原子的に再開準備する。"""
    allowed_conversations = {str(row["conversation_id"]) for row in records}
    repair_conversations = {
        str(row["conversation_id"])
        for row in existing_results
        if str(row.get("conversation_id")) in allowed_conversations
        and is_retryable_fallback(row)
    }
    if not repair_conversations:
        return existing_results, set()
    retained = [
        row
        for row in existing_results
        if str(row.get("conversation_id")) not in repair_conversations
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".repair.tmp")
    write_jsonl(retained, temporary)
    temporary.replace(path)
    return retained, repair_conversations


def build_transition_scoring_instructions(
    model: TransitionBayesModel,
    *,
    scoring_preset: str = "legacy",
) -> str:
    """状態遷移モデル用の観測ラベル判定指示を作る。"""
    if scoring_preset == "mathdial_tutoring":
        return build_mathdial_scoring_instructions(model)
    if scoring_preset != "legacy":
        raise ValueError(f"未知のscoring presetです: {scoring_preset}")
    observation_lines = "\n".join(f"- {name}: {model.observation_descriptions[name]}" for name in model.observations)
    state_lines = "\n".join(f"- {name}: {model.state_descriptions[name]}" for name in model.states)
    return (
        "あなたは複数ターン会話データの評価者です。"
        "目的は、promptに含まれる直前までの会話文脈に対して、responseがどの応答戦略を取っているかを安定して分類することです。"
        "この分類結果は状態遷移ベイズモデルの観測として使われ、後続のDPOデータ抽出に影響します。\n\n"
        "判定手順:\n"
        "1. prompt内の直前までの話題、相手の感情、具体的な出来事、会話の流れを読む。\n"
        "2. responseが、その文脈を受け止めているか、具体的に深めているか、自然にまとめているか、一般論や別話題へ逸れているかを判断する。\n"
        "3. responseが会話状態を望ましい方向へ進める観測か、自然に見えてもコーパス由来の目的から外れる観測かを判断する。\n"
        "4. 観測ラベルのうち最も近いものを必ず1つだけ選ぶ。同程度に見える場合は、より具体的にresponseの働きを説明しているラベルを選ぶ。\n"
        "5. どのラベルにも強く当てはまらず、文脈を浅く流す、一般論に戻す、助言や説明へ逸れる、会話を深めない場合は、negative/off_styleに近い観測ラベルを選ぶ。\n\n"
        "出力はJSONのみで、observation, score, reason を含めてください。"
        "scoreは0.0〜1.0で、選んだ観測ラベルへの確信度です。"
        "reasonには、promptのどの文脈とresponseのどの表現を根拠にしたかを日本語で簡潔に書いてください。\n\n"
        f"推定されたデータセット目的:\n{model.dataset_hypothesis}\n\n"
        f"会話状態:\n{state_lines}\n\n"
        f"観測ラベル:\n{observation_lines}"
    )


def build_mathdial_scoring_instructions(model: TransitionBayesModel) -> str:
    """state情報を見せずにMathDialの応答戦略だけを分類する指示を作る。"""
    observation_lines = "\n".join(
        f"- {name}: {model.observation_descriptions[name]}"
        for name in model.observations
    )
    allowed = ", ".join(model.observations)
    return (
        "あなたは個別指導対話のassistant応答戦略を分類する評価者です。"
        "数学知識や話題の一致ではなく、promptの学習者状態に対してresponseが果たす機能を判定してください。\n\n"
        "判定手順:\n"
        "1. promptから、学習者の試行、誤り、混乱、理解の進展を読む。\n"
        "2. responseが診断、焦点化、段階的ヒント、説明、理解確認、または目的外応答のどれに最も近いか判断する。\n"
        "3. 診断後に必要な説明や誤概念訂正を行うことは、正当な個別指導戦略になり得る。"
        "最終答えを示すこと自体ではなく、十分な診断や足場かけなしに答えを与えたかで区別する。\n"
        "4. 下記の観測ラベルから必ず1つだけ選ぶ。state名や新しいラベルは絶対に出力しない。\n\n"
        "出力はJSON objectのみとし、observation, score, reasonを含める。"
        "observationは許可ラベルと完全一致させる。scoreは0.0〜1.0の分類確信度、"
        "reasonは文脈と応答機能に基づく簡潔な日本語とする。\n\n"
        f"許可されるobservation: {allowed}\n\n"
        f"観測ラベル:\n{observation_lines}"
    )


def build_invalid_observation_retry_instructions(
    model: TransitionBayesModel,
    *,
    error: Exception,
) -> str:
    """未知ラベル・JSON不正を許可観測だけで修正する短い指示を作る。"""
    observation_lines = "\n".join(
        f"- {name}: {model.observation_descriptions[name]}"
        for name in model.observations
    )
    return (
        "直前の出力は観測分類schemaに適合しませんでした。"
        "会話を再判定し、下記のobservation名から必ず1つだけを選んでください。"
        "state名、新しいラベル、Markdownは出力禁止です。\n"
        f"不適合種別: {type(error).__name__}\n\n"
        "出力schema: {\"observation\": \"許可ラベル\", \"score\": 0.0, \"reason\": \"簡潔な根拠\"}\n\n"
        f"許可観測:\n{observation_lines}"
    )


def parse_transition_observation_score(
    payload: dict[str, Any],
    model: TransitionBayesModel,
) -> TransitionObservationScore:
    """LLM出力を状態遷移モデル用の観測評価へ変換する。"""
    observation = str(payload.get("observation", "")).strip()
    if observation not in model.observations:
        raise ValueError(f"未知の観測ラベルです: {observation}")
    raw_score = payload.get("score", 0.0)
    if not isinstance(raw_score, (int, float)):
        raise ValueError("`score` は数値である必要があります。")
    score = max(0.0, min(1.0, float(raw_score)))
    reason = str(payload.get("reason", "")).strip()
    return TransitionObservationScore(observation=observation, score=score, reason=reason)


def is_content_filter_error(exc: Exception) -> bool:
    """Azure/OpenAIのcontent filter由来の例外かを判定する。"""
    text = str(exc).lower()
    return "content_filter" in text or "content management policy" in text


def select_negative_fallback_observation(model: TransitionBayesModel) -> str:
    """LLM判定不能時に使うnegative寄りの観測ラベルを選ぶ。"""
    preferred_labels = ("generic_or_unrelated", "off_style", "generic", "blocking")
    for label in preferred_labels:
        if label in model.observations:
            return label

    negative_scores: dict[str, float] = {observation: 0.0 for observation in model.observations}
    for state in model.negative_states:
        for observation, probability in model.emission_likelihoods[state].items():
            negative_scores[observation] += probability
    return max(negative_scores, key=negative_scores.get)


def sanitize_text_for_content_filter_retry(text: str) -> str:
    """content filter誤検出時の再試行用に具体情報を中立化する。"""
    sanitized = SENSITIVE_WORD_PATTERN.sub("<neutral_detail>", text)
    sanitized = re.sub(r"\b\d+(?:st|nd|rd|th)?\b", "<number>", sanitized, flags=re.IGNORECASE)
    return sanitized


def build_safe_scoring_input(record: dict[str, Any]) -> str:
    """content filter再試行用の安全化入力を作る。"""
    safe_record = {
        **record,
        "prompt": sanitize_text_for_content_filter_retry(str(record["prompt"])),
        "response": sanitize_text_for_content_filter_retry(str(record["response"])),
    }
    return CONTENT_FILTER_RETRY_PREFIX + build_scoring_input(safe_record)


def _record_key(record: dict[str, Any]) -> tuple[str, int]:
    """再開判定用のレコードキーを返す。"""
    return str(record["conversation_id"]), int(record["turn_index"])


def read_existing_scored_records(path: Path | str) -> list[dict[str, Any]]:
    """既存のスコア済みJSONLを読み込む。"""
    records, skipped = read_jsonl_records(
        path,
        missing_ok=True,
        strict=False,
        label="既存スコア済み出力",
    )
    if skipped:
        print(f"[WARN] 既存スコア済み出力の壊れた行をskipしました: skipped={skipped}", flush=True)
    valid_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            _record_key(record)
        except (KeyError, TypeError, ValueError):
            print("[WARN] 既存スコア済み出力の再開キー欠落行をskipしました", flush=True)
            continue
        valid_records.append(record)
    return valid_records


def build_fallback_scoring_result(
    record: dict[str, Any],
    *,
    bayes_model: TransitionBayesModel,
    prior_distribution: dict[str, float] | None,
    reason: str,
    error_label: str,
    error_kind: str = "api_or_json",
    progress_label: str,
) -> dict[str, Any]:
    """LLM判定不能時のnegative寄りフォールバック結果を作る。"""
    fallback_observation = select_negative_fallback_observation(bayes_model)
    conversation_id = str(record["conversation_id"])
    print(
        f"{progress_label}: fallback {conversation_id}#{record['turn_index']} "
        f"-> {fallback_observation} ({error_label})",
        flush=True,
    )
    observation_score = TransitionObservationScore(
        observation=fallback_observation,
        score=0.0,
        reason=reason,
    )
    bayes_result = score_transition_observation(
        bayes_model,
        observation_score,
        prior_distribution=prior_distribution,
    )
    return {
        **record,
        "prior_state_distribution": prior_distribution,
        **bayes_result,
        "llm_error": error_label,
        "llm_error_kind": error_kind,
    }


def _generate_observation_score(
    record: dict[str, Any],
    *,
    bayes_model: TransitionBayesModel,
    generator: TextGenerator,
    model: str,
    max_output_tokens: int,
    instructions: str,
    input_text: str,
) -> TransitionObservationScore:
    """LLMへ1回分類を依頼して、検証済み観測へ変換する。"""
    output_text = generator.generate(
        instructions=instructions,
        input_text=input_text,
        model=model,
        max_output_tokens=max_output_tokens,
        response_text_format={"type": "json_object"},
    )
    return parse_transition_observation_score(
        extract_json_object(output_text),
        bayes_model,
    )


def _retry_invalid_observation(
    record: dict[str, Any],
    *,
    initial_error: Exception,
    attempts: int,
    bayes_model: TransitionBayesModel,
    generator: TextGenerator,
    model: str,
    max_output_tokens: int,
    progress_label: str,
) -> TransitionObservationScore:
    """許可観測だけを示し、schema不正な分類を再判定する。"""
    retry_error = initial_error
    for retry_index in range(max(0, attempts)):
        print(
            f"{progress_label}: invalid observation retry "
            f"{retry_index + 1}/{attempts} "
            f"{record['conversation_id']}#{record['turn_index']}",
            flush=True,
        )
        try:
            return _generate_observation_score(
                record,
                bayes_model=bayes_model,
                generator=generator,
                model=model,
                max_output_tokens=max_output_tokens,
                instructions=build_invalid_observation_retry_instructions(
                    bayes_model, error=retry_error
                ),
                input_text=build_scoring_input(record),
            )
        except Exception as exc:
            retry_error = exc
    raise retry_error


def score_single_record(
    record: dict[str, Any],
    *,
    bayes_model: TransitionBayesModel,
    generator: TextGenerator,
    model: str,
    max_output_tokens: int,
    instructions: str,
    prior_distribution: dict[str, float] | None,
    progress_label: str,
    fallback_on_errors: bool = False,
    scoring_preset: str = "legacy",
    invalid_observation_retries: int = 0,
) -> dict[str, Any]:
    """1レコードをLLM観測分類し、状態遷移ベイズモデルで更新する。"""
    conversation_id = str(record["conversation_id"])
    llm_error: str | None = None
    llm_retry: str | None = None
    try:
        observation_score = _generate_observation_score(
            record,
            bayes_model=bayes_model,
            generator=generator,
            model=model,
            max_output_tokens=max_output_tokens,
            instructions=instructions,
            input_text=build_scoring_input(record),
        )
    except Exception as exc:
        if not is_content_filter_error(exc):
            if scoring_preset == "mathdial_tutoring" and isinstance(exc, ValueError):
                try:
                    observation_score = _retry_invalid_observation(
                        record,
                        initial_error=exc,
                        attempts=invalid_observation_retries,
                        bayes_model=bayes_model,
                        generator=generator,
                        model=model,
                        max_output_tokens=max_output_tokens,
                        progress_label=progress_label,
                    )
                    llm_retry = "invalid_observation_retry"
                except Exception as retry_error:
                    if not fallback_on_errors:
                        raise retry_error
                    return build_fallback_scoring_result(
                        record,
                        bayes_model=bayes_model,
                        prior_distribution=prior_distribution,
                        reason=ERROR_FALLBACK_REASON,
                        error_label=f"{type(retry_error).__name__}: {retry_error}",
                        error_kind="invalid_observation",
                        progress_label=progress_label,
                    )
            else:
                if not fallback_on_errors:
                    raise
                return build_fallback_scoring_result(
                    record,
                    bayes_model=bayes_model,
                    prior_distribution=prior_distribution,
                    reason=ERROR_FALLBACK_REASON,
                    error_label=f"{type(exc).__name__}: {exc}",
                    error_kind="api_or_json",
                    progress_label=progress_label,
                )
        else:
            print(
                f"{progress_label}: content_filter retry with sanitized input "
                f"{conversation_id}#{record['turn_index']}",
                flush=True,
            )
            try:
                observation_score = _generate_observation_score(
                    record,
                    bayes_model=bayes_model,
                    generator=generator,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    instructions=instructions,
                    input_text=build_safe_scoring_input(record),
                )
                llm_retry = "content_filter_sanitized_retry"
            except Exception as retry_exc:
                if not is_content_filter_error(retry_exc):
                    if (
                        scoring_preset == "mathdial_tutoring"
                        and isinstance(retry_exc, ValueError)
                    ):
                        try:
                            observation_score = _retry_invalid_observation(
                                record,
                                initial_error=retry_exc,
                                attempts=invalid_observation_retries,
                                bayes_model=bayes_model,
                                generator=generator,
                                model=model,
                                max_output_tokens=max_output_tokens,
                                progress_label=progress_label,
                            )
                            llm_retry = "invalid_observation_retry"
                        except Exception as semantic_error:
                            if not fallback_on_errors:
                                raise
                            return build_fallback_scoring_result(
                                record,
                                bayes_model=bayes_model,
                                prior_distribution=prior_distribution,
                                reason=ERROR_FALLBACK_REASON,
                                error_label=(
                                    "content_filter_retry_invalid_observation: "
                                    f"{type(semantic_error).__name__}: {semantic_error}"
                                ),
                                error_kind="invalid_observation",
                                progress_label=progress_label,
                            )
                        # semantic retry成功時はこのexceptを抜けて更新へ進む。
                        retry_exc = None
                    if retry_exc is None:
                        pass
                    elif not fallback_on_errors:
                        raise
                    else:
                        return build_fallback_scoring_result(
                            record,
                            bayes_model=bayes_model,
                            prior_distribution=prior_distribution,
                            reason=ERROR_FALLBACK_REASON,
                            error_label=f"content_filter_retry_error: {type(retry_exc).__name__}: {retry_exc}",
                            error_kind="content_filter_retry_error",
                            progress_label=progress_label,
                        )
                else:
                    llm_error = f"content_filter: {retry_exc}"
                    return build_fallback_scoring_result(
                        record,
                        bayes_model=bayes_model,
                        prior_distribution=prior_distribution,
                        reason=CONTENT_FILTER_FALLBACK_REASON,
                        error_label=llm_error,
                        error_kind="content_filter",
                        progress_label=progress_label,
                    )
    bayes_result = score_transition_observation(
        bayes_model,
        observation_score,
        prior_distribution=prior_distribution,
    )
    result = {
        **record,
        "prior_state_distribution": prior_distribution,
        **bayes_result,
    }
    if llm_retry:
        result["llm_retry"] = llm_retry
    if llm_error:
        result["llm_error"] = llm_error
    return result


def score_conversation_records(
    records: list[dict[str, Any]],
    *,
    bayes_model: TransitionBayesModel,
    generator: TextGenerator,
    model: str,
    max_output_tokens: int,
    instructions: str,
    initial_distribution: dict[str, float] | None,
    progress_label: str,
    fallback_on_errors: bool = False,
    scoring_preset: str = "legacy",
    invalid_observation_retries: int = 0,
) -> list[dict[str, Any]]:
    """1会話内のレコードを順序通りにスコアリングする。"""
    results: list[dict[str, Any]] = []
    prior_distribution = initial_distribution
    for record in sorted(records, key=lambda item: int(item["turn_index"])):
        result = score_single_record(
            record,
            bayes_model=bayes_model,
            generator=generator,
            model=model,
            max_output_tokens=max_output_tokens,
            instructions=instructions,
            prior_distribution=prior_distribution,
            progress_label=progress_label,
            fallback_on_errors=fallback_on_errors,
            scoring_preset=scoring_preset,
            invalid_observation_retries=invalid_observation_retries,
        )
        prior_distribution = dict(result["state_posteriors"])
        results.append(result)
    return results


def score_records(
    records: list[dict[str, Any]],
    *,
    bayes_model: TransitionBayesModel,
    generator: TextGenerator,
    model: str,
    max_output_tokens: int,
    progress_label: str = "scoring",
    existing_results: list[dict[str, Any]] | None = None,
    output_path: Path | str | None = None,
    workers: int = 1,
    fallback_on_errors: bool = False,
    scoring_preset: str = "legacy",
    invalid_observation_retries: int = 0,
) -> list[dict[str, Any]]:
    """対話レコード群を状態遷移ベイズモデルでスコアリングする。"""
    results: list[dict[str, Any]] = list(existing_results or [])
    distribution_by_conversation: dict[str, dict[str, float]] = {}
    done_keys = {_record_key(record) for record in results}
    for record in sorted(results, key=lambda item: (str(item["conversation_id"]), int(item["turn_index"]))):
        state_posteriors = record.get("state_posteriors")
        if isinstance(state_posteriors, dict):
            distribution_by_conversation[str(record["conversation_id"])] = {
                str(state): float(value)
                for state, value in state_posteriors.items()
                if isinstance(value, (int, float))
            }
    instructions = build_transition_scoring_instructions(
        bayes_model, scoring_preset=scoring_preset
    )
    sorted_records = sorted(records, key=lambda item: (str(item["conversation_id"]), int(item["turn_index"])))
    pending_records = [record for record in sorted_records if _record_key(record) not in done_keys]
    if done_keys:
        print(f"{progress_label}: resume skipped={len(done_keys)} pending={len(pending_records)}", flush=True)
    total_records = len(pending_records)
    output_file = None
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ensure_jsonl_append_boundary(path)
        output_file = path.open("a", encoding="utf-8")
    try:
        if workers <= 1:
            for index, record in enumerate(pending_records, start=1):
                conversation_id = str(record["conversation_id"])
                progress = (index / total_records * 100.0) if total_records else 100.0
                print(
                    f"{progress_label}: {index}/{total_records} "
                    f"({progress:.1f}%) {conversation_id}#{record['turn_index']}",
                    flush=True,
                )
                result = score_single_record(
                    record,
                    bayes_model=bayes_model,
                    generator=generator,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    instructions=instructions,
                    prior_distribution=distribution_by_conversation.get(conversation_id),
                    progress_label=progress_label,
                    fallback_on_errors=fallback_on_errors,
                    scoring_preset=scoring_preset,
                    invalid_observation_retries=invalid_observation_retries,
                )
                distribution_by_conversation[conversation_id] = dict(result["state_posteriors"])
                results.append(result)
                if output_file is not None:
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
        else:
            records_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in pending_records:
                records_by_conversation[str(record["conversation_id"])].append(record)
            print(
                f"{progress_label}: parallel workers={workers} "
                f"conversations={len(records_by_conversation)}",
                flush=True,
            )
            completed_records = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_conversation = {
                    executor.submit(
                        score_conversation_records,
                        conversation_records,
                        bayes_model=bayes_model,
                        generator=generator,
                        model=model,
                        max_output_tokens=max_output_tokens,
                        instructions=instructions,
                        initial_distribution=distribution_by_conversation.get(conversation_id),
                        progress_label=progress_label,
                        fallback_on_errors=fallback_on_errors,
                        scoring_preset=scoring_preset,
                        invalid_observation_retries=invalid_observation_retries,
                    ): conversation_id
                    for conversation_id, conversation_records in records_by_conversation.items()
                }
                for future in as_completed(future_to_conversation):
                    conversation_id = future_to_conversation[future]
                    conversation_results = future.result()
                    if conversation_results:
                        distribution_by_conversation[conversation_id] = dict(conversation_results[-1]["state_posteriors"])
                    results.extend(conversation_results)
                    completed_records += len(conversation_results)
                    progress = (completed_records / total_records * 100.0) if total_records else 100.0
                    print(
                        f"{progress_label}: {completed_records}/{total_records} "
                        f"({progress:.1f}%) completed conversation {conversation_id}",
                        flush=True,
                    )
                    if output_file is not None:
                        for result in conversation_results:
                            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output_file.flush()
    finally:
        if output_file is not None:
            output_file.close()
    return results


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    source_records = read_dialogue_records(args.input)
    records = limit_records_by_conversation(
        source_records,
        args.max_records,
        include_crossing_conversation=args.include_crossing_conversation,
    )
    if len(records) < len(source_records):
        print(
            "[scoring] 十分な比較候補プールを確保したため入力を早期停止します: "
            f"selected={len(records)} source={len(source_records)} max_records={args.max_records}",
            flush=True,
        )
    bayes_model = load_transition_bayes_model(args.bayes_model)
    if args.dry_run:
        print("transition bayes scoring dry-run")
        print(f"  records: {len(records)}")
        print(f"  bayes_model: {bayes_model.name}")
        print(f"  model: {args.model or DEFAULT_MODEL}")
        return 0
    existing_results = read_existing_scored_records(args.output)
    if args.repair_retryable_fallbacks:
        existing_results, repair_conversations = prepare_retryable_fallback_repair(
            records,
            existing_results,
            args.output,
        )
        print(
            "[scoring repair] retryable fallback conversations="
            f"{len(repair_conversations)} retained_records={len(existing_results)}",
            flush=True,
        )
    scored = score_records(
        records,
        bayes_model=bayes_model,
        generator=OpenAIResponsesGenerator(),
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        progress_label=f"[STEP] scoring {args.model}",
        existing_results=existing_results,
        output_path=Path(args.output),
        workers=max(1, args.workers),
        fallback_on_errors=args.fallback_on_errors,
        scoring_preset=args.scoring_preset,
        invalid_observation_retries=max(0, args.invalid_observation_retries),
    )
    retry_count = sum(1 for record in scored if record.get("llm_retry") == "content_filter_sanitized_retry")
    fallback_count = sum(1 for record in scored if str(record.get("llm_error", "")).startswith("content_filter:"))
    append_audit_log(
        title="状態遷移ベイズモデルによる大規模対話スコアリング",
        target_files=[args.input, args.bayes_model, args.output],
        operation="DailyDialog等の大規模対話候補をLLMで観測ラベル化し、状態遷移ベイズモデルでposteriorを計算した。",
        reason="小コーパス由来の会話スタイルに近い応答を抽出するため。",
        alternatives=[
            "全件を手動評価する案はスケールしないため採用しなかった。",
            "content_filter対象を即除外する案はデータ損失が大きいため、安全化再試行を優先した。",
        ],
        command=(
            "python3 -m tools.score_dialogue_with_transition_bayes_model "
            f"--input {args.input} --bayes-model {args.bayes_model} --output {args.output} "
            f"--model {args.model} --workers {max(1, args.workers)} "
            f"--max-records {args.max_records or 'all'}"
        ),
        before_after=[
            f"元入力レコード数: {len(source_records)}",
            f"早期停止後の評価対象レコード数: {len(records)}",
            f"出力スコア済みレコード数: {len(scored)}",
            f"content_filter安全化再試行成功件数: {retry_count}",
            f"content_filterフォールバック件数: {fallback_count}",
        ],
        risks=[
            "content_filterフォールバック件数が多い場合、該当サンプルの観測ラベル品質を後で確認する必要がある。",
            "LLM観測分類は確率的なため、モデル名・ベイズモデル・出力JSONLを追跡する必要がある。",
        ],
        audit_log_path=args.audit_log,
    )
    print(f"状態遷移スコア済みJSONLを書き出しました: {args.output} ({len(scored)} 件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
