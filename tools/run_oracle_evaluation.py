"""Oracle正解応答を100点満点としてDPO前後モデルを評価する。"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.dpo_compare_text_chat import (  # noqa: E402
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_LORA_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    build_dpo_compare_prompt,
    generate_reply,
    load_compare_bundle,
)
from core.transition_bayes_model import TransitionBayesModel, load_transition_bayes_model  # noqa: E402
from tools.analyze_small_corpus import (  # noqa: E402
    OpenAIResponsesGenerator,
    build_corpus_text,
    read_jsonl as read_small_corpus_jsonl,
    resolve_analysis_model,
)
from tools.score_dialogue_with_bayes_model import extract_json_object, load_env_file  # noqa: E402


DEFAULT_PROMPTS_PATH = "configs/evaluation_prompts/reminiscence_oracle_eval_v2_100.jsonl"
DEFAULT_SMALL_CORPUS_PATH = "data/small_corpus.jsonl"
DEFAULT_BAYES_MODEL_PATH = "artifacts/bayes_models/generated_transition_bayes_model.json"
DEFAULT_OUTPUT_DIR = "artifacts/evaluations/oracle_eval_runs/reminiscence_oracle_eval_v2"
DEFAULT_ORACLE_MAX_OUTPUT_TOKENS = 4096
PROMPT_TEMPLATE_VERSION = "oracle_eval.v2"
REFERENCE_TEMPLATE_VERSION = "oracle_reference_generation.v2"
JUDGE_TEMPLATE_VERSION = "oracle_score_against_reference.v2"
REFERENCE_TEMPLATE_VERSION_V3 = "oracle_reference_generation.esconv_strategy.v3"
JUDGE_TEMPLATE_VERSION_V3 = "oracle_score_against_reference.esconv_strategy.v3"
DEFAULT_STYLE_PRESET = "reminiscence"
ESCONV_STRATEGY_V3_PRESET = "esconv_strategy_v3"
ESCONV_STRATEGY_V3_AXIS_KEYS = (
    "esconv_strategy_adherence",
    "emotional_reflection_validation",
    "premature_advice_avoidance",
    "supportive_tone",
    "contextual_grounding",
    "conversational_progression",
    "overall_helpfulness",
)
ESCONV_CORE_WEIGHTS = {
    "esconv_strategy_adherence": 0.40,
    "emotional_reflection_validation": 0.35,
    "premature_advice_avoidance": 0.25,
}
WEIGHTED_ESCONV_OVERALL_WEIGHTS = {
    "esconv_strategy_adherence": 0.25,
    "emotional_reflection_validation": 0.25,
    "premature_advice_avoidance": 0.20,
    "supportive_tone": 0.10,
    "contextual_grounding": 0.10,
    "conversational_progression": 0.05,
    "overall_helpfulness": 0.05,
}
WIN_TIE_THRESHOLD = 1.0


@dataclass(frozen=True)
class EvaluationPrompt:
    """Oracle評価用の1入力。"""

    prompt_id: str
    category: str
    prompt: str
    history: tuple[dict[str, str], ...] = ()
    axis_focus: tuple[str, ...] = ()


@dataclass(frozen=True)
class OracleRetryConfig:
    """Oracle API呼び出しのリトライ設定。"""

    max_retries: int = 5
    base_seconds: float = 5.0
    max_seconds: float = 60.0


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    load_env_file()
    default_oracle_model = resolve_analysis_model()
    parser = argparse.ArgumentParser(description="Oracle正解応答を100点満点としてbase/DPO応答を評価します。")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH, help=f"評価prompt JSONL（既定: {DEFAULT_PROMPTS_PATH}）。")
    parser.add_argument("--small-corpus", default=DEFAULT_SMALL_CORPUS_PATH, help=f"Oracleが参照する小コーパスJSONL（既定: {DEFAULT_SMALL_CORPUS_PATH}）。")
    parser.add_argument("--bayes-model", default=DEFAULT_BAYES_MODEL_PATH, help=f"状態遷移ベイズモデルJSON（既定: {DEFAULT_BAYES_MODEL_PATH}）。")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"出力ディレクトリ（既定: {DEFAULT_OUTPUT_DIR}）。")
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID, help=f"ベースモデルID（既定: {DEFAULT_BASE_MODEL_ID}）。")
    parser.add_argument("--lora-path", default=DEFAULT_LORA_PATH, help=f"LoRA adapterパス（既定: {DEFAULT_LORA_PATH}）。")
    parser.add_argument("--oracle-model", default=default_oracle_model, help=f"Oracle評価モデル（既定: {default_oracle_model}）。")
    parser.add_argument("--oracle-workers", type=int, default=1, help="Oracle参照生成・採点を並列実行するworker数。")
    parser.add_argument(
        "--style-preset",
        choices=("reminiscence", "esconv_support", ESCONV_STRATEGY_V3_PRESET),
        default=DEFAULT_STYLE_PRESET,
        help="Oracle正解応答・採点基準のスタイル。",
    )
    parser.add_argument("--max-prompts", type=int, default=None, help="評価prompt件数の上限。")
    parser.add_argument("--skip-prompts", type=int, default=0, help="評価prompt先頭からスキップする件数。")
    parser.add_argument(
        "--categories",
        default="",
        help="評価対象カテゴリをカンマ区切りで指定します。空なら全カテゴリを使います。",
    )
    parser.add_argument(
        "--local-prompt-mode",
        choices=("instruction", "context_only"),
        default="instruction",
        help="base/DPOへ渡すprompt形式。通常は補助指示つきのinstruction、厳密比較ではcontext_onlyを使います。",
    )
    parser.add_argument("--seed", type=int, default=42, help="乱数シード。")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Qwen生成の最大トークン数。")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Qwen生成temperature。")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Qwen生成top_p。")
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY, help="Qwen生成repetition penalty。")
    parser.add_argument("--oracle-max-output-tokens", type=int, default=DEFAULT_ORACLE_MAX_OUTPUT_TOKENS, help="Oracle出力の最大トークン数。")
    parser.add_argument("--small-corpus-max-chars", type=int, default=20000, help="Oracleに渡す小コーパス抜粋の最大文字数。")
    parser.add_argument("--use-4bit", action="store_true", help="ローカルQwenを4bitで読み込みます。通常は指定しません。")
    parser.add_argument("--no-4bit", action="store_true", help="互換性用オプション。既定で4bitは使わないため動作は変わりません。")
    parser.add_argument("--dry-run", action="store_true", help="モデル/APIを呼ばず、入力と設定だけ確認します。")
    return parser.parse_args()


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """JSONLを読み込む。"""
    input_path = Path(path)
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_number} をJSONとして読めません: {exc}") from exc
    if not records:
        raise ValueError(f"JSONLに有効なレコードがありません: {input_path}")
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path | str) -> None:
    """JSONLを書き出す。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(payload: dict[str, Any], path: Path | str) -> None:
    """JSONを書き出す。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl_record(record: dict[str, Any], path: Path | str) -> None:
    """JSONLへ1レコード追記し、途中停止に備えて同期する。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def read_jsonl_lenient(path: Path | str) -> list[dict[str, Any]]:
    """壊れた行を警告して無視しながらJSONLを読み込む。"""
    input_path = Path(path)
    if not input_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[Oracle Eval] warning: {input_path}:{line_number} をJSONとして読めないためスキップします: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if not isinstance(payload, dict):
                print(
                    f"[Oracle Eval] warning: {input_path}:{line_number} がJSON objectではないためスキップします。",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            records.append(payload)
    return records


def sample_key(record: dict[str, Any]) -> str:
    """再開判定に使うsample_id/prompt_idを返す。"""
    key = str(record.get("sample_id") or record.get("prompt_id") or "").strip()
    if not key:
        raise ValueError("recordに `sample_id` または `prompt_id` がありません。")
    return key


def records_by_sample_key(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """同じsampleが複数ある場合は後の行を優先する。"""
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        try:
            indexed[sample_key(record)] = record
        except ValueError:
            print("[Oracle Eval] warning: sample_id/prompt_idのない既存レコードをスキップします。", flush=True)
    return indexed


def retry_config_from_env() -> OracleRetryConfig:
    """環境変数からOracleリトライ設定を読む。"""
    return OracleRetryConfig(
        max_retries=max(0, int(os.environ.get("ORACLE_MAX_RETRIES", "5"))),
        base_seconds=max(0.0, float(os.environ.get("ORACLE_RETRY_BASE_SECONDS", "5"))),
        max_seconds=max(0.0, float(os.environ.get("ORACLE_RETRY_MAX_SECONDS", "60"))),
    )


def progress_detail_enabled() -> bool:
    """Oracle進捗詳細表示を行うかを返す。"""
    return os.environ.get("ORACLE_PROGRESS_DETAIL", "1") != "0"


def reason_max_chars_from_env() -> int:
    """進捗表示reasonの最大文字数を返す。"""
    return max(20, int(os.environ.get("ORACLE_REASON_MAX_CHARS", "140")))


def truncate_text(value: Any, *, max_chars: int) -> str:
    """進捗表示用に長文を短縮する。"""
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_signed(value: float) -> str:
    """符号付き小数表示を返す。"""
    return f"{value:+.1f}"


def format_judgment_progress(
    judgment: dict[str, Any],
    *,
    completed: int,
    total: int,
    reason_max_chars: int,
) -> str:
    """1サンプル分のOracle判定完了表示を作る。"""
    prompt_id = str(judgment.get("sample_id") or judgment.get("prompt_id") or "")
    winner = str(judgment.get("winner", "unknown"))
    parts = [
        f"[Oracle Eval] completed {completed}/{total} {prompt_id}",
        f"winner={winner}",
    ]
    if {"score_base", "score_dpo", "score_gap"} <= judgment.keys():
        parts.extend(
            [
                f"base={float(judgment['score_base']):.1f}",
                f"dpo={float(judgment['score_dpo']):.1f}",
                f"gap={format_signed(float(judgment['score_gap']))}",
            ]
        )
    if "esconv_core_score_gap" in judgment:
        parts.append(f"core_gap={format_signed(float(judgment['esconv_core_score_gap']))}")
    lines = [" ".join(parts)]

    axis_scores = judgment.get("axis_scores")
    if isinstance(axis_scores, dict):
        base_axes = axis_scores.get("base")
        dpo_axes = axis_scores.get("dpo")
        if isinstance(base_axes, dict) and isinstance(dpo_axes, dict):
            axis_labels = {
                "esconv_strategy_adherence": "strategy",
                "emotional_reflection_validation": "reflection",
                "premature_advice_avoidance": "premature_advice",
                "overall_helpfulness": "helpfulness",
            }
            axis_parts = []
            for axis_key, label in axis_labels.items():
                if axis_key in base_axes and axis_key in dpo_axes:
                    gap = float(dpo_axes[axis_key]) - float(base_axes[axis_key])
                    axis_parts.append(f"{label} {format_signed(gap)}")
            if axis_parts:
                lines.append("  axes: " + " / ".join(axis_parts))

    reason = truncate_text(judgment.get("reason", ""), max_chars=reason_max_chars)
    if reason:
        lines.append(f"  reason: {reason}")
    return "\n".join(lines)


def build_partial_summary(
    judgments: list[dict[str, Any]],
    *,
    total_prompts: int,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """完了済みjudgmentから途中経過summaryを作る。"""
    completed = len(judgments)
    dpo_wins = sum(1 for row in judgments if row.get("winner") == "dpo")
    base_wins = sum(1 for row in judgments if row.get("winner") == "base")
    summary: dict[str, Any] = {
        "completed_judgments": completed,
        "total_prompts": total_prompts,
        "base_mean_so_far": None,
        "dpo_mean_so_far": None,
        "dpo_win_rate_so_far": None,
        "base_win_rate_so_far": None,
        "axis_scores_so_far": {},
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if completed:
        summary.update(
            {
                "base_mean_so_far": sum(float(row["score_base"]) for row in judgments) / completed,
                "dpo_mean_so_far": sum(float(row["score_dpo"]) for row in judgments) / completed,
                "dpo_win_rate_so_far": dpo_wins / completed,
                "base_win_rate_so_far": base_wins / completed,
            }
        )
        axis_keys = sorted(
            {
                axis_key
                for row in judgments
                if isinstance(row.get("axis_scores"), dict)
                for axis_key in row["axis_scores"].get("base", {})
                if axis_key in row["axis_scores"].get("dpo", {})
            }
        )
        summary["axis_scores_so_far"] = {
            axis_key: axis_triplet(judgments, axis_key)
            for axis_key in axis_keys
        }
    if extra_metadata:
        summary.update(extra_metadata)
        if extra_metadata.get("comparison_kind") == "lora_pair":
            summary["bayes_dpo_win_rate"] = summary["base_win_rate_so_far"]
            summary["random_dpo_win_rate"] = summary["dpo_win_rate_so_far"]
    return summary


def write_partial_summary(
    judgments: list[dict[str, Any]],
    *,
    path: Path | str,
    total_prompts: int,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """途中経過summaryを書き出す。"""
    write_json(
        build_partial_summary(
            judgments,
            total_prompts=total_prompts,
            extra_metadata=extra_metadata,
        ),
        path,
    )


def append_failure_record(
    *,
    path: Path | str | None,
    prompt_id: str,
    stage: str,
    error: Exception,
    attempts: int,
) -> None:
    """Oracle評価失敗をfailures.jsonlへ記録する。"""
    if path is None:
        return
    append_jsonl_record(
        {
            "sample_id": prompt_id,
            "prompt_id": prompt_id,
            "status": "failed",
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "attempts": attempts,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def run_with_retry(
    operation: Any,
    *,
    prompt_id: str,
    stage: str,
    retry_config: OracleRetryConfig,
) -> Any:
    """一時的なAPI/JSONエラーを指数バックオフで再試行する。"""
    attempts = retry_config.max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= attempts:
                raise
            wait_seconds = min(
                retry_config.max_seconds,
                retry_config.base_seconds * (2 ** (attempt - 1)),
            )
            print(
                f"[Oracle Eval] retry {stage} {prompt_id} attempt={attempt}/{attempts} "
                f"wait={wait_seconds:.1f}s error={type(exc).__name__}: {exc}",
                flush=True,
            )
            if wait_seconds > 0:
                time.sleep(wait_seconds)
    raise RuntimeError("unreachable retry state")


def parse_category_filter(value: str) -> set[str]:
    """カテゴリ絞り込み指定を集合へ変換する。"""
    return {item.strip() for item in value.split(",") if item.strip()}


def read_evaluation_prompts(
    path: Path | str,
    *,
    max_prompts: int | None = None,
    skip_prompts: int = 0,
    categories: set[str] | None = None,
) -> list[EvaluationPrompt]:
    """評価prompt JSONLを検証して読み込む。"""
    if skip_prompts < 0:
        raise ValueError("`skip_prompts` は0以上にしてください。")
    prompts: list[EvaluationPrompt] = []
    seen_ids: set[str] = set()
    for line_number, record in enumerate(read_jsonl(path), start=1):
        prompt_id = str(record.get("id", "")).strip()
        category = str(record.get("category", "")).strip()
        prompt = str(record.get("prompt", "")).strip()
        history = parse_prompt_history(record.get("history", []), line_number=line_number)
        axis_focus = parse_axis_focus(record.get("axis_focus", []), line_number=line_number)
        if not prompt_id:
            raise ValueError(f"{line_number}行目の `id` が空です。")
        if prompt_id in seen_ids:
            raise ValueError(f"評価prompt idが重複しています: {prompt_id}")
        if not category:
            raise ValueError(f"{line_number}行目の `category` が空です。")
        if not prompt:
            raise ValueError(f"{line_number}行目の `prompt` が空です。")
        seen_ids.add(prompt_id)
        if categories and category not in categories:
            continue
        prompts.append(
            EvaluationPrompt(
                prompt_id=prompt_id,
                category=category,
                prompt=prompt,
                history=history,
                axis_focus=axis_focus,
            )
        )
    prompts = prompts[skip_prompts:]
    return prompts[:max_prompts] if max_prompts is not None else prompts


def parse_axis_focus(payload: Any, *, line_number: int) -> tuple[str, ...]:
    """v3評価promptの主要評価軸メタデータを検証する。"""
    if payload in (None, ""):
        return ()
    if not isinstance(payload, list):
        raise ValueError(f"{line_number}行目の `axis_focus` は配列である必要があります。")
    axis_focus: list[str] = []
    for item_index, item in enumerate(payload, start=1):
        value = str(item).strip()
        if not value:
            raise ValueError(f"{line_number}行目 axis_focus[{item_index}] が空です。")
        axis_focus.append(value)
    return tuple(axis_focus)


def parse_prompt_history(payload: Any, *, line_number: int) -> tuple[dict[str, str], ...]:
    """評価promptの会話履歴を検証する。"""
    if payload in (None, ""):
        return ()
    if not isinstance(payload, list):
        raise ValueError(f"{line_number}行目の `history` は配列である必要があります。")
    history: list[dict[str, str]] = []
    for turn_index, turn in enumerate(payload, start=1):
        if not isinstance(turn, dict):
            raise ValueError(f"{line_number}行目 history[{turn_index}] はオブジェクトである必要があります。")
        speaker = str(turn.get("speaker", "")).strip()
        text = str(turn.get("text", "")).strip()
        if speaker not in {"User", "AI"}:
            raise ValueError(f"{line_number}行目 history[{turn_index}] の speaker は User または AI にしてください。")
        if not text:
            raise ValueError(f"{line_number}行目 history[{turn_index}] の text が空です。")
        history.append({"speaker": speaker, "text": text})
    return tuple(history)


def prompt_history_as_list(prompt: EvaluationPrompt) -> list[dict[str, str]]:
    """DPO prompt生成へ渡せる会話履歴リストを返す。"""
    return [dict(turn) for turn in prompt.history]


def format_prompt_context(prompt: EvaluationPrompt) -> str:
    """Oracleへ渡す評価文脈を整形する。"""
    if not prompt.history:
        return f"user_prompt:\n{prompt.prompt}"
    history_lines = "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in prompt.history)
    return (
        "conversation_context:\n"
        f"{history_lines}\n\n"
        "latest_user_prompt:\n"
        f"{prompt.prompt}"
    )


def build_context_only_prompt(prompt: EvaluationPrompt) -> str:
    """DPO学習データに近い、会話文脈だけの生成promptを作る。"""
    lines = [f"{turn['speaker']}: {turn['text']}" for turn in prompt.history]
    lines.append(f"User: {prompt.prompt}")
    lines.append("AI:")
    return "\n".join(lines)


def build_local_model_prompt(prompt: EvaluationPrompt, *, mode: str) -> str:
    """Oracle評価でbase/DPOへ渡す生成promptを作る。"""
    if mode == "instruction":
        return build_dpo_compare_prompt(prompt.prompt, history_turns=prompt_history_as_list(prompt))
    if mode == "context_only":
        return build_context_only_prompt(prompt)
    raise ValueError(f"未知のlocal_prompt_modeです: {mode}")


def load_small_corpus_context(path: Path | str, *, max_chars: int) -> str:
    """Oracleが参照する小コーパス本文を読み込む。"""
    records = read_jsonl(path)
    if records and isinstance(records[0].get("dialog"), list):
        from tools.analyze_esconv_corpus_transition_bayes import build_esconv_corpus_text

        return build_esconv_corpus_text(records, max_chars=max_chars)
    return build_corpus_text(read_small_corpus_jsonl(path), max_chars=max_chars)


def reference_constraints(style_preset: str) -> str:
    """Oracle正解応答生成のstyle別制約を返す。"""
    if style_preset == ESCONV_STRATEGY_V3_PRESET:
        return (
            "- responseは日本語で1〜2文にしてください。\n"
            "- ESConvらしい支援応答として、感情反映・感情の受容・早すぎる助言の抑制を最優先してください。\n"
            "- まず相談者の発話にある感情語、状況語、迷い、怖さ、自己否定を具体的に拾ってください。\n"
            "- 助言、解決策、断定、一般論、励ましだけで押す応答は避けてください。\n"
            "- 確認質問、問題整理、次の一歩は、文脈上必要な場合だけ短く1つ入れてください。質問がない応答でも、感情反映と受容が十分なら良い応答です。\n"
            "- 情報提供が必要な場面でも、先に不安やためらいを受け止め、情報は小さく安全に添えてください。\n"
            "- データセット名や評価用語を返答本文に出さないでください。"
        )
    if style_preset == "esconv_support":
        return (
            "- responseは日本語で1〜2文にしてください。\n"
            "- 小コーパス由来の支援的対話として、相談者が安心して次を話せる応答にしてください。\n"
            "- ESConv由来の支援Strategyを重視し、感情反映、言い換え、確認質問、肯定・安心づけ、必要最小限の提案を文脈に合わせて選んでください。\n"
            "- まず相手の悩み・状況・感情を具体語で拾い、軽い共感または言い換えを入れてください。\n"
            "- 助言や提案を入れる場合は、相手の状態を確認した後に小さな一歩として1つだけ添えてください。\n"
            "- 一般論、説教、断定、早すぎる助言、長い説明、話題逸らし、相談の早すぎる終結は避けてください。\n"
            "- データセット名や評価用語を返答本文に出さないでください。"
        )
    return (
        "- responseは日本語で1〜2文にしてください。\n"
        "- 相手が話し続けやすいように、発話内の具体語を拾い、必要な場合だけ質問を1つ添えてください。\n"
        "- 一般論、助言、長い説明、話題逸らし、過剰な推測は避けてください。\n"
        "- データセット名や評価用語を返答本文に出さないでください。"
    )


def judge_rubric_text(style_preset: str) -> str:
    """Oracle採点のstyle別基準を返す。"""
    if style_preset == ESCONV_STRATEGY_V3_PRESET:
        axis_keys = ", ".join(ESCONV_STRATEGY_V3_AXIS_KEYS)
        return (
            "採点基準:\n"
            "- oracle_responseは高品質な参照例ですが、唯一の正解文ではありません。response_a/response_bを軸別に独立評価してください。\n"
            "- ESConvらしさの主要軸は、ESConv strategy adherence、Emotional reflection / validation、Avoidance of premature adviceです。\n"
            "- 確認質問、問題探索、情報提供、次の一歩は Conversational progression として別軸で評価し、質問があるだけで主要軸やweighted評価を過大評価しないでください。\n"
            "- 感情の具体的な反映、受容、相談者の自己否定や不安を急いで直そうとしない姿勢を高く評価してください。\n"
            "- 早い助言、断定、一般論、説教、ラベル付け、文脈から外れた情報提供は、premature_advice_avoidance と strategy_adherence を下げてください。\n"
            "- responseが温かく自然でも、ユーザーの具体語や感情語に沿っていない場合は contextual_grounding を下げてください。\n"
            "- responseが共感・受容に優れる一方で質問や問題整理が弱い場合、その弱さは conversational_progression に反映し、主要3軸とは分けてください。\n\n"
            "出力JSONは次の形にしてください:\n"
            "{\n"
            "  \"scores\": {\n"
            "    \"response_a\": {各評価軸: 0〜100},\n"
            "    \"response_b\": {各評価軸: 0〜100}\n"
            "  },\n"
            "  \"winner\": \"response_a\" | \"response_b\" | \"tie\",\n"
            "  \"reason\": \"短い理由\"\n"
            "}\n"
            f"各responseに必須の評価軸: {axis_keys}\n"
            "winnerはweighted_esconv_overallで比較してください。"
        )
    if style_preset == "esconv_support":
        return (
            "採点基準:\n"
            "- 100点: oracle_responseと同等に、小コーパス由来の支援的対話スタイルとESConv由来Strategyを満たす。\n"
            "- 90点: 感情反映、具体状況の言い換え、適切な確認質問または小さな提案が自然に入っている。\n"
            "- 80点: かなり良いが、感情反映・具体状況の拾い方・Strategy選択・会話継続性のどれかが少し弱い。\n"
            "- 60点: 自然だが、共感や確認が浅く、支援Strategyが一般的すぎる。\n"
            "- 40点: 一般論、早すぎる助言、断定、話題逸らし、相談の早い終結が目立つ。\n"
            "- 20点以下: 文脈不一致、不自然、相談者が話し続けにくい。\n\n"
            "rubric_scoresは、context_understanding, concrete_pickup, experiential_deepening, emotion_and_scene, "
            "conversation_continuity, avoids_generic_advice, japanese_naturalness の各項目を0〜100点で出してください。"
            "このpresetでは context_understanding は悩みの文脈理解、concrete_pickup は具体状況の拾い方、"
            "experiential_deepening は確認質問・言い換え・小さな提案による相談の進展、"
            "emotion_and_scene は感情反映と状況理解、conversation_continuity は次を話しやすい余白、"
            "avoids_generic_advice は一般論・早すぎる助言・断定の回避として採点してください。"
        )
    return (
        "採点基準:\n"
        "- 100点: oracle_responseと同等に、小コーパス由来スタイルを満たす。\n"
        "- 80点: かなり良いが、具体性・感情・会話継続性のどれかが少し弱い。\n"
        "- 60点: 自然だが、文脈の拾い方や深め方が浅い。\n"
        "- 40点: 一般論、助言、話題逸らし、早い終結が目立つ。\n"
        "- 20点以下: 文脈不一致、不自然、会話を続けにくい。\n\n"
        "rubric_scoresは、context_understanding, concrete_pickup, experiential_deepening, emotion_and_scene, "
        "conversation_continuity, avoids_generic_advice, japanese_naturalness の各項目を0〜100点で出してください。"
    )


def build_model_style_summary(model: TransitionBayesModel, *, small_corpus_text: str = "") -> str:
    """Oracleに渡す小コーパス由来スタイルの要約を作る。"""
    state_lines = "\n".join(f"- {name}: {model.state_descriptions[name]}" for name in model.states)
    observation_lines = "\n".join(f"- {name}: {model.observation_descriptions[name]}" for name in model.observations)
    corpus_section = ""
    if small_corpus_text.strip():
        corpus_section = (
            "\n\n小コーパス本文抜粋:\n"
            "以下は正解スタイルを推定する元になった小コーパスです。"
            "理想応答と採点では、この会話の進め方・応答戦略・質問の粒度を優先してください。\n"
            f"{small_corpus_text.strip()}"
        )
    return (
        f"推定されたデータセット目的:\n{model.dataset_hypothesis}\n\n"
        f"会話状態:\n{state_lines}\n\n"
        f"観測ラベル・応答戦略:\n{observation_lines}"
        f"{corpus_section}"
    )


def build_reference_instructions(
    model: TransitionBayesModel,
    *,
    small_corpus_text: str = "",
    style_preset: str = DEFAULT_STYLE_PRESET,
) -> str:
    """Oracle正解応答生成の指示を作る。"""
    return (
        "あなたは会話評価実験のOracleです。"
        "与えられたユーザー発話に対して、以下の小コーパス由来スタイルを最もよく満たす理想的なAI応答を1つ作ってください。"
        "この応答は後で100点満点の正解応答として使われます。\n\n"
        "制約:\n"
        "- 出力はJSONのみです。\n"
        f"{reference_constraints(style_preset)}\n\n"
        "必須キー: oracle_response, reason\n\n"
        f"{build_model_style_summary(model, small_corpus_text=small_corpus_text)}"
    )


def build_reference_input(prompt: EvaluationPrompt) -> str:
    """Oracle正解応答生成の入力を作る。"""
    axis_focus_section = ""
    if prompt.axis_focus:
        axis_focus_section = "\naxis_focus:\n" + "\n".join(f"- {item}" for item in prompt.axis_focus) + "\n"
    return (
        "json output only.\n"
        f"prompt_id: {prompt.prompt_id}\n"
        f"category: {prompt.category}\n\n"
        f"{axis_focus_section}"
        f"{format_prompt_context(prompt)}"
    )


def parse_reference_payload(payload: dict[str, Any]) -> dict[str, str]:
    """Oracle正解応答JSONを検証する。"""
    response = str(payload.get("oracle_response", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    if not response:
        raise ValueError("`oracle_response` が空です。")
    return {"oracle_response": response, "oracle_reason": reason}


def build_judge_instructions(
    model: TransitionBayesModel,
    *,
    small_corpus_text: str = "",
    style_preset: str = DEFAULT_STYLE_PRESET,
) -> str:
    """Oracle採点指示を作る。"""
    if style_preset == ESCONV_STRATEGY_V3_PRESET:
        return (
            "あなたは会話評価実験のOracle採点者です。"
            "response_aとresponse_bを、ESConvらしさを測る複数軸で個別に0〜100点評価してください。"
            "モデル名は伏せられているため、どちらかを優遇せず、応答本文だけで評価してください。\n\n"
            f"{judge_rubric_text(style_preset)}\n"
            "出力はJSONのみです。必須キーは scores, winner, reason です。\n\n"
            f"{build_model_style_summary(model, small_corpus_text=small_corpus_text)}"
        )
    return (
        "あなたは会話評価実験のOracle採点者です。"
        "oracle_responseを100点満点の正解応答とみなし、response_aとresponse_bを個別に0〜100点で採点してください。"
        "モデル名は伏せられているため、どちらかを優遇せず、応答本文だけで評価してください。\n\n"
        f"{judge_rubric_text(style_preset)}\n"
        "winnerは response_a, response_b, tie のいずれかです。"
        "出力はJSONのみです。必須キーは score_a, score_b, winner, rubric_scores, reason です。\n\n"
        f"{build_model_style_summary(model, small_corpus_text=small_corpus_text)}"
    )


def build_judge_input(
    *,
    prompt: EvaluationPrompt,
    oracle_response: str,
    response_a: str,
    response_b: str,
) -> str:
    """Oracle採点入力を作る。"""
    axis_focus_section = ""
    if prompt.axis_focus:
        axis_focus_section = "\naxis_focus:\n" + "\n".join(f"- {item}" for item in prompt.axis_focus) + "\n"
    return (
        "json output only.\n"
        f"prompt_id: {prompt.prompt_id}\n"
        f"category: {prompt.category}\n\n"
        f"{axis_focus_section}"
        f"{format_prompt_context(prompt)}\n\n"
        f"oracle_response_100_points:\n{oracle_response}\n\n"
        f"response_a:\n{response_a}\n\n"
        f"response_b:\n{response_b}"
    )


def _clamp_score(value: Any, *, key: str) -> float:
    """0〜100点の数値を検証して返す。"""
    if not isinstance(value, (int, float)):
        raise ValueError(f"`{key}` は数値である必要があります。")
    return max(0.0, min(100.0, float(value)))


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    """評価軸スコアから加重平均を計算する。"""
    return sum(scores[key] * weight for key, weight in weights.items())


def winner_from_gap(gap: float) -> str:
    """score差から勝者ラベルを返す。"""
    if abs(gap) < WIN_TIE_THRESHOLD:
        return "tie"
    return "response_a" if gap < 0 else "response_b"


def parse_v3_axis_scores(payload: Any, *, key: str) -> dict[str, float]:
    """v3の軸別スコアを検証する。"""
    if not isinstance(payload, dict):
        raise ValueError(f"`{key}` はオブジェクトである必要があります。")
    return {
        axis_key: _clamp_score(payload.get(axis_key), key=f"{key}.{axis_key}")
        for axis_key in ESCONV_STRATEGY_V3_AXIS_KEYS
    }


def parse_judge_payload(payload: dict[str, Any], *, style_preset: str = DEFAULT_STYLE_PRESET) -> dict[str, Any]:
    """Oracle採点JSONを検証する。"""
    if style_preset == ESCONV_STRATEGY_V3_PRESET:
        scores_payload = payload.get("scores")
        if not isinstance(scores_payload, dict):
            raise ValueError("`scores` はオブジェクトである必要があります。")
        axis_scores_a = parse_v3_axis_scores(scores_payload.get("response_a"), key="scores.response_a")
        axis_scores_b = parse_v3_axis_scores(scores_payload.get("response_b"), key="scores.response_b")
        core_score_a = weighted_score(axis_scores_a, ESCONV_CORE_WEIGHTS)
        core_score_b = weighted_score(axis_scores_b, ESCONV_CORE_WEIGHTS)
        weighted_overall_a = weighted_score(axis_scores_a, WEIGHTED_ESCONV_OVERALL_WEIGHTS)
        weighted_overall_b = weighted_score(axis_scores_b, WEIGHTED_ESCONV_OVERALL_WEIGHTS)
        winner = winner_from_gap(weighted_overall_b - weighted_overall_a)
        raw_winner = str(payload.get("winner", "")).strip()
        if raw_winner and raw_winner not in {"response_a", "response_b", "tie"}:
            raise ValueError("`winner` は response_a, response_b, tie のいずれかである必要があります。")
        return {
            "score_a": weighted_overall_a,
            "score_b": weighted_overall_b,
            "winner": winner,
            "raw_winner": raw_winner or winner,
            "rubric_scores": {
                "response_a": axis_scores_a,
                "response_b": axis_scores_b,
            },
            "axis_scores_a": axis_scores_a,
            "axis_scores_b": axis_scores_b,
            "esconv_core_score_a": core_score_a,
            "esconv_core_score_b": core_score_b,
            "weighted_esconv_overall_score_a": weighted_overall_a,
            "weighted_esconv_overall_score_b": weighted_overall_b,
            "reason": str(payload.get("reason", "")).strip(),
        }
    score_a = _clamp_score(payload.get("score_a"), key="score_a")
    score_b = _clamp_score(payload.get("score_b"), key="score_b")
    winner = str(payload.get("winner", "")).strip()
    if winner not in {"response_a", "response_b", "tie"}:
        raise ValueError("`winner` は response_a, response_b, tie のいずれかである必要があります。")
    rubric_payload = payload.get("rubric_scores")
    if not isinstance(rubric_payload, dict):
        raise ValueError("`rubric_scores` はオブジェクトである必要があります。")
    rubric_scores = {
        key: _clamp_score(rubric_payload.get(key, 0.0), key=f"rubric_scores.{key}")
        for key in (
            "context_understanding",
            "concrete_pickup",
            "experiential_deepening",
            "emotion_and_scene",
            "conversation_continuity",
            "avoids_generic_advice",
            "japanese_naturalness",
        )
    }
    return {
        "score_a": score_a,
        "score_b": score_b,
        "winner": winner,
        "rubric_scores": rubric_scores,
        "reason": str(payload.get("reason", "")).strip(),
    }


def model_order_for_prompt(prompt_id: str, *, seed: int) -> tuple[str, str]:
    """A/B順序をpromptごとに固定ランダム化する。"""
    rng = random.Random(f"{seed}:{prompt_id}")
    labels = ["base", "dpo"]
    rng.shuffle(labels)
    return labels[0], labels[1]


def reference_template_version(style_preset: str) -> str:
    """style_preset別のOracle参照生成template versionを返す。"""
    if style_preset == ESCONV_STRATEGY_V3_PRESET:
        return REFERENCE_TEMPLATE_VERSION_V3
    return REFERENCE_TEMPLATE_VERSION


def judge_template_version(style_preset: str) -> str:
    """style_preset別のOracle採点template versionを返す。"""
    if style_preset == ESCONV_STRATEGY_V3_PRESET:
        return JUDGE_TEMPLATE_VERSION_V3
    return JUDGE_TEMPLATE_VERSION


def generate_local_responses(
    prompts: list[EvaluationPrompt],
    *,
    base_model_id: str,
    lora_path: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    use_4bit: bool,
    local_prompt_mode: str,
    existing_response_records: list[dict[str, Any]] | None = None,
    responses_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """base/DPO応答を同一prompt条件で生成する。"""
    existing_by_key = records_by_sample_key(existing_response_records or [])
    if existing_by_key:
        print(f"[Oracle Eval] found existing responses: {len(existing_by_key)}", flush=True)
    missing_prompts = [prompt for prompt in prompts if prompt.prompt_id not in existing_by_key]
    bundle = None
    if missing_prompts:
        bundle = load_compare_bundle(base_model_id, lora_path, use_4bit=use_4bit)
    records: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        if prompt.prompt_id in existing_by_key:
            print(f"[Oracle Eval] skip local generation {index}/{len(prompts)} {prompt.prompt_id}", flush=True)
            records.append(existing_by_key[prompt.prompt_id])
            continue
        print(f"[Oracle Eval] local generation {index}/{len(prompts)} {prompt.prompt_id}", flush=True)
        if bundle is None:
            raise RuntimeError("local generation bundleが初期化されていません。")
        prompt_text = build_local_model_prompt(prompt, mode=local_prompt_mode)
        base_response = generate_reply(
            bundle,
            prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            use_adapter=False,
        )
        dpo_response = generate_reply(
            bundle,
            prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            use_adapter=True,
        )
        record = {
            "prompt_id": prompt.prompt_id,
            "category": prompt.category,
            "prompt": prompt.prompt,
            "history": prompt_history_as_list(prompt),
            "axis_focus": list(prompt.axis_focus),
            "model_prompt": prompt_text,
            "base_response": base_response,
            "dpo_response": dpo_response,
            "generation": {
                "base_model_id": base_model_id,
                "lora_path": lora_path,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "seed": seed,
                "use_4bit": use_4bit,
                "thinking": "disabled",
                "local_prompt_mode": local_prompt_mode,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            },
        }
        if responses_path is not None:
            append_jsonl_record(record, responses_path)
        records.append(record)
    if existing_by_key:
        print(
            f"[Oracle Eval] skipping completed response generation for "
            f"{sum(1 for prompt in prompts if prompt.prompt_id in existing_by_key)}/{len(prompts)} samples",
            flush=True,
        )
    return records


def run_oracle_judgment(
    response_records: list[dict[str, Any]],
    *,
    bayes_model: TransitionBayesModel,
    small_corpus_text: str,
    oracle_model: str,
    max_output_tokens: int,
    seed: int,
    style_preset: str,
    generator: Any,
    oracle_workers: int = 1,
    existing_judgment_records: list[dict[str, Any]] | None = None,
    judgments_path: Path | str | None = None,
    responses_path: Path | str | None = None,
    failures_path: Path | str | None = None,
    partial_summary_path: Path | str | None = None,
    partial_summary_metadata: dict[str, Any] | None = None,
    retry_config: OracleRetryConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Oracle正解応答と採点結果を生成する。"""
    retry_config = retry_config or retry_config_from_env()
    existing_judgment_by_key = records_by_sample_key(existing_judgment_records or [])
    if existing_judgment_by_key:
        print(f"[Oracle Eval] found existing judgments: {len(existing_judgment_by_key)}", flush=True)
    reference_instructions = build_reference_instructions(
        bayes_model,
        small_corpus_text=small_corpus_text,
        style_preset=style_preset,
    )
    judge_instructions = build_judge_instructions(
        bayes_model,
        small_corpus_text=small_corpus_text,
        style_preset=style_preset,
    )
    prompt_lookup = {
        record["prompt_id"]: EvaluationPrompt(
            prompt_id=record["prompt_id"],
            category=record["category"],
            prompt=record["prompt"],
            history=parse_prompt_history(record.get("history", []), line_number=0),
            axis_focus=parse_axis_focus(record.get("axis_focus", []), line_number=0),
        )
        for record in response_records
    }

    def judge_one(
        index: int,
        record: dict[str, Any],
    ) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        prompt = prompt_lookup[record["prompt_id"]]
        stage = "reference"
        try:
            if record.get("oracle_response"):
                reference = {
                    "oracle_response": str(record["oracle_response"]),
                    "oracle_reason": str(record.get("oracle_reason", "")),
                }
            else:
                print(f"[Oracle Eval] oracle reference {index}/{len(response_records)} {prompt.prompt_id}", flush=True)
                reference = run_with_retry(
                    lambda: parse_reference_payload(
                        extract_json_object(
                            generator.generate(
                                instructions=reference_instructions,
                                input_text=build_reference_input(prompt),
                                model=oracle_model,
                                max_output_tokens=max_output_tokens,
                                response_text_format={"type": "json_object"},
                            )
                        )
                    ),
                    prompt_id=prompt.prompt_id,
                    stage=stage,
                    retry_config=retry_config,
                )
            first_label, second_label = model_order_for_prompt(prompt.prompt_id, seed=seed)
            response_by_label = {
                "base": record["base_response"],
                "dpo": record["dpo_response"],
            }
            response_a = response_by_label[first_label]
            response_b = response_by_label[second_label]
            stage = "judgment"
            print(f"[Oracle Eval] oracle judgment {index}/{len(response_records)} {prompt.prompt_id}", flush=True)
            judgment = run_with_retry(
                lambda: parse_judge_payload(
                    extract_json_object(
                        generator.generate(
                            instructions=judge_instructions,
                            input_text=build_judge_input(
                                prompt=prompt,
                                oracle_response=reference["oracle_response"],
                                response_a=response_a,
                                response_b=response_b,
                            ),
                            model=oracle_model,
                            max_output_tokens=max_output_tokens,
                            response_text_format={"type": "json_object"},
                        )
                    ),
                    style_preset=style_preset,
                ),
                prompt_id=prompt.prompt_id,
                stage=stage,
                retry_config=retry_config,
            )
        except Exception as exc:
            return index, None, None, {
                "sample_id": prompt.prompt_id,
                "prompt_id": prompt.prompt_id,
                "status": "failed",
                "stage": stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "attempts": retry_config.max_retries + 1,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
        score_by_label = {
            first_label: judgment["score_a"],
            second_label: judgment["score_b"],
        }
        axis_scores_by_label: dict[str, dict[str, float]] = {}
        esconv_core_score_by_label: dict[str, float] = {}
        weighted_overall_score_by_label: dict[str, float] = {}
        if style_preset == ESCONV_STRATEGY_V3_PRESET:
            axis_scores_by_label = {
                first_label: judgment["axis_scores_a"],
                second_label: judgment["axis_scores_b"],
            }
            esconv_core_score_by_label = {
                first_label: judgment["esconv_core_score_a"],
                second_label: judgment["esconv_core_score_b"],
            }
            weighted_overall_score_by_label = {
                first_label: judgment["weighted_esconv_overall_score_a"],
                second_label: judgment["weighted_esconv_overall_score_b"],
            }
        if judgment["winner"] == "tie":
            winner_label = "tie"
        else:
            winner_label = first_label if judgment["winner"] == "response_a" else second_label
        response_with_oracle = {
            **record,
            "oracle_response": reference["oracle_response"],
            "oracle_reason": reference["oracle_reason"],
            "oracle_model": oracle_model,
            "oracle_reference_template_version": reference_template_version(style_preset),
            "style_preset": style_preset,
        }
        judgment_record = {
            "prompt_id": prompt.prompt_id,
            "category": prompt.category,
            "prompt": prompt.prompt,
            "history": prompt_history_as_list(prompt),
            "axis_focus": list(prompt.axis_focus),
            "oracle_response": reference["oracle_response"],
            "response_a_model": first_label,
            "response_b_model": second_label,
            "response_a": response_a,
            "response_b": response_b,
            "score_a": judgment["score_a"],
            "score_b": judgment["score_b"],
            "score_base": score_by_label["base"],
            "score_dpo": score_by_label["dpo"],
            "score_gap": score_by_label["dpo"] - score_by_label["base"],
            "winner": winner_label,
            "raw_winner": judgment.get("raw_winner", judgment["winner"]),
            "rubric_scores": judgment["rubric_scores"],
            "reason": judgment["reason"],
            "oracle_model": oracle_model,
            "oracle_judge_template_version": judge_template_version(style_preset),
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "style_preset": style_preset,
        }
        if style_preset == ESCONV_STRATEGY_V3_PRESET:
            judgment_record.update(
                {
                    "axis_scores": {
                        "response_a": judgment["axis_scores_a"],
                        "response_b": judgment["axis_scores_b"],
                        "base": axis_scores_by_label["base"],
                        "dpo": axis_scores_by_label["dpo"],
                    },
                    "esconv_core_score_a": judgment["esconv_core_score_a"],
                    "esconv_core_score_b": judgment["esconv_core_score_b"],
                    "esconv_core_score_base": esconv_core_score_by_label["base"],
                    "esconv_core_score_dpo": esconv_core_score_by_label["dpo"],
                    "esconv_core_score_gap": (
                        esconv_core_score_by_label["dpo"] - esconv_core_score_by_label["base"]
                    ),
                    "weighted_esconv_overall_score_a": judgment["weighted_esconv_overall_score_a"],
                    "weighted_esconv_overall_score_b": judgment["weighted_esconv_overall_score_b"],
                    "weighted_esconv_overall_score_base": weighted_overall_score_by_label["base"],
                    "weighted_esconv_overall_score_dpo": weighted_overall_score_by_label["dpo"],
                    "weighted_esconv_overall_score_gap": (
                        weighted_overall_score_by_label["dpo"] - weighted_overall_score_by_label["base"]
                    ),
                }
            )
        return index, response_with_oracle, judgment_record, None

    completed_results: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    completed_judgments: list[dict[str, Any]] = []
    pending_records: list[tuple[int, dict[str, Any]]] = []
    for index, record in enumerate(response_records, start=1):
        key = sample_key(record)
        if key in existing_judgment_by_key:
            print(f"[Oracle Eval] skip oracle judgment {index}/{len(response_records)} {key}", flush=True)
            judgment = existing_judgment_by_key[key]
            response_with_oracle = dict(record)
            if "oracle_response" not in response_with_oracle and "oracle_response" in judgment:
                response_with_oracle.update(
                    {
                        "oracle_response": judgment["oracle_response"],
                        "oracle_reason": response_with_oracle.get("oracle_reason", ""),
                        "oracle_model": oracle_model,
                        "oracle_reference_template_version": reference_template_version(style_preset),
                        "style_preset": style_preset,
                    }
                )
            completed_results.append((index, response_with_oracle, judgment))
            completed_judgments.append(judgment)
        else:
            pending_records.append((index, record))
    if existing_judgment_by_key:
        print(
            f"[Oracle Eval] resuming oracle judgments from {len(completed_judgments) + 1}/{len(response_records)}",
            flush=True,
        )
    if partial_summary_path is not None and completed_judgments:
        write_partial_summary(
            completed_judgments,
            path=partial_summary_path,
            total_prompts=len(response_records),
            extra_metadata=partial_summary_metadata,
        )

    def record_success(index: int, response_with_oracle: dict[str, Any], judgment: dict[str, Any]) -> None:
        completed_results.append((index, response_with_oracle, judgment))
        completed_judgments.append(judgment)
        if judgments_path is not None:
            append_jsonl_record(judgment, judgments_path)
        if partial_summary_path is not None:
            write_partial_summary(
                completed_judgments,
                path=partial_summary_path,
                total_prompts=len(response_records),
                extra_metadata=partial_summary_metadata,
            )
        if progress_detail_enabled():
            print(
                format_judgment_progress(
                    judgment,
                    completed=len(completed_judgments),
                    total=len(response_records),
                    reason_max_chars=reason_max_chars_from_env(),
                ),
                flush=True,
            )

    def record_failure(failure: dict[str, Any]) -> None:
        if failures_path is not None:
            append_jsonl_record(failure, failures_path)
        print(
            f"[Oracle Eval] failed {failure['stage']} {failure['prompt_id']} "
            f"{failure['error_type']}: {failure['error_message']}",
            flush=True,
        )

    if oracle_workers <= 1:
        for index, record in pending_records:
            _, response_with_oracle, judgment, failure = judge_one(index, record)
            if failure is not None:
                record_failure(failure)
                continue
            if response_with_oracle is None or judgment is None:
                continue
            record_success(index, response_with_oracle, judgment)
    else:
        print(f"[Oracle Eval] oracle parallel workers={oracle_workers}", flush=True)
        with ThreadPoolExecutor(max_workers=oracle_workers) as executor:
            futures = {
                executor.submit(judge_one, index, record): index
                for index, record in pending_records
            }
            for future in as_completed(futures):
                index, response_with_oracle, judgment, failure = future.result()
                if failure is not None:
                    record_failure(failure)
                    continue
                if response_with_oracle is None or judgment is None:
                    continue
                print(
                    f"[Oracle Eval] oracle completed {index}/{len(response_records)} "
                    f"{judgment['prompt_id']}",
                    flush=True,
                )
                record_success(index, response_with_oracle, judgment)

    ordered_results = sorted(completed_results, key=lambda item: item[0])
    responses_with_oracle = [response for _, response, _ in ordered_results]
    judgments = [judgment for _, _, judgment in ordered_results]
    return responses_with_oracle, judgments


def summarize_judgments(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    """Oracle採点結果を集計する。"""
    if not judgments:
        raise ValueError("集計対象のjudgmentがありません。")
    if all("weighted_esconv_overall_score_base" in row for row in judgments):
        return summarize_v3_judgments(judgments)
    score_base = [float(row["score_base"]) for row in judgments]
    score_dpo = [float(row["score_dpo"]) for row in judgments]
    gaps = [float(row["score_gap"]) for row in judgments]
    dpo_wins = sum(1 for row in judgments if row["winner"] == "dpo")
    base_wins = sum(1 for row in judgments if row["winner"] == "base")
    ties = sum(1 for row in judgments if row["winner"] == "tie")
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({str(row["category"]) for row in judgments}):
        rows = [row for row in judgments if row["category"] == category]
        by_category[category] = {
            "count": len(rows),
            "mean_score_base": sum(float(row["score_base"]) for row in rows) / len(rows),
            "mean_score_dpo": sum(float(row["score_dpo"]) for row in rows) / len(rows),
            "mean_score_gap": sum(float(row["score_gap"]) for row in rows) / len(rows),
            "dpo_win_rate": sum(1 for row in rows if row["winner"] == "dpo") / len(rows),
        }
    return {
        "records": len(judgments),
        "mean_score_base": sum(score_base) / len(score_base),
        "mean_score_dpo": sum(score_dpo) / len(score_dpo),
        "mean_score_gap": sum(gaps) / len(gaps),
        "dpo_win_rate": dpo_wins / len(judgments),
        "base_win_rate": base_wins / len(judgments),
        "tie_rate": ties / len(judgments),
        "dpo_wins": dpo_wins,
        "base_wins": base_wins,
        "ties": ties,
        "by_category": by_category,
    }


def mean_value(rows: list[dict[str, Any]], key: str) -> float:
    """dictリスト内の数値平均を返す。"""
    return sum(float(row[key]) for row in rows) / len(rows)


def score_triplet(rows: list[dict[str, Any]], *, base_key: str, dpo_key: str, gap_key: str) -> dict[str, float]:
    """base/dpo/gapの平均をまとめる。"""
    return {
        "mean_base": mean_value(rows, base_key),
        "mean_dpo": mean_value(rows, dpo_key),
        "mean_gap": mean_value(rows, gap_key),
    }


def axis_triplet(rows: list[dict[str, Any]], axis_key: str) -> dict[str, float]:
    """v3軸別base/dpo/gap平均を返す。"""
    base_values = [float(row["axis_scores"]["base"][axis_key]) for row in rows]
    dpo_values = [float(row["axis_scores"]["dpo"][axis_key]) for row in rows]
    gaps = [dpo - base for base, dpo in zip(base_values, dpo_values)]
    return {
        "mean_base": sum(base_values) / len(base_values),
        "mean_dpo": sum(dpo_values) / len(dpo_values),
        "mean_gap": sum(gaps) / len(gaps),
        "dpo_win_rate": sum(1 for gap in gaps if gap >= WIN_TIE_THRESHOLD) / len(gaps),
        "base_win_rate": sum(1 for gap in gaps if gap <= -WIN_TIE_THRESHOLD) / len(gaps),
    }


def example_record(row: dict[str, Any]) -> dict[str, Any]:
    """summaryに載せる代表例を短い形へ整える。"""
    return {
        "prompt_id": row["prompt_id"],
        "category": row["category"],
        "prompt": row["prompt"],
        "esconv_core_score_base": row["esconv_core_score_base"],
        "esconv_core_score_dpo": row["esconv_core_score_dpo"],
        "esconv_core_score_gap": row["esconv_core_score_gap"],
        "weighted_esconv_overall_score_base": row["weighted_esconv_overall_score_base"],
        "weighted_esconv_overall_score_dpo": row["weighted_esconv_overall_score_dpo"],
        "weighted_esconv_overall_score_gap": row["weighted_esconv_overall_score_gap"],
        "winner": row["winner"],
        "reason": row.get("reason", ""),
    }


def top_examples(rows: list[dict[str, Any]], *, key: str, reverse: bool, limit: int = 5) -> list[dict[str, Any]]:
    """指定scoreで並べた代表例を返す。"""
    return [example_record(row) for row in sorted(rows, key=lambda item: float(item[key]), reverse=reverse)[:limit]]


def summarize_v3_judgments(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    """ESConv strategy v3の軸別採点結果を集計する。"""
    dpo_wins = sum(1 for row in judgments if row["winner"] == "dpo")
    base_wins = sum(1 for row in judgments if row["winner"] == "base")
    ties = sum(1 for row in judgments if row["winner"] == "tie")
    axis_scores = {
        axis_key: axis_triplet(judgments, axis_key)
        for axis_key in ESCONV_STRATEGY_V3_AXIS_KEYS
    }
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({str(row["category"]) for row in judgments}):
        rows = [row for row in judgments if row["category"] == category]
        by_category[category] = {
            "count": len(rows),
            "esconv_core_score": score_triplet(
                rows,
                base_key="esconv_core_score_base",
                dpo_key="esconv_core_score_dpo",
                gap_key="esconv_core_score_gap",
            ),
            "weighted_esconv_overall": score_triplet(
                rows,
                base_key="weighted_esconv_overall_score_base",
                dpo_key="weighted_esconv_overall_score_dpo",
                gap_key="weighted_esconv_overall_score_gap",
            ),
            "axis_scores": {
                axis_key: axis_triplet(rows, axis_key)
                for axis_key in ESCONV_STRATEGY_V3_AXIS_KEYS
            },
            "dpo_win_rate": sum(1 for row in rows if row["winner"] == "dpo") / len(rows),
            "base_win_rate": sum(1 for row in rows if row["winner"] == "base") / len(rows),
        }
    return {
        "records": len(judgments),
        "score_definition": {
            "esconv_core_score": ESCONV_CORE_WEIGHTS,
            "weighted_esconv_overall": WEIGHTED_ESCONV_OVERALL_WEIGHTS,
            "winner": "weighted_esconv_overall_score_gap を1.0点未満tieとして判定",
        },
        "esconv_core_score": score_triplet(
            judgments,
            base_key="esconv_core_score_base",
            dpo_key="esconv_core_score_dpo",
            gap_key="esconv_core_score_gap",
        ),
        "weighted_esconv_overall": score_triplet(
            judgments,
            base_key="weighted_esconv_overall_score_base",
            dpo_key="weighted_esconv_overall_score_dpo",
            gap_key="weighted_esconv_overall_score_gap",
        ),
        "axis_scores": axis_scores,
        "dpo_win_rate": dpo_wins / len(judgments),
        "base_win_rate": base_wins / len(judgments),
        "tie_rate": ties / len(judgments),
        "dpo_wins": dpo_wins,
        "base_wins": base_wins,
        "ties": ties,
        "by_category": by_category,
        "dpo_esconv_core_win_examples": top_examples(
            [row for row in judgments if float(row["esconv_core_score_gap"]) >= WIN_TIE_THRESHOLD],
            key="esconv_core_score_gap",
            reverse=True,
        ),
        "dpo_weighted_overall_win_examples": top_examples(
            [row for row in judgments if float(row["weighted_esconv_overall_score_gap"]) >= WIN_TIE_THRESHOLD],
            key="weighted_esconv_overall_score_gap",
            reverse=True,
        ),
        "dpo_esconv_core_win_overall_loss_examples": top_examples(
            [
                row
                for row in judgments
                if float(row["esconv_core_score_gap"]) >= WIN_TIE_THRESHOLD
                and float(row["weighted_esconv_overall_score_gap"]) <= -WIN_TIE_THRESHOLD
            ],
            key="esconv_core_score_gap",
            reverse=True,
        ),
        "base_esconv_core_win_examples": top_examples(
            [row for row in judgments if float(row["esconv_core_score_gap"]) <= -WIN_TIE_THRESHOLD],
            key="esconv_core_score_gap",
            reverse=False,
        ),
        "base_weighted_overall_win_examples": top_examples(
            [row for row in judgments if float(row["weighted_esconv_overall_score_gap"]) <= -WIN_TIE_THRESHOLD],
            key="weighted_esconv_overall_score_gap",
            reverse=False,
        ),
    }


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    category_filter = parse_category_filter(args.categories)
    prompts = read_evaluation_prompts(
        args.prompts,
        max_prompts=args.max_prompts,
        skip_prompts=args.skip_prompts,
        categories=category_filter,
    )
    bayes_model = load_transition_bayes_model(args.bayes_model)
    small_corpus_text = load_small_corpus_context(args.small_corpus, max_chars=args.small_corpus_max_chars)
    output_dir = Path(args.output_dir)
    responses_path = output_dir / "responses.jsonl"
    judgments_path = output_dir / "judgments.jsonl"
    summary_path = output_dir / "summary.json"
    partial_summary_path = output_dir / "summary.partial.json"
    failures_path = output_dir / "failures.jsonl"
    manifest_path = output_dir / "manifest.json"

    if args.dry_run:
        print("Oracle評価 dry-run")
        print(f"  prompts: {args.prompts} ({len(prompts)} 件)")
        if args.skip_prompts:
            print(f"  skip_prompts: {args.skip_prompts}")
        if category_filter:
            print(f"  categories: {','.join(sorted(category_filter))}")
        print(f"  small_corpus: {args.small_corpus} ({len(small_corpus_text)} chars)")
        print(f"  bayes_model: {bayes_model.name}")
        print(f"  base_model_id: {args.base_model_id}")
        print(f"  lora_path: {args.lora_path}")
        print(f"  oracle_model: {args.oracle_model}")
        print(f"  oracle_workers: {max(1, args.oracle_workers)}")
        print(f"  style_preset: {args.style_preset}")
        print(f"  local_prompt_mode: {args.local_prompt_mode}")
        print(f"  output_dir: {output_dir}")
        return 0

    existing_responses = read_jsonl_lenient(responses_path)
    existing_judgments = read_jsonl_lenient(judgments_path)
    existing_judgment_keys = set(records_by_sample_key(existing_judgments))
    prompt_keys = {prompt.prompt_id for prompt in prompts}
    if summary_path.exists() and prompt_keys and prompt_keys <= existing_judgment_keys:
        print(f"[Oracle Eval] 完了済みsummaryを検出したため既存成果物を上書きしません: {summary_path}")
        print(f"[Oracle Eval] completed judgments: {len(existing_judgment_keys)}/{len(prompt_keys)}")
        return 0

    response_records = generate_local_responses(
        prompts,
        base_model_id=args.base_model_id,
        lora_path=args.lora_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        use_4bit=args.use_4bit,
        local_prompt_mode=args.local_prompt_mode,
        existing_response_records=existing_responses,
        responses_path=responses_path,
    )
    responses_with_oracle, judgments = run_oracle_judgment(
        response_records,
        bayes_model=bayes_model,
        small_corpus_text=small_corpus_text,
        oracle_model=args.oracle_model,
        max_output_tokens=args.oracle_max_output_tokens,
        seed=args.seed,
        style_preset=args.style_preset,
        generator=OpenAIResponsesGenerator(),
        oracle_workers=max(1, args.oracle_workers),
        existing_judgment_records=existing_judgments,
        judgments_path=judgments_path,
        responses_path=responses_path,
        failures_path=failures_path,
        partial_summary_path=partial_summary_path,
        retry_config=retry_config_from_env(),
    )
    if not judgments:
        raise RuntimeError("Oracle評価で成功したjudgmentがありません。failures.jsonlを確認してください。")
    summary = summarize_judgments(judgments)
    if len(judgments) == len(prompts):
        write_jsonl(responses_with_oracle, responses_path)
    write_jsonl(judgments, judgments_path)
    write_json(summary, summary_path)
    write_json(
        {
            "prompts": args.prompts,
            "small_corpus": args.small_corpus,
            "small_corpus_chars": len(small_corpus_text),
            "bayes_model": args.bayes_model,
            "output_dir": args.output_dir,
            "base_model_id": args.base_model_id,
            "lora_path": args.lora_path,
            "oracle_model": args.oracle_model,
            "oracle_workers": max(1, args.oracle_workers),
            "style_preset": args.style_preset,
            "skip_prompts": args.skip_prompts,
            "categories": sorted(category_filter),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "use_4bit": args.use_4bit,
            "local_prompt_mode": args.local_prompt_mode,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "oracle_reference_template_version": reference_template_version(args.style_preset),
            "oracle_judge_template_version": judge_template_version(args.style_preset),
        },
        manifest_path,
    )
    print(f"Oracle評価responsesを書き出しました: {responses_path}")
    print(f"Oracle評価judgmentsを書き出しました: {judgments_path}")
    print(f"Oracle評価summaryを書き出しました: {summary_path}")
    if "weighted_esconv_overall" in summary:
        weighted_summary = summary["weighted_esconv_overall"]
        core_summary = summary["esconv_core_score"]
        print(
            "結果: "
            f"weighted_esconv_overall_base={weighted_summary['mean_base']:.2f} "
            f"weighted_esconv_overall_dpo={weighted_summary['mean_dpo']:.2f} "
            f"weighted_gap={weighted_summary['mean_gap']:.2f} "
            f"esconv_core_gap={core_summary['mean_gap']:.2f} "
            f"dpo_win_rate={summary['dpo_win_rate']:.2%}"
        )
    else:
        print(
            "結果: "
            f"base_mean={summary['mean_score_base']:.2f} "
            f"dpo_mean={summary['mean_score_dpo']:.2f} "
            f"gap={summary['mean_score_gap']:.2f} "
            f"dpo_win_rate={summary['dpo_win_rate']:.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
