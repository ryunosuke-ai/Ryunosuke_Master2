"""4カテゴリOracle評価で共有する入出力、プロンプト、集計処理。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analyze_small_corpus import OpenAIResponsesGenerator, resolve_analysis_model  # noqa: E402
from tools.run_oracle_evaluation import (  # noqa: E402
    DEFAULT_ORACLE_MAX_OUTPUT_TOKENS,
    OracleRetryConfig,
    append_jsonl_record,
    read_jsonl_lenient,
    run_with_retry,
)
from tools.score_dialogue_with_bayes_model import extract_json_object, load_env_file  # noqa: E402


WIN_TIE_THRESHOLD = 0.1
TEN_POINT_WIN_TIE_THRESHOLD = 0.25
BOOTSTRAP_SAMPLES = 1000
DEFAULT_CATEGORY = "uncategorized"
DEFAULT_OUTPUT_TOKEN_LIMIT = DEFAULT_ORACLE_MAX_OUTPUT_TOKENS


@dataclass(frozen=True)
class RubricAxis:
    """1〜5点評価の軸定義。"""

    key: str
    title: str
    description: str
    high: str
    low: str
    ten_point_guidance: str = ""


@dataclass(frozen=True)
class EvaluationSpec:
    """単独採点カテゴリの評価仕様。"""

    category_key: str
    category_title: str
    output_subdir: str
    axes: tuple[RubricAxis, ...]
    prompt_version: str
    reference_note: str


@dataclass(frozen=True)
class StrategyEvaluationSpec:
    """戦略遷移カテゴリの評価仕様。"""

    category_key: str
    category_title: str
    output_subdir: str
    prompt_version: str
    reference_note: str
    score_axes: tuple[RubricAxis, ...]


@dataclass(frozen=True)
class EvalSample:
    """Oracle評価対象の1応答。"""

    sample_id: str
    model_name: str
    prompt: str
    response: str
    history: tuple[dict[str, str], ...] = ()
    category: str = DEFAULT_CATEGORY
    metadata: dict[str, Any] | None = None


def add_common_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: str,
) -> None:
    """4カテゴリ共通のCLI引数を追加する。"""
    load_env_file()
    default_judge_model = resolve_analysis_model()
    parser.add_argument("--input", required=True, help="評価対象JSONL/CSV。")
    parser.add_argument("--output_dir", default=default_output_dir, help="出力先ディレクトリ。")
    parser.add_argument("--judge_model", default=default_judge_model, help=f"評価用LLM名（既定: {default_judge_model}）。")
    parser.add_argument("--limit", type=int, default=None, help="評価件数の上限。")
    parser.add_argument("--resume", action="store_true", help="既存raw.jsonlがあれば未評価分だけ評価します。")
    parser.add_argument("--temperature", type=float, default=0.0, help="評価用LLMのtemperature。")
    parser.add_argument("--max_retries", type=int, default=5, help="API/JSON失敗時の最大リトライ回数。")
    parser.add_argument("--oracle-workers", type=int, default=1, help="Oracle評価の並列worker数。")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_OUTPUT_TOKEN_LIMIT, help="Oracle出力最大トークン数。")
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、ダミー判定でraw/summaryを生成します。")
    parser.add_argument("--seed", type=int, default=42, help="bootstrapとdry-run用seed。")
    parser.add_argument("--score-scale", type=int, choices=(5, 10), default=5, help="Oracle採点スケール。5または10。")
    parser.add_argument(
        "--pairwise-tie-threshold",
        type=float,
        default=None,
        help="pairwise Win/Tie/LossのTie判定閾値。未指定時は5点=0.1、10点=0.25。",
    )


def read_input_records(path: Path | str) -> list[dict[str, Any]]:
    """JSONLまたはCSVを読み込む。"""
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_number} をJSONとして読めません: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{input_path}:{line_number} はJSON objectである必要があります。")
            records.append(payload)
    if not records:
        raise ValueError(f"入力に有効なレコードがありません: {input_path}")
    return records


def parse_history(value: Any) -> tuple[dict[str, str], ...]:
    """入力レコードの会話履歴を標準形へ変換する。"""
    if value in (None, ""):
        return ()
    payload = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return ({"speaker": "context", "text": text},)
    if not isinstance(payload, list):
        return ({"speaker": "context", "text": str(payload)},)
    turns: list[dict[str, str]] = []
    for index, turn in enumerate(payload, start=1):
        if isinstance(turn, dict):
            speaker = str(turn.get("speaker") or turn.get("role") or f"turn_{index}").strip()
            text = str(turn.get("text") or turn.get("content") or "").strip()
        else:
            speaker = f"turn_{index}"
            text = str(turn).strip()
        if text:
            turns.append({"speaker": speaker, "text": text})
    return tuple(turns)


def _first_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    """候補キーから最初の非空文字列を返す。"""
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _sample_id(record: dict[str, Any], *, line_number: int) -> str:
    """入力レコードからsample idを作る。"""
    explicit = _first_text(record, ("sample_id", "prompt_id", "id", "item_id"))
    if explicit:
        return explicit
    conversation_id = _first_text(record, ("conversation_id", "dialogue_id"))
    turn_index = _first_text(record, ("turn_index", "turn_id"))
    if conversation_id and turn_index:
        return f"{conversation_id}:{turn_index}"
    if conversation_id:
        return conversation_id
    return f"row_{line_number}"


def _history_payload(record: dict[str, Any]) -> Any:
    """入力レコードから会話履歴フィールドを取り出す。"""
    for key in ("history", "conversation_history", "conversation_context", "context", "messages"):
        if record.get(key) not in (None, ""):
            return record[key]
    return None


def _response_candidates(record: dict[str, Any]) -> list[tuple[str, str]]:
    """1レコード内の応答候補をmodel名つきで展開する。"""
    default_model = _first_text(
        record,
        ("model_name", "model", "target_model", "evaluated_model", "response_model"),
    ) or "unknown_model"
    candidates: list[tuple[str, str]] = []
    for key in ("response", "target_response", "model_response", "assistant_response"):
        response = _first_text(record, (key,))
        if response:
            candidates.append((default_model, response))
    field_specs = (
        ("base_response", "base_field_label", "base"),
        ("dpo_response", "dpo_field_label", "dpo"),
        ("bayes_dpo_response", "", "bayes_dpo"),
        ("basis_response", "", "basis"),
        ("random_dpo_response", "", "random_dpo"),
        ("prompt_only_response", "", "prompt_only_fewshot"),
        ("comparison_response", "comparison_model_name", "comparison_model"),
    )
    for response_key, label_key, fallback_label in field_specs:
        response = _first_text(record, (response_key,))
        if not response:
            continue
        model_name = str(record.get(label_key) or fallback_label).strip() if label_key else fallback_label
        candidates.append((model_name or fallback_label, response))
    return candidates


def load_eval_samples(
    path: Path | str,
    *,
    limit: int | None = None,
    allow_dry_placeholder: bool = False,
) -> list[EvalSample]:
    """評価入力を標準サンプルへ変換する。"""
    samples: list[EvalSample] = []
    for line_number, record in enumerate(read_input_records(path), start=1):
        sample_id = _sample_id(record, line_number=line_number)
        category = _first_text(record, ("category", "source_category")) or DEFAULT_CATEGORY
        prompt = _first_text(
            record,
            ("prompt", "user_prompt", "latest_user_prompt", "input", "query"),
        )
        history = parse_history(_history_payload(record))
        if not prompt and history:
            prompt = history[-1]["text"]
            history = history[:-1]
        candidates = _response_candidates(record)
        if not candidates and allow_dry_placeholder:
            candidates = [("dry_run_model", "つらい状況を話してくださってありがとうございます。もう少し今の気持ちを聞かせてもらえますか。")]
        if not prompt:
            raise ValueError(f"{path}:{line_number} のprompt/user_promptが空です。")
        for model_name, response in candidates:
            samples.append(
                EvalSample(
                    sample_id=sample_id,
                    model_name=model_name,
                    prompt=prompt,
                    response=response,
                    history=history,
                    category=category,
                    metadata={key: value for key, value in record.items() if key not in {"response"}},
                )
            )
            if limit is not None and len(samples) >= limit:
                return samples
    if not samples:
        raise ValueError("評価対象応答がありません。`response` または `base_response`/`dpo_response` などを入力してください。")
    return samples


def raw_key(sample_id: str, model_name: str) -> str:
    """resume判定用キーを返す。"""
    return f"{sample_id}\t{model_name}"


def completed_raw_keys(records: list[dict[str, Any]]) -> set[str]:
    """既存raw出力から完了済みsample/model集合を作る。"""
    keys: set[str] = set()
    for record in records:
        sample_id = str(record.get("sample_id") or record.get("prompt_id") or "").strip()
        model_name = str(record.get("model_name") or "").strip()
        if sample_id and model_name:
            keys.add(raw_key(sample_id, model_name))
    return keys


def format_context(sample: EvalSample) -> str:
    """Oracle入力に含める会話文脈を整形する。"""
    history_text = "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in sample.history)
    if history_text:
        return f"conversation_history:\n{history_text}\n\nlatest_user_prompt:\n{sample.prompt}"
    return f"user_prompt:\n{sample.prompt}"


def score_bounds(score_scale: int) -> tuple[int, int]:
    """採点スケールから最小値・最大値を返す。"""
    if score_scale not in (5, 10):
        raise ValueError(f"score_scaleは5または10である必要があります: {score_scale}")
    return 1, score_scale


def default_pairwise_tie_threshold(score_scale: int) -> float:
    """採点スケールに応じた既定Tie閾値を返す。"""
    if score_scale == 10:
        return TEN_POINT_WIN_TIE_THRESHOLD
    if score_scale == 5:
        return WIN_TIE_THRESHOLD
    raise ValueError(f"score_scaleは5または10である必要があります: {score_scale}")


def resolve_pairwise_tie_threshold(score_scale: int, value: float | None) -> float:
    """CLI指定または既定のTie閾値を検証して返す。"""
    threshold = default_pairwise_tie_threshold(score_scale) if value is None else float(value)
    if threshold < 0:
        raise ValueError("pairwise_tie_thresholdは0以上である必要があります。")
    return threshold


def axis_rubric_text(axis: RubricAxis, *, score_scale: int) -> str:
    """評価軸のrubric文を作る。"""
    text = (
        f"### {axis.key}\n"
        f"{axis.description}\n"
        f"高得点: {axis.high}\n"
        f"低得点: {axis.low}"
    )
    if score_scale == 10 and axis.ten_point_guidance:
        text += f"\n10段階の軸別基準:\n{axis.ten_point_guidance}"
    return text


def score_definition_text(score_scale: int) -> str:
    """採点スケールごとの共通スコア定義を返す。"""
    if score_scale == 10:
        return (
            "1〜2: 評価軸をほとんど満たしていない。明確に不適切、破綻、文脈無視がある。\n"
            "3〜4: 評価軸を弱く満たしているが、不十分。表面的、文脈への適合が弱い、または問題が目立つ。\n"
            "5〜6: 最低限は満たしている。大きな破綻はないが、一般的・浅い・改善点が多い。\n"
            "7〜8: 評価軸を十分に満たしている。文脈に合い、自然で、明確に良い応答。\n"
            "9〜10: 評価軸を非常によく満たしている。文脈適合性・スタイル・自然さが高く、改善点が少ない。"
            "10点はほぼ理想的な応答に限定する。"
        )
    if score_scale == 5:
        return (
            "1: 明確に問題があり、評価軸をほとんど満たさない。\n"
            "2: 弱く満たすが、重要な不足や不自然さがある。\n"
            "3: 最低限は満たすが、浅さや一般性が残る。\n"
            "4: よく満たしており、小さな不足だけがある。\n"
            "5: 文脈に即して非常によく満たしている。"
        )
    raise ValueError(f"score_scaleは5または10である必要があります: {score_scale}")


def build_score_instructions(spec: EvaluationSpec, *, score_scale: int = 5) -> str:
    """指定スケールの評価カテゴリOracle指示を作る。"""
    score_min, score_max = score_bounds(score_scale)
    axis_keys = ", ".join(axis.key for axis in spec.axes)
    rubrics = "\n\n".join(axis_rubric_text(axis, score_scale=score_scale) for axis in spec.axes)
    return (
        "あなたは会話評価実験のLLM Oracleです。"
        f"評価カテゴリは「{spec.category_title}」です。"
        f"会話履歴と評価対象応答だけに基づいて、各評価軸を{score_min}〜{score_max}点の整数で採点してください。"
        "モデル名で有利・不利を判断せず、応答の長さだけで高評価にしないでください。"
        "長い推論過程は出力せず、reasonは1〜2文にしてください。\n\n"
        "共通スコア定義:\n"
        f"{score_definition_text(score_scale)}\n\n"
        f"評価軸:\n{rubrics}\n\n"
        "出力はJSONのみです。次のスキーマを厳守してください:\n"
        "{\n"
        f"  \"scores\": {{各評価軸: {score_min}〜{score_max}の整数}},\n"
        f"  \"overall_score\": {score_min}〜{score_max}の数値,\n"
        "  \"reason\": \"短い理由\"\n"
        "}\n"
        f"必須評価軸: {axis_keys}"
    )


def build_score_input(sample: EvalSample) -> str:
    """評価カテゴリのOracle入力を作る。"""
    return (
        "json output only.\n"
        f"sample_id: {sample.sample_id}\n"
        f"category: {sample.category}\n\n"
        f"{format_context(sample)}\n\n"
        "評価対象応答:\n"
        f"{sample.response}"
    )


def _score_in_range(value: Any, *, key: str, score_scale: int) -> int:
    """指定スケール内の整数スコアを検証する。"""
    score_min, score_max = score_bounds(score_scale)
    if not isinstance(value, (int, float)):
        raise ValueError(f"`{key}` は数値である必要があります。")
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"`{key}` は整数である必要があります。")
    score = int(numeric)
    if score < score_min or score > score_max:
        raise ValueError(f"`{key}` は{score_min}〜{score_max}点である必要があります。")
    return score


def parse_score_payload(
    payload: dict[str, Any],
    spec: EvaluationSpec,
    sample: EvalSample,
    *,
    score_scale: int = 5,
) -> dict[str, Any]:
    """単独採点JSONを検証してrawレコードへ変換する。"""
    score_min, score_max = score_bounds(score_scale)
    scores_payload = payload.get("scores")
    if not isinstance(scores_payload, dict):
        raise ValueError("`scores` はobjectである必要があります。")
    scores = {
        axis.key: _score_in_range(scores_payload.get(axis.key), key=f"scores.{axis.key}", score_scale=score_scale)
        for axis in spec.axes
    }
    overall = payload.get("overall_score")
    overall_score = sum(scores.values()) / len(scores) if overall in (None, "") or score_scale == 10 else float(overall)
    if overall_score < float(score_min) or overall_score > float(score_max):
        raise ValueError(f"`overall_score` は{score_min}〜{score_max}の範囲である必要があります。")
    return {
        "sample_id": sample.sample_id,
        "category": sample.category,
        "model_name": sample.model_name,
        "scores": scores,
        "overall_score": overall_score,
        "reason": str(payload.get("reason", "")).strip(),
        "prompt": sample.prompt,
        "history": [dict(turn) for turn in sample.history],
        "response": sample.response,
        "oracle_eval_category": spec.category_key,
        "oracle_prompt_version": spec.prompt_version,
        "score_scale": score_scale,
        "score_min": score_min,
        "score_max": score_max,
    }


def dry_score_payload(spec: EvaluationSpec, *, score_scale: int = 5) -> dict[str, Any]:
    """dry-run用の採点payloadを返す。"""
    score = 8 if score_scale == 10 else 4
    scores = {axis.key: score for axis in spec.axes}
    return {
        "scores": scores,
        "overall_score": sum(scores.values()) / len(scores),
        "reason": "dry-run用のダミー判定です。",
    }


def evaluate_score_samples(
    samples: list[EvalSample],
    *,
    spec: EvaluationSpec,
    output_dir: Path,
    judge_model: str,
    temperature: float,
    max_retries: int,
    max_output_tokens: int,
    resume: bool,
    dry_run: bool,
    oracle_workers: int,
    score_scale: int,
) -> list[dict[str, Any]]:
    """評価カテゴリを実行し、raw.jsonlへ追記する。"""
    raw_path = output_dir / "raw.jsonl"
    errors_path = output_dir / "errors.jsonl"
    existing = read_jsonl_lenient(raw_path) if resume else []
    existing_keys = completed_raw_keys(existing)
    results: list[dict[str, Any]] = list(existing)
    pending = [sample for sample in samples if raw_key(sample.sample_id, sample.model_name) not in existing_keys]
    instructions = build_score_instructions(spec, score_scale=score_scale)
    retry_config = OracleRetryConfig(max_retries=max_retries, base_seconds=5.0, max_seconds=60.0)
    generator = None if dry_run else OpenAIResponsesGenerator()

    def judge_one(index: int, sample: EvalSample) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
        try:
            if dry_run:
                payload = dry_score_payload(spec, score_scale=score_scale)
            else:
                assert generator is not None
                payload = run_with_retry(
                    lambda: extract_json_object(
                        generator.generate(
                            instructions=instructions,
                            input_text=build_score_input(sample),
                            model=judge_model,
                            max_output_tokens=max_output_tokens,
                            response_text_format={"type": "json_object"},
                        )
                    ),
                    prompt_id=sample.sample_id,
                    stage=spec.category_key,
                    retry_config=retry_config,
                )
            record = parse_score_payload(payload, spec, sample, score_scale=score_scale)
            record["judge_model"] = judge_model
            return index, record, None
        except Exception as exc:
            error = error_record(sample, stage=spec.category_key, exc=exc, attempts=max_retries + 1)
            return index, None, error

    def record_success(record: dict[str, Any]) -> None:
        append_jsonl_record(record, raw_path)
        results.append(record)
        print(
            f"[{spec.category_key}] completed {len(results)}/{len(samples)} "
            f"{record['sample_id']} {record['model_name']} overall={record['overall_score']:.2f}",
            flush=True,
        )

    def record_error(error: dict[str, Any]) -> None:
        append_jsonl_record(error, errors_path)
        print(
            f"[{spec.category_key}] failed {error['sample_id']} {error['model_name']} "
            f"{error['error_type']}: {error['error_message']}",
            flush=True,
        )

    if oracle_workers <= 1:
        for index, sample in enumerate(pending, start=1):
            _, record, error = judge_one(index, sample)
            if record is not None:
                record_success(record)
            if error is not None:
                record_error(error)
    else:
        with ThreadPoolExecutor(max_workers=oracle_workers) as executor:
            futures = {executor.submit(judge_one, index, sample): sample for index, sample in enumerate(pending, start=1)}
            for future in as_completed(futures):
                _, record, error = future.result()
                if record is not None:
                    record_success(record)
                if error is not None:
                    record_error(error)
    return sorted(results, key=lambda row: (str(row.get("sample_id")), str(row.get("model_name"))))


def error_record(sample: EvalSample, *, stage: str, exc: Exception, attempts: int) -> dict[str, Any]:
    """評価失敗レコードを作る。"""
    return {
        "sample_id": sample.sample_id,
        "model_name": sample.model_name,
        "stage": stage,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "attempts": attempts,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }


def mean(values: list[float]) -> float:
    """平均を返す。"""
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    """標本標準偏差を返す。"""
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def bootstrap_ci(values: list[float], *, seed: int, samples: int = BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    """平均の95% bootstrap信頼区間を返す。"""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(draw))
    means.sort()
    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))
    return means[low_index], means[high_index]


def summarize_score_records(records: list[dict[str, Any]], spec: EvaluationSpec, *, seed: int) -> list[dict[str, Any]]:
    """単独採点結果をモデル別summary行にする。"""
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_model[str(record["model_name"])].append(record)
    rows: list[dict[str, Any]] = []
    for model_name in sorted(by_model):
        model_records = by_model[model_name]
        overall_values = [float(row["overall_score"]) for row in model_records]
        ci_low, ci_high = bootstrap_ci(overall_values, seed=seed)
        summary: dict[str, Any] = {
            "model_name": model_name,
            "count": len(model_records),
            "overall_score_mean": mean(overall_values),
            "overall_score_std": stdev(overall_values),
            "overall_score_ci95_low": ci_low,
            "overall_score_ci95_high": ci_high,
        }
        for axis in spec.axes:
            values = [float(row["scores"][axis.key]) for row in model_records]
            summary[f"{axis.key}_mean"] = mean(values)
            summary[f"{axis.key}_std"] = stdev(values)
        rows.append(summary)
    return rows


def pairwise_winrate_rows(records: list[dict[str, Any]], *, threshold: float = WIN_TIE_THRESHOLD) -> list[dict[str, Any]]:
    """同一sample上のモデル間Win/Tie/Lossを計算する。"""
    by_sample: dict[str, dict[str, float]] = defaultdict(dict)
    for record in records:
        by_sample[str(record["sample_id"])][str(record["model_name"])] = float(record["overall_score"])

    pairs = (
        ("BASiS_vs_Base", ("basis", "bayes_dpo", "BASiS"), ("base", "Base")),
        ("BASiS_vs_Random", ("basis", "bayes_dpo", "BASiS"), ("random", "random_dpo", "Random")),
        ("Base_vs_Random", ("base", "Base"), ("random", "random_dpo", "Random")),
    )
    rows: list[dict[str, Any]] = []
    for label, left_names, right_names in pairs:
        wins = ties = losses = compared = 0
        for scores in by_sample.values():
            left_key = _find_model_key(scores, left_names)
            right_key = _find_model_key(scores, right_names)
            if left_key is None or right_key is None:
                continue
            gap = scores[left_key] - scores[right_key]
            compared += 1
            if abs(gap) < threshold:
                ties += 1
            elif gap > 0:
                wins += 1
            else:
                losses += 1
        if compared:
            rows.append(
                {
                    "comparison": label,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "count": compared,
                    "win_rate": wins / compared,
                    "tie_rate": ties / compared,
                    "loss_rate": losses / compared,
                    "threshold": threshold,
                }
            )
    return rows


def _find_model_key(scores: dict[str, float], aliases: tuple[str, ...]) -> str | None:
    """モデル名のaliasに合うキーを探す。"""
    lowered = {key.lower(): key for key in scores}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    for key in scores:
        key_lower = key.lower()
        if any(alias.lower() in key_lower for alias in aliases):
            return key
    return None


def write_csv_rows(rows: list[dict[str, Any]], path: Path | str) -> None:
    """dict行をCSVに書く。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def assert_resume_compatible(
    output_dir: Path,
    *,
    resume: bool,
    judge_model: str,
    score_scale: int,
    score_min: int,
    score_max: int,
    dry_run: bool,
) -> None:
    """resume時に既存出力と今回設定が混ざらないか確認する。"""
    if not resume:
        return
    metadata_path = output_dir / "metadata.json"
    raw_path = output_dir / "raw.jsonl"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checks = {
            "judge_model": judge_model,
            "score_min": score_min,
            "score_max": score_max,
            "dry_run": dry_run,
        }
        metadata_score_scale = metadata.get("score_scale")
        if metadata_score_scale is None:
            if score_scale != 5:
                raise ValueError(
                    f"{metadata_path} はscore_scale未記録の既存5段階評価metadataの可能性があります。"
                    "10段階評価としてresumeできません。"
                )
        elif int(metadata_score_scale) != score_scale:
            raise ValueError(
                f"{metadata_path} のscore_scale={metadata_score_scale} と今回のscore_scale={score_scale} が一致しません。"
            )
        for key, expected in checks.items():
            if key in metadata and metadata[key] != expected:
                raise ValueError(f"{metadata_path} の{key}={metadata[key]!r} と今回の{key}={expected!r} が一致しません。")
        return
    if not raw_path.exists():
        return
    records = read_jsonl_lenient(raw_path)
    for record in records:
        raw_scale = record.get("score_scale")
        if raw_scale is None:
            if score_scale != 5:
                raise ValueError(
                    f"{raw_path} にscore_scale未記録の既存rawがあります。"
                    "10段階評価としてresumeできません。"
                )
        elif int(raw_scale) != score_scale:
            raise ValueError(f"{raw_path} のscore_scale={raw_scale} と今回のscore_scale={score_scale} が一致しません。")
        raw_model = record.get("judge_model")
        if raw_model not in (None, "", judge_model):
            raise ValueError(f"{raw_path} のjudge_model={raw_model!r} と今回のjudge_model={judge_model!r} が一致しません。")


def write_metadata(
    *,
    path: Path,
    spec: EvaluationSpec | None,
    category_key: str,
    judge_model: str,
    input_path: str,
    temperature: float,
    max_retries: int,
    dry_run: bool,
    score_scale: int = 5,
    pairwise_tie_threshold: float = WIN_TIE_THRESHOLD,
    extra: dict[str, Any] | None = None,
) -> None:
    """metadata.jsonを書き出す。"""
    score_min, score_max = score_bounds(score_scale)
    payload: dict[str, Any] = {
        "oracle_eval_category": category_key,
        "input": input_path,
        "judge_model": judge_model,
        "temperature": temperature,
        "max_retries": max_retries,
        "dry_run": dry_run,
        "score_scale": score_scale,
        "score_min": score_min,
        "score_max": score_max,
        "evaluation_scale_name": "10-point Oracle evaluation" if score_scale == 10 else "5-point Oracle evaluation",
        "pairwise_tie_threshold": pairwise_tie_threshold,
        "win_tie_threshold": pairwise_tie_threshold,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if spec is not None:
        payload.update(
            {
                "category_title": spec.category_title,
                "prompt_version": spec.prompt_version,
                "reference_note": spec.reference_note,
                "axes": [
                    {
                        "key": axis.key,
                        "title": axis.title,
                        "description": axis.description,
                        "high": axis.high,
                        "low": axis.low,
                    }
                    for axis in spec.axes
                ],
            }
        )
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_score_category_cli(args: argparse.Namespace, spec: EvaluationSpec) -> int:
    """単独採点カテゴリCLIの共通本体。"""
    output_dir = Path(args.output_dir)
    score_min, score_max = score_bounds(args.score_scale)
    pairwise_tie_threshold = resolve_pairwise_tie_threshold(args.score_scale, args.pairwise_tie_threshold)
    assert_resume_compatible(
        output_dir,
        resume=args.resume,
        judge_model=args.judge_model,
        score_scale=args.score_scale,
        score_min=score_min,
        score_max=score_max,
        dry_run=args.dry_run,
    )
    samples = load_eval_samples(
        args.input,
        limit=args.limit,
        allow_dry_placeholder=args.dry_run,
    )
    records = evaluate_score_samples(
        samples,
        spec=spec,
        output_dir=output_dir,
        judge_model=args.judge_model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        resume=args.resume,
        dry_run=args.dry_run,
        oracle_workers=max(1, args.oracle_workers),
        score_scale=args.score_scale,
    )
    summary_rows = summarize_score_records(records, spec, seed=args.seed)
    pairwise_rows = pairwise_winrate_rows(records, threshold=pairwise_tie_threshold)
    write_csv_rows(summary_rows, output_dir / "summary.csv")
    write_csv_rows(pairwise_rows, output_dir / "pairwise_winrate.csv")
    write_metadata(
        path=output_dir / "metadata.json",
        spec=spec,
        category_key=spec.category_key,
        judge_model=args.judge_model,
        input_path=args.input,
        temperature=args.temperature,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
        score_scale=args.score_scale,
        pairwise_tie_threshold=pairwise_tie_threshold,
    )
    print(f"{spec.category_title} rawを書き出しました: {output_dir / 'raw.jsonl'}")
    print(f"{spec.category_title} summaryを書き出しました: {output_dir / 'summary.csv'}")
    return 0


STRATEGY_LABELS = (
    "emotional_reflection",
    "empathy_validation",
    "clarification_question",
    "exploration_question",
    "problem_reframing",
    "information_provision",
    "suggestion_advice",
    "encouragement",
    "self_disclosure",
    "other",
)
USER_STATE_LABELS = (
    "emotional_disclosure",
    "situation_description",
    "emotional_confusion",
    "feeling_organized",
    "problem_exploration",
    "solution_consideration",
    "action_planning",
    "closure",
    "other",
)

DEFAULT_STRATEGY_SCORE_AXES = (
    RubricAxis(
        key="strategy_appropriateness_score",
        title="Strategy Appropriateness",
        description="応答戦略が現在の文脈と相談者状態に合っているか。",
        high="相談者の感情や状況に合い、共感・探索・整理などの支援戦略を適切に選んでいる。",
        low="戦略が文脈に合わない、表面的、有害、または支援対話として不適切。",
        ten_point_guidance=(
            "1〜2: 戦略が文脈にほぼ合わず、支援対話として不適切または有害。\n"
            "3〜4: 一部合うが、表面的で、相談者の感情や状況への適合が弱い。\n"
            "5〜6: 最低限は合うが一般的で、ESConvらしい戦略選択としては中程度。\n"
            "7〜8: 文脈と相談者状態に合い、共感・探索・整理などの支援戦略が明確。\n"
            "9〜10: 相談者の状態に非常によく合う理想的な戦略選択。10点はほぼ理想的な場合に限る。"
        ),
    ),
    RubricAxis(
        key="transition_smoothness_score",
        title="Transition Smoothness",
        description="応答前後の相談者状態遷移が自然で支援的か。",
        high="相談者の感情整理や探索に自然につながり、会話の流れを壊さない。",
        low="応答前後の状態遷移が不自然で、唐突さや文脈からのずれが目立つ。",
        ten_point_guidance=(
            "1〜2: 応答前後の状態遷移が不自然で、会話の流れを壊している。\n"
            "3〜4: 遷移は弱く、唐突さや文脈からのずれが目立つ。\n"
            "5〜6: 大きな破綻はないが、状態変化を促す力は限定的で一般的。\n"
            "7〜8: 相談者の感情整理や探索に自然につながる滑らかな遷移。\n"
            "9〜10: 相談者が安心して次に進める、非常に自然で支援的な遷移。10点はほぼ理想的な場合に限る。"
        ),
    ),
)
DEFAULT_STRATEGY_SPEC = StrategyEvaluationSpec(
    category_key="strategy_transition",
    category_title="ESConv戦略分布・状態遷移評価",
    output_subdir="oracle_strategy_transition",
    prompt_version="oracle_strategy_transition.v1",
    reference_note=(
        "ESConv支援対話の応答戦略、相談者状態、応答前後の状態遷移をLLM Oracleで推定する評価。"
    ),
    score_axes=DEFAULT_STRATEGY_SCORE_AXES,
)
STRATEGY_SUMMARY_FIELD_BY_AXIS = {
    "strategy_appropriateness_score": "mean_strategy_appropriateness",
    "transition_smoothness_score": "mean_transition_smoothness",
}


def add_strategy_cli_args(parser: argparse.ArgumentParser, *, default_output_dir: str) -> None:
    """戦略遷移評価用CLI引数を追加する。"""
    add_common_cli_args(parser, default_output_dir=default_output_dir)
    parser.add_argument(
        "--reference_input",
        default="",
        help="ESConv参照分布に使うraw/JSONL/CSV。未指定ならideal_strategy分布を擬似参照にします。",
    )


def strategy_scale_guidance(score_scale: int, *, spec: StrategyEvaluationSpec | None = None) -> str:
    """戦略遷移評価のスケール別基準を返す。"""
    spec = spec or DEFAULT_STRATEGY_SPEC
    rubrics = "\n\n".join(axis_rubric_text(axis, score_scale=score_scale) for axis in spec.score_axes)
    return f"{score_definition_text(score_scale)}\n\n評価軸:\n{rubrics}"


def build_strategy_instructions(
    *,
    score_scale: int = 5,
    spec: StrategyEvaluationSpec | None = None,
) -> str:
    """戦略・遷移ラベル付けのOracle指示を作る。"""
    spec = spec or DEFAULT_STRATEGY_SPEC
    score_min, score_max = score_bounds(score_scale)
    score_schema = ",\n".join(
        f'    "{axis.key}": {score_min}〜{score_max}の整数' for axis in spec.score_axes
    )
    axis_keys = ", ".join(axis.key for axis in spec.score_axes)
    return (
        "あなたはESConv支援対話の戦略・状態遷移を評価するLLM Oracleです。"
        "会話履歴と評価対象応答だけに基づいて、応答戦略、応答前後の相談者状態、理想戦略、"
        f"「{spec.category_title}」の評価軸を判定してください。モデル名は採点に使わないでください。"
        "応答の長さや一般的な会話の続けやすさだけで高評価にしないでください。\n\n"
        f"応答戦略ラベル候補: {', '.join(STRATEGY_LABELS)}\n"
        f"相談者状態ラベル候補: {', '.join(USER_STATE_LABELS)}\n\n"
        "スコア定義:\n"
        f"{strategy_scale_guidance(score_scale, spec=spec)}\n\n"
        "出力はJSONのみです。次のスキーマを厳守してください:\n"
        "{\n"
        "  \"labels\": {\n"
        "    \"predicted_user_state_before_response\": \"候補ラベル\",\n"
        "    \"response_strategy\": \"候補ラベル\",\n"
        "    \"predicted_user_state_after_response\": \"候補ラベル\",\n"
        "    \"transition_type\": \"before -> strategy -> after\",\n"
        "    \"ideal_strategy_for_context\": \"候補ラベル\"\n"
        "  },\n"
        "  \"scores\": {\n"
        f"{score_schema}\n"
        "  },\n"
        "  \"reason\": \"短い理由\"\n"
        "}\n"
        f"必須評価軸: {axis_keys}"
    )


def parse_strategy_payload(
    payload: dict[str, Any],
    sample: EvalSample,
    *,
    score_scale: int = 5,
    spec: StrategyEvaluationSpec | None = None,
) -> dict[str, Any]:
    """戦略遷移Oracle JSONを検証してrawレコードへ変換する。"""
    spec = spec or DEFAULT_STRATEGY_SPEC
    score_min, score_max = score_bounds(score_scale)
    labels = payload.get("labels")
    scores = payload.get("scores")
    if not isinstance(labels, dict):
        raise ValueError("`labels` はobjectである必要があります。")
    if not isinstance(scores, dict):
        raise ValueError("`scores` はobjectである必要があります。")
    before = _label(labels.get("predicted_user_state_before_response"), USER_STATE_LABELS, "predicted_user_state_before_response")
    strategy = _label(labels.get("response_strategy"), STRATEGY_LABELS, "response_strategy")
    after = _label(labels.get("predicted_user_state_after_response"), USER_STATE_LABELS, "predicted_user_state_after_response")
    ideal = _label(labels.get("ideal_strategy_for_context"), STRATEGY_LABELS, "ideal_strategy_for_context")
    transition_type = str(labels.get("transition_type") or f"{before} -> {strategy} -> {after}").strip()
    parsed_scores = {
        axis.key: _score_in_range(
            scores.get(axis.key),
            key=f"scores.{axis.key}",
            score_scale=score_scale,
        )
        for axis in spec.score_axes
    }
    return {
        "sample_id": sample.sample_id,
        "category": sample.category,
        "model_name": sample.model_name,
        "labels": {
            "predicted_user_state_before_response": before,
            "response_strategy": strategy,
            "predicted_user_state_after_response": after,
            "transition_type": transition_type,
            "ideal_strategy_for_context": ideal,
        },
        "scores": parsed_scores,
        "overall_score": mean([float(value) for value in parsed_scores.values()]),
        "reason": str(payload.get("reason", "")).strip(),
        "prompt": sample.prompt,
        "history": [dict(turn) for turn in sample.history],
        "response": sample.response,
        "oracle_eval_category": spec.category_key,
        "oracle_prompt_version": spec.prompt_version,
        "score_scale": score_scale,
        "score_min": score_min,
        "score_max": score_max,
    }


def _label(value: Any, candidates: tuple[str, ...], key: str) -> str:
    """候補ラベルを検証する。"""
    label = str(value or "").strip()
    if label not in candidates:
        raise ValueError(f"`{key}` は候補ラベルのいずれかである必要があります: {label}")
    return label


def dry_strategy_payload(
    *,
    score_scale: int = 5,
    spec: StrategyEvaluationSpec | None = None,
) -> dict[str, Any]:
    """dry-run用の戦略payloadを返す。"""
    spec = spec or DEFAULT_STRATEGY_SPEC
    score = 8 if score_scale == 10 else 4
    return {
        "labels": {
            "predicted_user_state_before_response": "emotional_disclosure",
            "response_strategy": "empathy_validation",
            "predicted_user_state_after_response": "feeling_organized",
            "transition_type": "emotional_disclosure -> empathy_validation -> feeling_organized",
            "ideal_strategy_for_context": "empathy_validation",
        },
        "scores": {axis.key: score for axis in spec.score_axes},
        "reason": "dry-run用のダミー判定です。",
    }


def evaluate_strategy_samples(
    samples: list[EvalSample],
    *,
    spec: StrategyEvaluationSpec | None = None,
    output_dir: Path,
    judge_model: str,
    temperature: float,
    max_retries: int,
    max_output_tokens: int,
    resume: bool,
    dry_run: bool,
    oracle_workers: int,
    score_scale: int,
) -> list[dict[str, Any]]:
    """戦略遷移評価を実行し、raw.jsonlへ追記する。"""
    spec = spec or DEFAULT_STRATEGY_SPEC
    raw_path = output_dir / "raw.jsonl"
    errors_path = output_dir / "errors.jsonl"
    existing = read_jsonl_lenient(raw_path) if resume else []
    existing_keys = completed_raw_keys(existing)
    results: list[dict[str, Any]] = list(existing)
    pending = [sample for sample in samples if raw_key(sample.sample_id, sample.model_name) not in existing_keys]
    instructions = build_strategy_instructions(score_scale=score_scale, spec=spec)
    retry_config = OracleRetryConfig(max_retries=max_retries, base_seconds=5.0, max_seconds=60.0)
    generator = None if dry_run else OpenAIResponsesGenerator()

    def judge_one(index: int, sample: EvalSample) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
        try:
            if dry_run:
                payload = dry_strategy_payload(score_scale=score_scale, spec=spec)
            else:
                assert generator is not None
                payload = run_with_retry(
                    lambda: extract_json_object(
                        generator.generate(
                            instructions=instructions,
                            input_text=build_score_input(sample),
                            model=judge_model,
                            max_output_tokens=max_output_tokens,
                            response_text_format={"type": "json_object"},
                        )
                    ),
                    prompt_id=sample.sample_id,
                    stage=spec.category_key,
                    retry_config=retry_config,
                )
            record = parse_strategy_payload(payload, sample, score_scale=score_scale, spec=spec)
            record["judge_model"] = judge_model
            return index, record, None
        except Exception as exc:
            return index, None, error_record(sample, stage=spec.category_key, exc=exc, attempts=max_retries + 1)

    def record_success(record: dict[str, Any]) -> None:
        append_jsonl_record(record, raw_path)
        results.append(record)
        scores = record["scores"]
        score_text = " ".join(f"{key}={value}" for key, value in scores.items())
        print(
            f"[{spec.category_key}] completed {len(results)}/{len(samples)} "
            f"{record['sample_id']} {record['model_name']} "
            f"{score_text}",
            flush=True,
        )

    def record_error(error: dict[str, Any]) -> None:
        append_jsonl_record(error, errors_path)
        print(
            f"[{spec.category_key}] failed {error['sample_id']} {error['model_name']} "
            f"{error['error_type']}: {error['error_message']}",
            flush=True,
        )

    if oracle_workers <= 1:
        for index, sample in enumerate(pending, start=1):
            _, record, error = judge_one(index, sample)
            if record is not None:
                record_success(record)
            if error is not None:
                record_error(error)
    else:
        with ThreadPoolExecutor(max_workers=oracle_workers) as executor:
            futures = {executor.submit(judge_one, index, sample): sample for index, sample in enumerate(pending, start=1)}
            for future in as_completed(futures):
                _, record, error = future.result()
                if record is not None:
                    record_success(record)
                if error is not None:
                    record_error(error)
    return sorted(results, key=lambda row: (str(row.get("sample_id")), str(row.get("model_name"))))


def distribution(labels: list[str], universe: tuple[str, ...] | list[str] | None = None) -> dict[str, float]:
    """ラベル列から確率分布を作る。"""
    keys = list(universe or sorted(set(labels)))
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return {key: 0.0 for key in keys}
    return {key: counts.get(key, 0) / total for key in keys}


def entropy(dist: dict[str, float]) -> float:
    """Shannon entropyを返す。"""
    return -sum(value * math.log2(value) for value in dist.values() if value > 0)


def tvd(left: dict[str, float], right: dict[str, float]) -> float:
    """Total Variation Distanceを返す。"""
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def jsd(left: dict[str, float], right: dict[str, float]) -> float:
    """Jensen-Shannon Divergenceを返す。"""
    keys = set(left) | set(right)
    midpoint = {key: 0.5 * (left.get(key, 0.0) + right.get(key, 0.0)) for key in keys}
    return 0.5 * kl_div(left, midpoint) + 0.5 * kl_div(right, midpoint)


def kl_div(left: dict[str, float], right: dict[str, float]) -> float:
    """KL divergenceを返す。"""
    total = 0.0
    for key, value in left.items():
        if value <= 0:
            continue
        denom = right.get(key, 0.0)
        if denom <= 0:
            continue
        total += value * math.log2(value / denom)
    return total


def f1_scores(pairs: list[tuple[str, str]], labels: tuple[str, ...]) -> tuple[float, float, float]:
    """accuracy, macro-F1, weighted-F1を返す。"""
    if not pairs:
        return 0.0, 0.0, 0.0
    accuracy = sum(1 for predicted, ideal in pairs if predicted == ideal) / len(pairs)
    f1_by_label: dict[str, float] = {}
    support_by_label: dict[str, int] = {}
    for label in labels:
        tp = sum(1 for predicted, ideal in pairs if predicted == label and ideal == label)
        fp = sum(1 for predicted, ideal in pairs if predicted == label and ideal != label)
        fn = sum(1 for predicted, ideal in pairs if predicted != label and ideal == label)
        support = sum(1 for _, ideal in pairs if ideal == label)
        support_by_label[label] = support
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_by_label[label] = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    macro = mean(list(f1_by_label.values()))
    total_support = sum(support_by_label.values())
    weighted = (
        sum(f1_by_label[label] * support_by_label[label] for label in labels) / total_support
        if total_support
        else 0.0
    )
    return accuracy, macro, weighted


def transition_labels(record: dict[str, Any]) -> tuple[str, str, str]:
    """rawレコードから遷移ラベルを作る。"""
    labels = record["labels"]
    before = labels["predicted_user_state_before_response"]
    strategy = labels["response_strategy"]
    after = labels["predicted_user_state_after_response"]
    return (
        f"{before} -> {strategy}",
        f"{strategy} -> {after}",
        f"{before} -> {strategy} -> {after}",
    )


def load_reference_strategy_distribution(path: str, raw_records: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float], str]:
    """参照戦略分布と参照遷移分布を読む。未指定ならideal分布を使う。"""
    if path:
        records = read_input_records(path)
        strategy_labels: list[str] = []
        transition_items: list[str] = []
        for record in records:
            labels = record.get("labels") if isinstance(record.get("labels"), dict) else record
            strategy = str(labels.get("response_strategy") or labels.get("strategy") or "").strip()
            if strategy in STRATEGY_LABELS:
                strategy_labels.append(strategy)
            before = str(labels.get("predicted_user_state_before_response") or labels.get("state_before") or "").strip()
            after = str(labels.get("predicted_user_state_after_response") or labels.get("state_after") or "").strip()
            if before and strategy and after:
                transition_items.append(f"{before} -> {strategy} -> {after}")
        if strategy_labels:
            return (
                distribution(strategy_labels, STRATEGY_LABELS),
                distribution(transition_items),
                "reference_input",
            )
    ideal_labels = [row["labels"]["ideal_strategy_for_context"] for row in raw_records]
    transition_items = [transition_labels(row)[2] for row in raw_records]
    return (
        distribution(ideal_labels, STRATEGY_LABELS),
        distribution(transition_items),
        "oracle_derived_ideal_strategy_pseudo_reference",
    )


def summarize_strategy_records(
    records: list[dict[str, Any]],
    *,
    reference_input: str = "",
    spec: StrategyEvaluationSpec | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """戦略遷移結果をモデル別summary行にする。"""
    spec = spec or DEFAULT_STRATEGY_SPEC
    reference_strategy_dist, reference_transition_dist, reference_source = load_reference_strategy_distribution(reference_input, records)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_model[str(record["model_name"])].append(record)
    rows: list[dict[str, Any]] = []
    for model_name in sorted(by_model):
        model_records = by_model[model_name]
        pairs = [
            (row["labels"]["response_strategy"], row["labels"]["ideal_strategy_for_context"])
            for row in model_records
        ]
        accuracy, macro_f1, weighted_f1 = f1_scores(pairs, STRATEGY_LABELS)
        strategy_values = [row["labels"]["response_strategy"] for row in model_records]
        strategy_dist = distribution(strategy_values, STRATEGY_LABELS)
        transition_values = [transition_labels(row)[2] for row in model_records]
        transition_universe = sorted(set(transition_values) | set(reference_transition_dist))
        transition_dist = distribution(transition_values, transition_universe)
        ref_transition_aligned = {key: reference_transition_dist.get(key, 0.0) for key in transition_universe}
        most_common_strategy, most_common_count = Counter(strategy_values).most_common(1)[0]
        overall_values = [
            float(row.get("overall_score", mean([float(value) for value in row["scores"].values()])))
            for row in model_records
        ]
        summary: dict[str, Any] = {
            "model_name": model_name,
            "count": len(model_records),
            "overall_score_mean": mean(overall_values),
            "overall_score_std": stdev(overall_values),
            "strategy_accuracy": accuracy,
            "strategy_macro_f1": macro_f1,
            "strategy_weighted_f1": weighted_f1,
            "strategy_jsd_to_esconv": jsd(strategy_dist, reference_strategy_dist),
            "strategy_tvd_to_esconv": tvd(strategy_dist, reference_strategy_dist),
            "strategy_entropy": entropy(strategy_dist),
            "most_frequent_strategy": most_common_strategy,
            "most_frequent_strategy_ratio": most_common_count / len(model_records),
            "transition_jsd_to_esconv": jsd(transition_dist, ref_transition_aligned),
            "transition_tvd_to_esconv": tvd(transition_dist, ref_transition_aligned),
            "transition_entropy": entropy(transition_dist),
        }
        for axis in spec.score_axes:
            field_name = STRATEGY_SUMMARY_FIELD_BY_AXIS.get(axis.key, f"{axis.key}_mean")
            summary[field_name] = mean([float(row["scores"][axis.key]) for row in model_records])
            summary[f"{axis.key}_std"] = stdev([float(row["scores"][axis.key]) for row in model_records])
        rows.append(summary)
    return rows, reference_source


def strategy_pairwise_rows(
    records: list[dict[str, Any]],
    *,
    threshold: float = WIN_TIE_THRESHOLD,
    spec: StrategyEvaluationSpec | None = None,
) -> list[dict[str, Any]]:
    """戦略カテゴリのpairwise winrateを平均2スコアで計算する。"""
    spec = spec or DEFAULT_STRATEGY_SPEC
    converted: list[dict[str, Any]] = []
    for record in records:
        score = float(record.get("overall_score", mean([float(record["scores"][axis.key]) for axis in spec.score_axes])))
        converted.append(
            {
                "sample_id": record["sample_id"],
                "model_name": record["model_name"],
                "overall_score": score,
            }
        )
    return pairwise_winrate_rows(converted, threshold=threshold)


def run_strategy_cli(args: argparse.Namespace, spec: StrategyEvaluationSpec | None = None) -> int:
    """戦略遷移評価CLIの共通本体。"""
    spec = spec or DEFAULT_STRATEGY_SPEC
    output_dir = Path(args.output_dir)
    score_min, score_max = score_bounds(args.score_scale)
    pairwise_tie_threshold = resolve_pairwise_tie_threshold(args.score_scale, args.pairwise_tie_threshold)
    assert_resume_compatible(
        output_dir,
        resume=args.resume,
        judge_model=args.judge_model,
        score_scale=args.score_scale,
        score_min=score_min,
        score_max=score_max,
        dry_run=args.dry_run,
    )
    samples = load_eval_samples(
        args.input,
        limit=args.limit,
        allow_dry_placeholder=args.dry_run,
    )
    records = evaluate_strategy_samples(
        samples,
        spec=spec,
        output_dir=output_dir,
        judge_model=args.judge_model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        resume=args.resume,
        dry_run=args.dry_run,
        oracle_workers=max(1, args.oracle_workers),
        score_scale=args.score_scale,
    )
    summary_rows, reference_source = summarize_strategy_records(records, reference_input=args.reference_input, spec=spec)
    write_csv_rows(summary_rows, output_dir / "summary.csv")
    write_csv_rows(strategy_pairwise_rows(records, threshold=pairwise_tie_threshold, spec=spec), output_dir / "pairwise_winrate.csv")
    write_metadata(
        path=output_dir / "metadata.json",
        spec=None,
        category_key=spec.category_key,
        judge_model=args.judge_model,
        input_path=args.input,
        temperature=args.temperature,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
        score_scale=args.score_scale,
        pairwise_tie_threshold=pairwise_tie_threshold,
        extra={
            "prompt_version": spec.prompt_version,
            "category_title": spec.category_title,
            "reference_note": spec.reference_note,
            "score_axes": [axis.key for axis in spec.score_axes],
            "strategy_labels": list(STRATEGY_LABELS),
            "user_state_labels": list(USER_STATE_LABELS),
            "reference_source": reference_source,
            "reference_note": (
                "reference_input未指定時のStrategy Accuracy/F1と分布比較は、"
                "LLM Oracleが推定したideal_strategy_for_contextに基づくOracle-derived labelであり、"
                "人手gold labelではありません。"
            ),
        },
    )
    print(f"{spec.category_title} rawを書き出しました: {output_dir / 'raw.jsonl'}")
    print(f"{spec.category_title} summaryを書き出しました: {output_dir / 'summary.csv'}")
    return 0
