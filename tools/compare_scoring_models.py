"""gpt-5.4とgpt-5.4-proのベイズスコアリング結果を比較する。"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from core.transition_bayes_model import load_transition_bayes_model
from tools.analyze_small_corpus import resolve_analysis_model, write_json
from tools.score_dialogue_with_bayes_model import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    OpenAIResponsesGenerator,
    load_env_file,
    read_dialogue_records,
    resolve_scoring_model,
)
from tools.score_dialogue_with_transition_bayes_model import score_records


DEFAULT_INPUT_PATH = "data/dailydialog_for_scoring_sample.jsonl"
DEFAULT_BAYES_MODEL_PATH = "artifacts/bayes_models/generated_transition_bayes_model.json"
DEFAULT_OUTPUT_PATH = "artifacts/scored_dialogues/scoring_model_comparison.json"
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_TOP_K = 30


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    load_env_file()
    parser = argparse.ArgumentParser(description="2つの評価モデルのDailyDialogスコアリング結果を比較します。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help=f"入力JSONL（既定: {DEFAULT_INPUT_PATH}）。")
    parser.add_argument("--bayes-model", default=DEFAULT_BAYES_MODEL_PATH, help=f"状態遷移ベイズモデルJSON（既定: {DEFAULT_BAYES_MODEL_PATH}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"比較結果JSON（既定: {DEFAULT_OUTPUT_PATH}）。")
    parser.add_argument("--model-a", default=resolve_scoring_model(), help="比較対象A。既定はgpt-5.4系。")
    parser.add_argument("--model-b", default=resolve_analysis_model(), help="比較対象B。既定はgpt-5.4-pro系。")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="比較する件数。")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="上位一致率のK。")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="最大出力トークン数。")
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、件数だけ確認します。")
    return parser.parse_args()


def _record_key(record: dict[str, Any]) -> str:
    """比較用のレコードキーを返す。"""
    return f"{record.get('conversation_id')}#{record.get('turn_index')}"


def _posterior_map(records: list[dict[str, Any]]) -> dict[str, float]:
    """レコードキーごとのposterior辞書を作る。"""
    return {_record_key(record): float(record["posterior"]) for record in records}


def _summary(values: list[float]) -> dict[str, float]:
    """スコア分布の要約を返す。"""
    if not values:
        return {"count": 0.0, "mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _rank_map(scores: dict[str, float]) -> dict[str, int]:
    """高posterior順の順位を返す。"""
    return {
        key: rank
        for rank, (key, _score) in enumerate(sorted(scores.items(), key=lambda item: item[1], reverse=True), start=1)
    }


def spearman_rank_correlation(scores_a: dict[str, float], scores_b: dict[str, float]) -> float:
    """同一キー集合に対する簡易Spearman順位相関を計算する。"""
    keys = sorted(set(scores_a) & set(scores_b))
    n = len(keys)
    if n < 2:
        return 0.0
    ranks_a = _rank_map({key: scores_a[key] for key in keys})
    ranks_b = _rank_map({key: scores_b[key] for key in keys})
    squared_diff_sum = sum((ranks_a[key] - ranks_b[key]) ** 2 for key in keys)
    return 1.0 - (6.0 * squared_diff_sum) / (n * (n * n - 1))


def top_k_overlap(scores_a: dict[str, float], scores_b: dict[str, float], *, top_k: int) -> float:
    """上位K件の一致率を返す。"""
    if top_k <= 0:
        return 0.0
    top_a = {key for key, _score in sorted(scores_a.items(), key=lambda item: item[1], reverse=True)[:top_k]}
    top_b = {key for key, _score in sorted(scores_b.items(), key=lambda item: item[1], reverse=True)[:top_k]}
    if not top_a or not top_b:
        return 0.0
    return len(top_a & top_b) / min(len(top_a), len(top_b))


def compare_scored_records(
    scored_a: list[dict[str, Any]],
    scored_b: list[dict[str, Any]],
    *,
    model_a: str,
    model_b: str,
    top_k: int,
) -> dict[str, Any]:
    """2モデルのスコアリング結果を比較する。"""
    scores_a = _posterior_map(scored_a)
    scores_b = _posterior_map(scored_b)
    common_keys = sorted(set(scores_a) & set(scores_b))
    differences = [abs(scores_a[key] - scores_b[key]) for key in common_keys]
    return {
        "model_a": model_a,
        "model_b": model_b,
        "records_compared": len(common_keys),
        "score_distribution": {
            model_a: _summary([scores_a[key] for key in common_keys]),
            model_b: _summary([scores_b[key] for key in common_keys]),
        },
        "absolute_difference": _summary(differences),
        "top_k": top_k,
        "top_k_overlap": top_k_overlap(scores_a, scores_b, top_k=min(top_k, len(common_keys))),
        "spearman_rank_correlation": spearman_rank_correlation(scores_a, scores_b),
        "recommendation": "use_model_a_for_bulk_scoring"
        if differences and statistics.fmean(differences) < 0.10 and top_k_overlap(scores_a, scores_b, top_k=min(top_k, len(common_keys))) >= 0.70
        else "review_before_bulk_scoring",
    }


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    records = read_dialogue_records(args.input)[: args.sample_size]
    bayes_model = load_transition_bayes_model(args.bayes_model)
    if args.dry_run:
        print("評価モデル比較 dry-run")
        print(f"  records: {len(records)}")
        print(f"  bayes_model: {bayes_model.name}")
        print(f"  model_a: {args.model_a}")
        print(f"  model_b: {args.model_b}")
        return 0
    generator = OpenAIResponsesGenerator()
    print(f"model_aでスコアリング中: {args.model_a} ({len(records)} 件)")
    scored_a = score_records(
        records,
        bayes_model=bayes_model,
        generator=generator,
        model=args.model_a,
        max_output_tokens=args.max_output_tokens,
        progress_label=f"[STEP 2/6] scoring {args.model_a}",
    )
    print(f"model_bでスコアリング中: {args.model_b} ({len(records)} 件)")
    scored_b = score_records(
        records,
        bayes_model=bayes_model,
        generator=generator,
        model=args.model_b,
        max_output_tokens=args.max_output_tokens,
        progress_label=f"[STEP 2/6] scoring {args.model_b}",
    )
    report = compare_scored_records(
        scored_a,
        scored_b,
        model_a=args.model_a,
        model_b=args.model_b,
        top_k=args.top_k,
    )
    report["input"] = args.input
    report["bayes_model"] = args.bayes_model
    report["sample_size"] = len(records)
    report["comparison_note"] = "差が小さい場合は本番大量スコアリングにmodel_aを使い、差が大きい場合だけpro利用を検討する。"
    write_json(report, args.output)
    print(f"評価モデル比較JSONを書き出しました: {args.output}")
    print(f"recommendation: {report['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
