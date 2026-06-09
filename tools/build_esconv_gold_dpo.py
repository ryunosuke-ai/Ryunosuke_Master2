"""ESConvの高品質assistant発話をgold DPOデータへ変換する。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from core.transition_bayes_model import load_transition_bayes_model
from tools.analyze_esconv_corpus_transition_bayes import read_esconv_analysis_jsonl
from tools.analyze_small_corpus import resolve_analysis_model, write_json
from tools.audit_logging import DEFAULT_AUDIT_LOG_PATH, append_audit_log
from tools.prepare_esconv_for_analysis import (
    DEFAULT_DATASET_NAME,
    convert_esconv_dataset,
    load_esconv_dataset,
)
from tools.score_dialogue_with_bayes_model import (
    OpenAIResponsesGenerator,
    load_env_file,
    resolve_scoring_model,
)
from tools.translate_and_generate_dpo import (
    DpoGenerationStats,
    build_dpo_records,
    bayes_model_version,
    read_existing_dpo_records,
    write_jsonl,
)


DEFAULT_OUTPUT_PATH = "artifacts/datasets/esconv_gold_ja_dpo_preferences.jsonl"
DEFAULT_BAYES_MODEL_PATH = "artifacts/bayes_models/generated_transition_bayes_model_esconv.json"
DEFAULT_TARGET_RECORDS = 500
DEFAULT_MAX_CONTEXT_TURNS = 8
DEFAULT_CANDIDATES = 4
DEFAULT_MIN_SCORE_GAP = 0.25
DEFAULT_MIN_CHOSEN_POSTERIOR = 0.70
DEFAULT_MAX_REJECTED_POSTERIOR = 0.55
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_SEED = 42
STRATEGY_PRIORITY = (
    "Reflection of feelings",
    "Restatement or Paraphrasing",
    "Question",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Self-disclosure",
    "Others",
)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    load_env_file()
    parser = argparse.ArgumentParser(description="ESConvをgold DPO JSONLへ変換します。")
    parser.add_argument("--input", default="", help="prepare_esconv_for_analysis済みJSONL。空ならHFから読み込みます。")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help=f"Hugging Face dataset名（既定: {DEFAULT_DATASET_NAME}）。")
    parser.add_argument("--split", default="train", choices=("train", "validation", "test", "all"), help="ESConv split。")
    parser.add_argument("--bayes-model", default=DEFAULT_BAYES_MODEL_PATH, help=f"状態遷移ベイズモデルJSON（既定: {DEFAULT_BAYES_MODEL_PATH}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"出力DPO JSONL（既定: {DEFAULT_OUTPUT_PATH}）。")
    parser.add_argument("--model", default=resolve_analysis_model(), help="翻訳・rejected生成モデル。ESConv goldではpro系を推奨します。")
    parser.add_argument("--score-model", default=resolve_scoring_model(), help="再スコアリングモデル。")
    parser.add_argument("--target-records", type=int, default=DEFAULT_TARGET_RECORDS, help="採用するgold DPO件数の目標。")
    parser.add_argument("--max-source-conversations", type=int, default=0, help="読み込むESConv会話数。0以下なら全件。")
    parser.add_argument("--max-source-records", type=int, default=None, help="処理するassistant候補数の上限。未指定ならtargetの4倍。")
    parser.add_argument("--max-context-turns", type=int, default=DEFAULT_MAX_CONTEXT_TURNS, help="promptに含める直前発話数。")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="API最大出力トークン数。")
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES, help="rejected候補数。")
    parser.add_argument("--min-score-gap", type=float, default=DEFAULT_MIN_SCORE_GAP, help="採用するscore_gap下限。")
    parser.add_argument("--min-chosen-posterior", type=float, default=DEFAULT_MIN_CHOSEN_POSTERIOR, help="chosen posterior下限。")
    parser.add_argument("--max-rejected-posterior", type=float, default=DEFAULT_MAX_REJECTED_POSTERIOR, help="rejected posterior上限。")
    parser.add_argument(
        "--gap-rescue-max-rejected-posterior",
        type=float,
        default=None,
        help="score_gapが十分大きい場合に許容するrejected posterior上限。未指定なら救済条件を使いません。",
    )
    parser.add_argument(
        "--gap-rescue-min-score-gap",
        type=float,
        default=None,
        help="rejected上限を緩める場合に必要なscore_gap下限。未指定なら救済条件を使いません。",
    )
    parser.add_argument("--workers", type=int, default=1, help="DPO生成worker数。")
    parser.add_argument(
        "--skip-sample-errors",
        action="store_true",
        help="個別gold候補のAPI/JSON/再スコア失敗をskipして処理を継続します。",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="乱数seed。")
    parser.add_argument("--audit-log", default=DEFAULT_AUDIT_LOG_PATH, help="audit_log.mdのパス。")
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、候補件数だけ確認します。")
    return parser.parse_args()


def load_esconv_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """ESConv分析用レコードを読み込む。"""
    if args.input:
        return read_esconv_analysis_jsonl(args.input)
    dataset = load_esconv_dataset(args.dataset_name)
    max_conversations = args.max_source_conversations
    return convert_esconv_dataset(
        dataset,
        split=args.split,
        max_conversations=max_conversations if max_conversations > 0 else None,
        seed=args.seed,
        sampling="stratified",
    )


def _speaker_label(turn: dict[str, Any]) -> str:
    """ESConvの話者をUser/AIへ変換する。"""
    return "AI" if str(turn.get("speaker", "")).strip().lower() == "assistant" else "User"


def build_context_prompt(
    dialog: list[dict[str, Any]],
    *,
    target_index_zero_based: int,
    max_context_turns: int,
) -> str:
    """target直前までのESConv文脈をUser/AI形式で作る。"""
    start_index = max(0, target_index_zero_based - max_context_turns)
    lines: list[str] = []
    for turn in dialog[start_index:target_index_zero_based]:
        text = str(turn.get("text", "")).strip()
        if text:
            lines.append(f"{_speaker_label(turn)}: {text}")
    return "\n".join(lines)


def collect_gold_candidates(
    records: list[dict[str, Any]],
    *,
    max_context_turns: int,
) -> list[dict[str, Any]]:
    """ESConv assistant発話をgold chosen候補へ変換する。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        dialog = [turn for turn in record.get("dialog", []) if isinstance(turn, dict)]
        for index, turn in enumerate(dialog):
            if str(turn.get("speaker", "")).strip().lower() != "assistant":
                continue
            response = str(turn.get("text", "")).strip()
            prompt = build_context_prompt(dialog, target_index_zero_based=index, max_context_turns=max_context_turns)
            if not response or not prompt:
                continue
            strategy = str(turn.get("strategy", "")).strip() or "Others"
            candidate = {
                "conversation_id": str(record["conversation_id"]),
                "turn_index": int(turn.get("turn_index", index + 1)),
                "prompt": prompt,
                "response": response,
                "posterior": 1.0,
                "metadata": {
                    "source_dataset": "ESConv",
                    "source_split": record.get("source_split"),
                    "source_dialogue_index": record.get("source_dialogue_index"),
                    "context_turns": len([line for line in prompt.splitlines() if line.strip()]),
                    "strategy": strategy,
                    "emotion_type": record.get("emotion_type", ""),
                    "problem_type": record.get("problem_type", ""),
                    "experience_type": record.get("experience_type", ""),
                    "survey_score": record.get("survey_score", {}),
                },
            }
            grouped.setdefault(strategy, []).append(candidate)

    ordered_strategies = [strategy for strategy in STRATEGY_PRIORITY if strategy in grouped]
    ordered_strategies.extend(sorted(strategy for strategy in grouped if strategy not in set(ordered_strategies)))
    for strategy in ordered_strategies:
        grouped[strategy].sort(key=lambda item: (item["conversation_id"], item["turn_index"]))

    ordered: list[dict[str, Any]] = []
    offsets = {strategy: 0 for strategy in ordered_strategies}
    while True:
        changed = False
        for strategy in ordered_strategies:
            offset = offsets[strategy]
            bucket = grouped[strategy]
            if offset >= len(bucket):
                continue
            ordered.append(bucket[offset])
            offsets[strategy] += 1
            changed = True
        if not changed:
            break
    return ordered


def summarize_candidates(records: list[dict[str, Any]]) -> dict[str, Any]:
    """候補の概要を返す。"""
    strategies = Counter(str(record.get("metadata", {}).get("strategy", "")) for record in records)
    return {
        "records": len(records),
        "strategies": dict(strategies.most_common()),
        "max_context_turns": max((int(record.get("metadata", {}).get("context_turns") or 0) for record in records), default=0),
    }


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    esconv_records = load_esconv_records(args)
    candidates = collect_gold_candidates(esconv_records, max_context_turns=args.max_context_turns)
    max_source_records = args.max_source_records
    if max_source_records is None and args.target_records is not None:
        max_source_records = max(args.target_records * 4, args.target_records)
    source_records = candidates[:max_source_records] if max_source_records else candidates
    summary = summarize_candidates(source_records)
    if args.dry_run:
        print("ESConv gold DPO dry-run")
        print(f"  esconv_conversations: {len(esconv_records)}")
        print(f"  source_candidates: {len(source_records)} / {len(candidates)}")
        print(f"  strategies: {json.dumps(summary['strategies'], ensure_ascii=False)}")
        print(f"  target_records: {args.target_records}")
        print(f"  generation_model: {args.model}")
        print(f"  score_model: {args.score_model}")
        return 0

    bayes_model = load_transition_bayes_model(args.bayes_model)
    stats = DpoGenerationStats()
    dpo_records = build_dpo_records(
        source_records,
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
        max_records=None,
        gap_rescue_max_rejected_posterior=args.gap_rescue_max_rejected_posterior,
        gap_rescue_min_score_gap=args.gap_rescue_min_score_gap,
        target_records=args.target_records,
        workers=max(1, args.workers),
        style_preset="esconv_support",
        skip_sample_errors=args.skip_sample_errors,
        output_path=args.output,
        existing_records=read_existing_dpo_records(args.output),
        audit_log_path=args.audit_log,
        stats=stats,
    )
    write_jsonl(dpo_records, args.output)
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    write_json(
        {
            "input": args.input,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "output": args.output,
            "bayes_model": args.bayes_model,
            "bayes_model_version": bayes_model_version(args.bayes_model),
            "generation_model": args.model,
            "score_model": args.score_model,
            "seed": args.seed,
            "source_candidates": len(source_records),
            "target_records": args.target_records,
            "records_written": len(dpo_records),
            "strategy_distribution": summary["strategies"],
            "skip_sample_errors": args.skip_sample_errors,
            "generation_stats": stats.as_dict(),
        },
        manifest_path,
    )
    append_audit_log(
        title="ESConv gold DPO preferenceデータ生成",
        target_files=[args.output, str(manifest_path), args.bayes_model],
        operation="ESConv元コーパスのassistant発話をgold chosenとして、日本語DPO preference JSONLへ変換した。",
        reason="DailyDialog抽出データに加えて、元の高品質小コーパスをDPO学習へ混ぜ、ESConv支援戦略の学習効果を強めるため。",
        alternatives=[
            "DailyDialog抽出データだけで学習する案は、ESConv固有の支援戦略が薄まる可能性があるため補強した。",
            "ESConv assistant応答をそのまま英語で混ぜる案は、日本語評価・日本語チャットとの分布差が大きいため採用しなかった。",
        ],
        command=(
            "python3 -m tools.build_esconv_gold_dpo "
            f"--bayes-model {args.bayes_model} --output {args.output} "
            f"--model {args.model} --score-model {args.score_model} "
            f"--target-records {args.target_records} --workers {max(1, args.workers)}"
        ),
        before_after=[
            f"ESConv会話数: {len(esconv_records)}",
            f"候補assistant発話数: {len(source_records)}",
            f"採用DPO件数: {len(dpo_records)}",
            f"strategy分布: {json.dumps(summary['strategies'], ensure_ascii=False)}",
        ],
        risks=[
            "ESConv goldを混ぜることで評価に対する分布適合は強くなるが、完全な未知ドメイン汎化評価ではなくなる。",
            "content_filterやscore_gap条件で採用件数が不足する場合はmax-source-recordsまたはtarget-recordsを調整する必要がある。",
        ],
        audit_log_path=args.audit_log,
    )
    print(f"ESConv gold DPO JSONLを書き出しました: {args.output} ({len(dpo_records)} 件)")
    print(f"再現性manifestを書き出しました: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
