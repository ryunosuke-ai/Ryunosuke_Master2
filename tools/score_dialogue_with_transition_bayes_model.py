"""状態遷移ベイズモデルを使い、大きな対話データの各応答を評価する。"""

from __future__ import annotations

import argparse
import json
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


DEFAULT_INPUT_PATH = "data/large_dialogue.jsonl"
DEFAULT_MODEL_PATH = "artifacts/bayes_models/generated_transition_bayes_model.json"
DEFAULT_OUTPUT_PATH = "artifacts/scored_dialogues/transition_bayes_scored_dialogue.jsonl"


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
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、入力件数だけ確認します。")
    return parser.parse_args()


def build_transition_scoring_instructions(model: TransitionBayesModel) -> str:
    """状態遷移モデル用の観測ラベル判定指示を作る。"""
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


def _record_key(record: dict[str, Any]) -> tuple[str, int]:
    """再開判定用のレコードキーを返す。"""
    return str(record["conversation_id"]), int(record["turn_index"])


def read_existing_scored_records(path: Path | str) -> list[dict[str, Any]]:
    """既存のスコア済みJSONLを読み込む。"""
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
                raise ValueError(f"既存出力の{line_number}行目をJSONとして読めません: {exc}") from exc
    return records


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
    instructions = build_transition_scoring_instructions(bayes_model)
    sorted_records = sorted(records, key=lambda item: (str(item["conversation_id"]), int(item["turn_index"])))
    pending_records = [record for record in sorted_records if _record_key(record) not in done_keys]
    if done_keys:
        print(f"{progress_label}: resume skipped={len(done_keys)} pending={len(pending_records)}", flush=True)
    total_records = len(pending_records)
    output_file = None
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        output_file = path.open("a", encoding="utf-8")
    try:
        for index, record in enumerate(pending_records, start=1):
            conversation_id = str(record["conversation_id"])
            progress = (index / total_records * 100.0) if total_records else 100.0
            print(
                f"{progress_label}: {index}/{total_records} "
                f"({progress:.1f}%) {conversation_id}#{record['turn_index']}",
                flush=True,
            )
            prior_distribution = distribution_by_conversation.get(conversation_id)
            output_text = generator.generate(
                instructions=instructions,
                input_text=build_scoring_input(record),
                model=model,
                max_output_tokens=max_output_tokens,
                response_text_format={"type": "json_object"},
            )
            observation_score = parse_transition_observation_score(
                extract_json_object(output_text),
                bayes_model,
            )
            bayes_result = score_transition_observation(
                bayes_model,
                observation_score,
                prior_distribution=prior_distribution,
            )
            distribution_by_conversation[conversation_id] = dict(bayes_result["state_posteriors"])
            result = {
                **record,
                "prior_state_distribution": prior_distribution,
                **bayes_result,
            }
            results.append(result)
            if output_file is not None:
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()
    finally:
        if output_file is not None:
            output_file.close()
    return results


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    records = read_dialogue_records(args.input)
    bayes_model = load_transition_bayes_model(args.bayes_model)
    if args.dry_run:
        print("transition bayes scoring dry-run")
        print(f"  records: {len(records)}")
        print(f"  bayes_model: {bayes_model.name}")
        print(f"  model: {args.model or DEFAULT_MODEL}")
        return 0
    existing_results = read_existing_scored_records(args.output)
    scored = score_records(
        records,
        bayes_model=bayes_model,
        generator=OpenAIResponsesGenerator(),
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        progress_label=f"[STEP] scoring {args.model}",
        existing_results=existing_results,
        output_path=Path(args.output),
    )
    print(f"状態遷移スコア済みJSONLを書き出しました: {args.output} ({len(scored)} 件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
