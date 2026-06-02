"""小さい会話コーパスをLLMで分析し、状態遷移ベイズモデルJSONを生成する。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from core.transition_bayes_model import parse_transition_bayes_model
from tools.analyze_small_corpus import (
    TextGenerator,
    OpenAIResponsesGenerator,
    build_corpus_text,
    extract_json_object,
    load_env_file,
    read_jsonl,
    resolve_analysis_model,
    summarize_corpus,
    write_json,
)


DEFAULT_INPUT_PATH = "data/small_corpus.jsonl"
DEFAULT_OUTPUT_PATH = "artifacts/bayes_models/generated_transition_bayes_model.json"
DEFAULT_TRANSITION_MAX_OUTPUT_TOKENS = 20000


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    load_env_file()
    parser = argparse.ArgumentParser(description="小コーパスから状態遷移ベイズモデルJSONを生成します。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help=f"入力JSONL（既定: {DEFAULT_INPUT_PATH}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"出力JSON（既定: {DEFAULT_OUTPUT_PATH}）。")
    default_model = resolve_analysis_model()
    parser.add_argument("--model", default=default_model, help=f"分析LLMモデルまたはAzure deployment名（既定: {default_model}）。")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_TRANSITION_MAX_OUTPUT_TOKENS,
        help="最大出力トークン数。",
    )
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、入力概要だけ表示します。")
    return parser.parse_args()


def build_transition_analysis_instructions() -> str:
    """状態遷移ベイズモデル生成用のLLM指示を作る。"""
    return """あなたは、会話コーパス分析と動的ベイズモデル設計の専門家です。

以下の小規模会話コーパスを分析し、このコーパスがどのような目的・場面・会話スタイルを重視しているかを推定してください。
その上で、このコーパスらしい複数ターン会話を大量の prompt/response データから抽出するための、状態遷移を持つベイズモデルJSONを作成してください。

このモデルの目的:
- 小規模コーパスに含まれる会話の目的、進行状態、応答戦略、暗黙の評価基準を推定する
- 単発応答だけでなく、会話文脈に沿った状態遷移を評価する
- コーパスらしい会話状態へ進む response を高く評価する
- コーパスらしい進行から外れる response を低く評価する
- 後続処理で posterior が高い応答を DPO の chosen、低い応答を rejected に使う

作業手順:
1. コーパス全体を読み、どのような目的・場面・会話スタイルのデータセットかを推定してください。
2. 会話がどのような状態を経て進むかを推定し、状態ラベルを3〜6個作ってください。
3. 状態は「会話の進行段階・対話目的上の局面」を表し、単なる表現技法や1文だけの特徴にはしないでください。
4. 似た状態、低頻度すぎる状態、後段評価で区別しにくい状態は統合し、安定して判定できるontologyにしてください。
5. 各状態は、後段の評価LLMが prompt/response と直前までの文脈から推定できる粒度にしてください。
6. 望ましい会話状態を positive_states、コーパスから外れる状態を negative_states に分類してください。
7. negative_states は、文法的に破綻した応答や攻撃的応答だけでなく、一見自然でもこのコーパスの目的・進行・応答戦略から外れる状態を表してください。
8. 状態間の遷移確率 transition_likelihoods を P(next_state | current_state) として設定してください。
9. prompt/response から観測できる応答戦略ラベルを3〜6個作ってください。
10. 観測ラベルは「そのターンの response が文脈に対して何をしているか」を表し、状態ラベルと同じ意味にしないでください。
11. 各状態で各観測ラベルが出る確率 emission_likelihoods を P(observation | state) として設定してください。
12. initial_state_prior は会話開始時に各状態がどれくらいありそうかを設定してください。

重要:
- データセットの目的や会話スタイルは、事前知識で決めつけず、与えられたコーパスから推定してください。
- コーパスに明確に現れていない特徴を過度に追加しないでください。
- ただし、大量データ評価で negative 側を判定できるよう、コーパスらしくない状態や観測は合理的に補ってください。
- 状態と観測ラベルのontologyは固定的に使われるため、毎回ぶれにくい短いラベル名にしてください。
- 状態ラベルは会話の局面、観測ラベルは応答戦略として設計し、役割を混同しないでください。
- ラベルを細かくしすぎると大量データ評価が不安定になるため、意味が近いラベルは統合してください。
- positive_states はコーパスの暗黙の評価基準に沿う状態、negative_states はその基準から外れる状態として、後段のDPO抽出に使いやすくしてください。
- 状態ラベルと観測ラベルは、英小文字、数字、アンダースコアのみの短いラベルにしてください。
- 下の出力JSON形式に含まれるラベル名、説明文、確率値は構造を示すための例です。そのまま使わず、必ずコーパス分析に基づいて最適な states, observations, transition_likelihoods, emission_likelihoods を設計してください。

出力制約:
- 出力はJSONオブジェクトのみです。
- Markdown、説明文、コードブロックは出力しないでください。
- 必須キーは name, model_type, states, positive_states, negative_states, observations, initial_state_prior, transition_likelihoods, emission_likelihoods, state_descriptions, observation_descriptions, dataset_hypothesis です。
- model_type は "transition_bayes_network" にしてください。
- states と observations に重複を入れないでください。
- positive_states と negative_states は states に含まれるラベルのみを使ってください。
- positive_states と negative_states は重複させないでください。
- initial_state_prior は states の全ラベルを必ず含め、合計を1.0にしてください。
- transition_likelihoods は各stateを行として持ち、各行は states の全ラベルを必ず含め、行ごとの合計を1.0にしてください。
- emission_likelihoods は各stateを行として持ち、各行は observations の全ラベルを必ず含め、行ごとの合計を1.0にしてください。
- 各確率は 0.0 より大きく 1.0 より小さい数値にしてください。
- state_descriptions と observation_descriptions には、後段の評価LLMが分類できるように日本語で具体的に書いてください。
- dataset_hypothesis には、コーパスから推定したデータセット目的を日本語で簡潔に書いてください。
- JSONを短く保つため、状態と観測は本当に必要なものだけに絞ってください。

出力JSON形式:
{
  "name": "inferred_transition_dialogue_model",
  "model_type": "transition_bayes_network",
  "states": ["state_a", "state_b", "state_c"],
  "positive_states": ["state_b"],
  "negative_states": ["state_c"],
  "observations": ["observation_a", "observation_b", "observation_c"],
  "initial_state_prior": {"state_a": 0.50, "state_b": 0.30, "state_c": 0.20},
  "transition_likelihoods": {
    "state_a": {"state_a": 0.20, "state_b": 0.60, "state_c": 0.20},
    "state_b": {"state_a": 0.10, "state_b": 0.75, "state_c": 0.15},
    "state_c": {"state_a": 0.20, "state_b": 0.20, "state_c": 0.60}
  },
  "emission_likelihoods": {
    "state_a": {"observation_a": 0.60, "observation_b": 0.25, "observation_c": 0.15},
    "state_b": {"observation_a": 0.20, "observation_b": 0.65, "observation_c": 0.15},
    "state_c": {"observation_a": 0.15, "observation_b": 0.20, "observation_c": 0.65}
  },
  "state_descriptions": {
    "state_a": "この状態の意味を日本語で具体的に書く。",
    "state_b": "この状態の意味を日本語で具体的に書く。",
    "state_c": "この状態の意味を日本語で具体的に書く。"
  },
  "observation_descriptions": {
    "observation_a": "この観測ラベルの意味を日本語で具体的に書く。",
    "observation_b": "この観測ラベルの意味を日本語で具体的に書く。",
    "observation_c": "この観測ラベルの意味を日本語で具体的に書く。"
  },
  "dataset_hypothesis": "このコーパスが重視している会話目的を、分析結果に基づいて簡潔に書く。"
}

以下が分析対象コーパスです。""".strip()


def build_json_repair_instructions() -> str:
    """壊れたJSON出力を修正するためのLLM指示を作る。"""
    return """あなたはJSON修復専用のアシスタントです。

入力には、状態遷移ベイズモデルJSONとして出力されたが、構文エラーを含むテキストが渡されます。
内容やラベル名や確率値はできるだけ変更せず、JSON構文だけを修正してください。

出力制約:
- 出力は有効なJSONオブジェクトのみです。
- Markdown、説明文、コードブロックは出力しないでください。
- 必須キーは name, model_type, states, positive_states, negative_states, observations, initial_state_prior, transition_likelihoods, emission_likelihoods, state_descriptions, observation_descriptions, dataset_hypothesis です。
- model_type は "transition_bayes_network" のままにしてください。
- 末尾カンマ、カンマ抜け、括弧の不足、引用符の不足だけを修正してください。
""".strip()


def generate_transition_bayes_model(
    records: list[dict[str, Any]],
    *,
    generator: TextGenerator,
    model: str,
    max_output_tokens: int,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """小コーパスから状態遷移ベイズモデルJSONを生成する。"""
    if progress:
        progress("小コーパスをLLM入力用テキストに整形しています。")
    corpus_text = build_corpus_text(records)
    if progress:
        progress(f"{model} に状態遷移ベイズモデル生成を依頼しています。")
    output_text = generator.generate(
        instructions=build_transition_analysis_instructions(),
        input_text=corpus_text,
        model=model,
        max_output_tokens=max_output_tokens,
        response_text_format={"type": "json_object"},
    )
    try:
        if progress:
            progress("LLM出力からJSONオブジェクトを抽出しています。")
        payload = extract_json_object(output_text)
    except ValueError:
        if progress:
            progress("JSON構文の修復をLLMに依頼しています。")
        repaired_text = generator.generate(
            instructions=build_json_repair_instructions(),
            input_text=output_text,
            model=model,
            max_output_tokens=max_output_tokens,
            response_text_format={"type": "json_object"},
        )
        if progress:
            progress("修復後のLLM出力からJSONオブジェクトを抽出しています。")
        payload = extract_json_object(repaired_text)
    if progress:
        progress("状態・遷移確率・観測確率のJSON仕様を検証しています。")
    parse_transition_bayes_model(payload)
    if progress:
        progress("状態遷移ベイズモデルJSONの生成と検証が完了しました。")
    return payload


def main() -> int:
    """CLIエントリポイント。"""
    started_at = time.monotonic()

    def report(message: str) -> None:
        elapsed = time.monotonic() - started_at
        print(f"[{elapsed:6.1f}s] {message}", flush=True)

    args = parse_args()
    report(f"小コーパスを読み込んでいます: {args.input}")
    records = read_jsonl(args.input)
    summary = summarize_corpus(records)
    report(
        "入力概要: "
        f"records={summary['records']}, "
        f"conversations={summary['conversations']}, "
        f"speakers={summary['speakers']}, "
        f"max_text_chars={summary['max_text_chars']}"
    )
    if args.dry_run:
        print("transition bayes small corpus dry-run")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print(f"  model: {args.model}")
        return 0

    payload = generate_transition_bayes_model(
        records,
        generator=OpenAIResponsesGenerator(),
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        progress=report,
    )
    report(f"JSONを書き出しています: {args.output}")
    write_json(payload, Path(args.output))
    report(f"状態遷移ベイズモデルJSONを書き出しました: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
