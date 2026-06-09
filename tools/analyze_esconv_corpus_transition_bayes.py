"""ESConvを専用プロンプトで分析し、状態遷移ベイズモデルJSONを生成する。"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from core.transition_bayes_model import parse_transition_bayes_model
from tools.analyze_small_corpus import (
    OpenAIResponsesGenerator,
    TextGenerator,
    extract_json_object,
    load_env_file,
    resolve_analysis_model,
    write_json,
)
from tools.analyze_small_corpus_transition_bayes import build_json_repair_instructions


DEFAULT_INPUT_PATH = "data/esconv_analysis_corpus.jsonl"
DEFAULT_OUTPUT_PATH = "artifacts/bayes_models/generated_transition_bayes_model_esconv.json"
DEFAULT_MAX_OUTPUT_TOKENS = 24000
DEFAULT_MAX_INPUT_CHARS = 180000
DEFAULT_STRATEGY_GUIDANCE = "strong"


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    load_env_file()
    parser = argparse.ArgumentParser(description="ESConv専用分析で状態遷移ベイズモデルJSONを生成します。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help=f"入力JSONL（既定: {DEFAULT_INPUT_PATH}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"出力JSON（既定: {DEFAULT_OUTPUT_PATH}）。")
    default_model = resolve_analysis_model()
    parser.add_argument("--model", default=default_model, help=f"分析LLMモデルまたはAzure deployment名（既定: {default_model}）。")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="最大出力トークン数。",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=DEFAULT_MAX_INPUT_CHARS,
        help="LLMへ渡す分析対象テキストの最大文字数。",
    )
    parser.add_argument(
        "--strategy-guidance",
        choices=("strong", "supporting"),
        default=DEFAULT_STRATEGY_GUIDANCE,
        help="annotated_strategyの扱い。strongは結果重視で強い根拠として使います。",
    )
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、入力概要だけ表示します。")
    return parser.parse_args()


def read_esconv_analysis_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """ESConv専用分析JSONLを読み込む。"""
    input_path = Path(path)
    records: list[dict[str, Any]] = []
    try:
        with input_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{line_number}行目をJSONとして読めません: {exc}") from exc
                records.append(validate_esconv_analysis_record(record, line_number=line_number))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"ESConv分析用JSONLが見つかりません: {input_path}") from exc
    if not records:
        raise ValueError("ESConv分析用JSONLに有効なレコードがありません。")
    return records


def validate_esconv_analysis_record(record: dict[str, Any], *, line_number: int) -> dict[str, Any]:
    """ESConv分析用1会話レコードの必須列を検証する。"""
    required = ("conversation_id", "dialog")
    for key in required:
        if key not in record:
            raise ValueError(f"{line_number}行目に `{key}` がありません。")
    if not str(record["conversation_id"]).strip():
        raise ValueError(f"{line_number}行目の `conversation_id` が空です。")
    dialog = record.get("dialog")
    if not isinstance(dialog, list) or len(dialog) < 2:
        raise ValueError(f"{line_number}行目の `dialog` は2発話以上の配列である必要があります。")
    for turn_index, turn in enumerate(dialog, start=1):
        if not isinstance(turn, dict):
            raise ValueError(f"{line_number}行目 dialog[{turn_index}] はオブジェクトである必要があります。")
        if not str(turn.get("speaker", "")).strip():
            raise ValueError(f"{line_number}行目 dialog[{turn_index}] の `speaker` が空です。")
        if not str(turn.get("text", "")).strip():
            raise ValueError(f"{line_number}行目 dialog[{turn_index}] の `text` が空です。")
    return record


def summarize_esconv_corpus(records: list[dict[str, Any]]) -> dict[str, Any]:
    """ESConv分析用コーパスの概要を返す。"""
    turns = [turn for record in records for turn in record.get("dialog", [])]
    strategies = Counter(
        str(turn.get("strategy"))
        for turn in turns
        if turn.get("speaker") == "assistant" and turn.get("strategy")
    )
    emotions = Counter(str(record.get("emotion_type", "")) for record in records if record.get("emotion_type"))
    problems = Counter(str(record.get("problem_type", "")) for record in records if record.get("problem_type"))
    return {
        "conversations": len(records),
        "turns": len(turns),
        "assistant_turns": sum(1 for turn in turns if turn.get("speaker") == "assistant"),
        "user_turns": sum(1 for turn in turns if turn.get("speaker") == "user"),
        "strategy_types": len(strategies),
        "top_strategies": strategies.most_common(8),
        "top_emotions": emotions.most_common(8),
        "top_problems": problems.most_common(8),
        "max_text_chars": max(len(str(turn.get("text", ""))) for turn in turns),
    }


def _compact_json(value: Any) -> str:
    """LLM入力に入れやすい短いJSON文字列へ整形する。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_esconv_corpus_text(records: list[dict[str, Any]], *, max_chars: int = DEFAULT_MAX_INPUT_CHARS) -> str:
    """ESConvの本文・アノテーションをLLM分析用テキストへ整形する。"""
    sorted_records = sorted(records, key=lambda item: str(item["conversation_id"]))
    lines: list[str] = []
    for record in sorted_records:
        lines.append(f"\n# conversation_id={record['conversation_id']}")
        lines.append("## conversation_annotations")
        lines.append(f"source_dataset: {record.get('source_dataset', 'ESConv')}")
        lines.append(f"source_split: {record.get('source_split', '')}")
        lines.append(f"experience_type: {record.get('experience_type', '')}")
        lines.append(f"emotion_type: {record.get('emotion_type', '')}")
        lines.append(f"problem_type: {record.get('problem_type', '')}")
        lines.append(f"situation: {record.get('situation', '')}")
        lines.append(f"survey_score: {_compact_json(record.get('survey_score', {}))}")
        lines.append(f"seeker_question1: {record.get('seeker_question1', '')}")
        lines.append(f"seeker_question2: {record.get('seeker_question2', '')}")
        lines.append(f"supporter_question1: {record.get('supporter_question1', '')}")
        lines.append(f"supporter_question2: {record.get('supporter_question2', '')}")
        lines.append("## dialog")
        for turn in record.get("dialog", []):
            strategy = str(turn.get("strategy", "")).strip()
            strategy_suffix = f" [annotated_strategy={strategy}]" if strategy else ""
            turn_index = int(turn.get("turn_index", 0))
            lines.append(f"{turn_index}. {turn.get('speaker')}{strategy_suffix}: {turn.get('text')}")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n...（長いためここで切り詰め）"
    return text


def build_esconv_transition_analysis_instructions(*, strategy_guidance: str = DEFAULT_STRATEGY_GUIDANCE) -> str:
    """ESConv専用の状態遷移ベイズモデル生成指示を作る。"""
    if strategy_guidance == "strong":
        strategy_policy = """
Strategy利用方針（結果重視）:
- annotated_strategy は、assistant発話の意図を示す高価値なアノテーションとして強く参照してください。
- 観測ラベルは、annotated_strategy の分布と会話本文を照合して設計してください。
- Question, Reflection of feelings, Restatement or Paraphrasing, Affirmation and Reassurance, Providing Suggestions, Information, Self-disclosure, Others などは、必要に応じて統合・再命名して、後段LLMがprompt/responseから安定分類できる粒度にしてください。
- positive側では、本文・survey・コメント上で支援的に働いているstrategy系列の尤度を高くしてください。
- negative側では、一見自然でも、感情反映不足、早すぎる助言、一般論、文脈を拾わない質問など、ESConvの支援目的から外れる応答戦略の尤度を高くしてください。
""".strip()
    else:
        strategy_policy = """
Strategy利用方針:
- annotated_strategy は有用な補助情報として参照してください。
- 観測ラベルは、本文上の応答機能とstrategyラベルの両方に整合するように設計してください。
""".strip()
    return """あなたは、会話コーパス分析、支援的対話分析、動的ベイズモデル設計の専門家です。

以下のESConv形式の小規模会話コーパスを分析し、このコーパスがどのような目的・場面・会話スタイルを重視しているかを、会話本文とアノテーションから推定してください。
その上で、このコーパスらしい複数ターン会話を大量の prompt/response データから抽出するための、状態遷移を持つベイズモデルJSONを作成してください。

この分析で使える情報:
- dialog: user と assistant の実際の会話本文
- annotated_strategy: assistant発話に付与された既存の支援戦略ラベル
- emotion_type: 会話全体に関係する感情カテゴリ
- problem_type: 会話全体に関係する問題カテゴリ
- situation: user が置かれている状況説明
- survey_score: seeker/supporter による会話評価や感情強度
- seeker/supporter comments: 会話後の自由記述コメント

重要な方針:
- データセット名や外部知識で目的を決めつけず、与えられた本文とアノテーションから推定してください。
- ESConvのアノテーションは有用な補助情報として積極的に使ってください。
- annotated_strategy を機械的にそのまま観測ラベルとしてコピーしないでください。会話本文と整合するよう、似た戦略を統合し、後段評価しやすい安定したontologyにしてください。
- emotion_type, problem_type, situation は、userのニーズや会話状態を推定する手がかりとして使ってください。
- survey_score と会話後コメントは、高品質な会話に現れやすい進行・応答戦略を推定する補助情報として使ってください。
- 低頻度すぎるラベルや、prompt/response評価で区別しにくいラベルは統合してください。
- negative_states は、攻撃的・文法破綻だけではなく、一見自然でもこのコーパスの高品質会話目的から外れる状態を表してください。

{strategy_policy}

作業手順:
1. コーパス全体を読み、本文とアノテーションからデータセット目的・会話場面・望ましい会話スタイルを推定してください。
2. 会話がどのような状態を経て進むかを推定し、状態ラベルを4〜7個作ってください。
3. 状態は「会話の進行局面・対話目的上の局面」を表し、単なる表現技法や1文だけの特徴にはしないでください。
4. 望ましい会話状態を positive_states、コーパスから外れる状態を negative_states に分類してください。
5. 状態間の遷移確率 transition_likelihoods を P(next_state | current_state) として設定してください。
6. prompt/responseから観測できるassistant応答戦略ラベルを4〜8個作ってください。
7. 観測ラベルは「そのターンの response が文脈に対して何をしているか」を表し、状態ラベルと同じ意味にしないでください。
8. 各状態で各観測ラベルが出る確率 emission_likelihoods を P(observation | state) として設定してください。
9. initial_state_prior は会話開始時に各状態がどれくらいありそうかを設定してください。

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
- ラベルは英小文字、数字、アンダースコアのみの短いラベルにしてください。
- state_descriptions と observation_descriptions には、後段の評価LLMが分類できるように日本語で具体的に書いてください。
- dataset_hypothesis には、本文とアノテーションから推定したデータセット目的を日本語で簡潔に書いてください。
- JSONを短く保つため、状態と観測は本当に必要なものだけに絞ってください。

出力JSON形式:
{
  "name": "inferred_esconv_transition_dialogue_model",
  "model_type": "transition_bayes_network",
  "states": ["state_a", "state_b", "state_c", "state_d"],
  "positive_states": ["state_b", "state_c"],
  "negative_states": ["state_d"],
  "observations": ["observation_a", "observation_b", "observation_c", "observation_d"],
  "initial_state_prior": {"state_a": 0.40, "state_b": 0.30, "state_c": 0.20, "state_d": 0.10},
  "transition_likelihoods": {
    "state_a": {"state_a": 0.20, "state_b": 0.50, "state_c": 0.20, "state_d": 0.10},
    "state_b": {"state_a": 0.10, "state_b": 0.45, "state_c": 0.35, "state_d": 0.10},
    "state_c": {"state_a": 0.10, "state_b": 0.20, "state_c": 0.60, "state_d": 0.10},
    "state_d": {"state_a": 0.15, "state_b": 0.15, "state_c": 0.10, "state_d": 0.60}
  },
  "emission_likelihoods": {
    "state_a": {"observation_a": 0.50, "observation_b": 0.20, "observation_c": 0.20, "observation_d": 0.10},
    "state_b": {"observation_a": 0.20, "observation_b": 0.50, "observation_c": 0.20, "observation_d": 0.10},
    "state_c": {"observation_a": 0.15, "observation_b": 0.25, "observation_c": 0.50, "observation_d": 0.10},
    "state_d": {"observation_a": 0.10, "observation_b": 0.15, "observation_c": 0.15, "observation_d": 0.60}
  },
  "state_descriptions": {
    "state_a": "この状態の意味を日本語で具体的に書く。",
    "state_b": "この状態の意味を日本語で具体的に書く。",
    "state_c": "この状態の意味を日本語で具体的に書く。",
    "state_d": "この状態の意味を日本語で具体的に書く。"
  },
  "observation_descriptions": {
    "observation_a": "この観測ラベルの意味を日本語で具体的に書く。",
    "observation_b": "この観測ラベルの意味を日本語で具体的に書く。",
    "observation_c": "この観測ラベルの意味を日本語で具体的に書く。",
    "observation_d": "この観測ラベルの意味を日本語で具体的に書く。"
  },
  "dataset_hypothesis": "このコーパスが重視している会話目的を、本文とアノテーションに基づいて簡潔に書く。"
}

以下が分析対象コーパスです。""".replace("{strategy_policy}", strategy_policy).strip()


def generate_esconv_transition_bayes_model(
    records: list[dict[str, Any]],
    *,
    generator: TextGenerator,
    model: str,
    max_output_tokens: int,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    strategy_guidance: str = DEFAULT_STRATEGY_GUIDANCE,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """ESConv分析用コーパスから状態遷移ベイズモデルJSONを生成する。"""
    if progress:
        progress("ESConv本文とアノテーションをLLM入力用テキストに整形しています。")
    corpus_text = build_esconv_corpus_text(records, max_chars=max_input_chars)
    if progress:
        progress(f"{model} にESConv専用状態遷移ベイズモデル生成を依頼しています。")
    output_text = generator.generate(
        instructions=build_esconv_transition_analysis_instructions(strategy_guidance=strategy_guidance),
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
        progress("ESConv専用状態遷移ベイズモデルJSONの生成と検証が完了しました。")
    return payload


def main() -> int:
    """CLIエントリポイント。"""
    started_at = time.monotonic()

    def report(message: str) -> None:
        elapsed = time.monotonic() - started_at
        print(f"[{elapsed:6.1f}s] {message}", flush=True)

    args = parse_args()
    report(f"ESConv分析用コーパスを読み込んでいます: {args.input}")
    records = read_esconv_analysis_jsonl(args.input)
    summary = summarize_esconv_corpus(records)
    report(
        "入力概要: "
        f"conversations={summary['conversations']}, "
        f"turns={summary['turns']}, "
        f"assistant_turns={summary['assistant_turns']}, "
        f"user_turns={summary['user_turns']}, "
        f"strategy_types={summary['strategy_types']}, "
        f"max_text_chars={summary['max_text_chars']}"
    )
    if args.dry_run:
        print("ESConv transition bayes dry-run")
        print(f"  model: {args.model}")
        print(f"  max_input_chars: {args.max_input_chars}")
        print(f"  strategy_guidance: {args.strategy_guidance}")
        print(f"  top_strategies: {summary['top_strategies']}")
        print(f"  top_emotions: {summary['top_emotions']}")
        print(f"  top_problems: {summary['top_problems']}")
        return 0

    payload = generate_esconv_transition_bayes_model(
        records,
        generator=OpenAIResponsesGenerator(),
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        max_input_chars=args.max_input_chars,
        strategy_guidance=args.strategy_guidance,
        progress=report,
    )
    report(f"JSONを書き出しています: {args.output}")
    write_json(payload, Path(args.output))
    report(f"ESConv専用状態遷移ベイズモデルJSONを書き出しました: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
