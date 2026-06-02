"""小さい会話コーパスをLLMで分析し、ベイズモデルJSONを生成する。"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

from core.generated_bayes_model import parse_bayes_model


DEFAULT_INPUT_PATH = "data/small_corpus.jsonl"
DEFAULT_OUTPUT_PATH = "artifacts/bayes_models/generated_bayes_model.json"
DEFAULT_MODEL = "gpt-5.4-pro"
DEFAULT_MAX_OUTPUT_TOKENS = 8192
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class TextGenerator(Protocol):
    """テキスト生成器の最小インターフェース。"""

    def generate(
        self,
        *,
        instructions: str,
        input_text: str,
        model: str,
        max_output_tokens: int,
        response_text_format: dict[str, Any] | None = None,
    ) -> str:
        """LLMからテキストを生成する。"""


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    load_env_file()
    parser = argparse.ArgumentParser(description="小コーパスからベイズモデルJSONを生成します。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help=f"入力JSONL（既定: {DEFAULT_INPUT_PATH}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"出力JSON（既定: {DEFAULT_OUTPUT_PATH}）。")
    default_model = resolve_analysis_model()
    parser.add_argument("--model", default=default_model, help=f"分析LLMモデルまたはAzure deployment名（既定: {default_model}）。")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="最大出力トークン数。")
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、入力概要だけ表示します。")
    return parser.parse_args()


def load_env_file() -> None:
    """存在すれば.envを読み込む。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def read_env_value(name: str, default: str = "") -> str:
    """環境変数から空白を除いた値を読む。"""
    return os.getenv(name, default).strip() or default


def read_env_value_with_fallback(*names: str, default: str = "") -> str:
    """複数の環境変数名を順に見て、最初の有効値を返す。"""
    for name in names:
        value = read_env_value(name)
        if value:
            return value
    return default


def resolve_analysis_model() -> str:
    """分析用モデルまたはAzure deployment名を解決する。"""
    return read_env_value_with_fallback(
        "ANALYSIS_LLM_MODEL",
        "AZURE_OPENAI_GPT54_PRO_DEPLOYMENT_NAME",
        "OPENAI_GPT54_PRO_MODEL",
        default=DEFAULT_MODEL,
    )


def resolve_analysis_azure_api_key() -> str:
    """分析用Azure OpenAI APIキーを解決する。"""
    return read_env_value_with_fallback(
        "AZURE_OPENAI_GPT54_PRO_API_KEY",
        "OPENAI_GPT54_PRO_API_KEY",
        "AZURE_OPENAI_API_KEY",
    )


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """JSONLを読み込む。"""
    records: list[dict[str, Any]] = []
    input_path = Path(path)
    try:
        with input_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{line_number}行目をJSONとして読めません: {exc}") from exc
                records.append(validate_corpus_record(record, line_number=line_number))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"小コーパスJSONLが見つかりません: {input_path}") from exc
    if not records:
        raise ValueError("小コーパスJSONLに有効なレコードがありません。")
    return records


def validate_corpus_record(record: dict[str, Any], *, line_number: int) -> dict[str, Any]:
    """小コーパス1行の必須列を検証する。"""
    required = ("conversation_id", "turn_index", "speaker", "text")
    for key in required:
        if key not in record:
            raise ValueError(f"{line_number}行目に `{key}` がありません。")
    if not str(record["conversation_id"]).strip():
        raise ValueError(f"{line_number}行目の `conversation_id` が空です。")
    if not isinstance(record["turn_index"], int):
        raise ValueError(f"{line_number}行目の `turn_index` は整数である必要があります。")
    if not str(record["speaker"]).strip():
        raise ValueError(f"{line_number}行目の `speaker` が空です。")
    if not str(record["text"]).strip():
        raise ValueError(f"{line_number}行目の `text` が空です。")
    return record


def summarize_corpus(records: list[dict[str, Any]]) -> dict[str, int]:
    """小コーパスの概要を返す。"""
    conversation_ids = {str(record["conversation_id"]) for record in records}
    speakers = {str(record["speaker"]) for record in records}
    return {
        "records": len(records),
        "conversations": len(conversation_ids),
        "speakers": len(speakers),
        "max_text_chars": max(len(str(record["text"])) for record in records),
    }


def build_corpus_text(records: list[dict[str, Any]], *, max_chars: int = 120000) -> str:
    """LLM入力用に会話を整形する。"""
    sorted_records = sorted(records, key=lambda item: (str(item["conversation_id"]), int(item["turn_index"])))
    lines = []
    current_conversation = None
    for record in sorted_records:
        conversation_id = str(record["conversation_id"])
        if conversation_id != current_conversation:
            lines.append(f"\n# conversation_id={conversation_id}")
            current_conversation = conversation_id
        lines.append(f"{record['turn_index']}. {record['speaker']}: {record['text']}")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n...（長いためここで切り詰め）"
    return text


def build_analysis_instructions() -> str:
    """ベイズモデル生成用のLLM指示を作る。"""
    return """あなたは、会話コーパス分析と会話評価用ベイズモデル設計の専門家です。

以下の小規模会話コーパスを分析し、このコーパスが重視している会話スタイルを推定してください。
その上で、このコーパスらしい応答を大量の prompt/response データから抽出するためのベイズモデルJSONを作成してください。

このモデルの目的:
- 小規模コーパスに含まれる会話の特徴、応答戦略、暗黙の評価基準を推定する
- そのコーパスらしい response を高く評価する
- そのコーパスから外れる response を低く評価する
- 後続処理で posterior が高い応答を DPO の chosen、低い応答を rejected に使う

作業手順:
1. コーパス全体を読み、どのような目的・場面・会話スタイルのデータセットかを推定してください。
2. このコーパスで望ましいと考えられる応答戦略を抽出してください。
3. このコーパスらしくない応答傾向も想定してください。
4. 大量データ評価で使える観測ラベルを3〜8個作ってください。
5. 観測ラベルは、prompt と response の1組から判定できる粒度にしてください。
6. 各観測ラベルについて、target_style と non_target_style での出やすさを尤度として設定してください。
7. target_style 側では、推定したコーパスらしい応答戦略の尤度を高くしてください。
8. non_target_style 側では、推定したコーパスから外れる応答傾向の尤度を高くしてください。

重要:
- データセットの目的や会話スタイルは、事前知識で決めつけず、与えられたコーパスから推定してください。
- コーパスに明確に現れていない特徴を過度に追加しないでください。
- ただし、大量データ評価で negative 側を判定できるよう、コーパスらしくない応答傾向は合理的に補ってください。
- 観測ラベルは抽象的すぎず、後段の評価LLMが prompt/response から分類できるものにしてください。
- 下の出力JSON形式に含まれるラベル名、説明文、尤度値は構造を示すための例です。そのまま使わず、必ずコーパス分析に基づいて最適な observations, likelihoods, strategy_descriptions を設計してください。

出力制約:
- 出力はJSONオブジェクトのみです。
- Markdown、説明文、コードブロックは出力しないでください。
- 必須キーは name, positive_state, negative_state, observations, likelihoods, prior, strategy_descriptions です。
- positive_state は "target_style" にしてください。
- negative_state は "non_target_style" にしてください。
- prior は 0.5 にしてください。
- observations は英小文字、数字、アンダースコアのみの短いラベルにしてください。
- observations に重複を入れないでください。
- likelihoods.target_style と likelihoods.non_target_style は observations の全ラベルを必ず含めてください。
- 各尤度は 0.0 より大きく 1.0 より小さい数値にしてください。
- target_style 側の尤度合計は 1.0 にしてください。
- non_target_style 側の尤度合計も 1.0 にしてください。
- strategy_descriptions には、後段の評価LLMが分類できるように、各ラベルの意味を日本語で具体的に書いてください。

出力JSON形式:
{
  "name": "inferred_dialogue_style_model",
  "positive_state": "target_style",
  "negative_state": "non_target_style",
  "observations": [
    "style_aligned_response",
    "contextual_followup",
    "generic_response"
  ],
  "likelihoods": {
    "target_style": {
      "style_aligned_response": 0.45,
      "contextual_followup": 0.35,
      "generic_response": 0.20
    },
    "non_target_style": {
      "style_aligned_response": 0.15,
      "contextual_followup": 0.25,
      "generic_response": 0.60
    }
  },
  "prior": 0.5,
  "strategy_descriptions": {
    "style_aligned_response": "コーパスから推定される望ましい会話スタイルに沿って応答している。",
    "contextual_followup": "直前の発話内容を踏まえ、会話の文脈を自然に継続または深めている。",
    "generic_response": "直前の発話内容やコーパスの会話スタイルを十分に拾わず、一般的または定型的に返している。"
  }
}

以下が分析対象コーパスです。""".strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """LLM出力からJSONオブジェクトを抽出する。"""
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("LLM出力が空です。")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = JSON_OBJECT_PATTERN.search(stripped)
    if not match:
        raise ValueError("LLM出力からJSONオブジェクトを抽出できません。")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"抽出したJSONを解析できません: {exc}") from exc


def _get_response_attr(item: Any, name: str, default: Any = None) -> Any:
    """Responses APIのオブジェクト/辞書どちらからでも値を読む。"""
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _collect_response_text_from_output(output_items: Any) -> str:
    """Responses APIのoutput配列からテキスト本文を拾う。"""
    texts: list[str] = []
    for item in output_items or []:
        content_items = _get_response_attr(item, "content", []) or []
        for content in content_items:
            text = _get_response_attr(content, "text")
            if text:
                texts.append(str(text))
    return "\n".join(texts).strip()


def extract_response_text(response: Any) -> str:
    """Responses APIレスポンスから生成本文を取り出す。"""
    output_text = _get_response_attr(response, "output_text", "")
    if output_text:
        return str(output_text).strip()

    output_text = _collect_response_text_from_output(_get_response_attr(response, "output", []))
    if output_text:
        return output_text

    status = _get_response_attr(response, "status", "")
    incomplete_details = _get_response_attr(response, "incomplete_details", None)
    if status == "incomplete" or incomplete_details:
        reason = _get_response_attr(incomplete_details, "reason", incomplete_details)
        raise RuntimeError(
            "LLM出力が途中で打ち切られ、JSON本文が返りませんでした。"
            f"理由: {reason}。"
            "`--max-output-tokens` を増やすか、プロンプトを短くしてください。"
        )

    output_types = [
        str(_get_response_attr(item, "type", "unknown"))
        for item in (_get_response_attr(response, "output", []) or [])
    ]
    raise RuntimeError(
        "Responses APIの返答から本文を取り出せませんでした。"
        f"status={status or 'unknown'}, output_types={output_types}"
    )


class OpenAIResponsesGenerator:
    """OpenAIまたはAzure OpenAI Responses APIを呼び出す生成器。"""

    def __init__(self) -> None:
        try:
            from openai import AzureOpenAI, OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI API利用に必要な `openai` と `python-dotenv` をインストールしてください。") from exc
        load_env_file()
        self.azure_client_class = AzureOpenAI
        self.openai_client_class = OpenAI

    def generate(
        self,
        *,
        instructions: str,
        input_text: str,
        model: str,
        max_output_tokens: int,
        response_text_format: dict[str, Any] | None = None,
    ) -> str:
        """Responses APIでテキストを生成する。"""
        azure_endpoint = read_env_value("AZURE_OPENAI_ENDPOINT")
        azure_api_key = resolve_analysis_azure_api_key()
        azure_api_version = read_env_value("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        if azure_endpoint and azure_api_key:
            client = self.azure_client_class(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
            )
        else:
            client = self.openai_client_class(api_key=read_env_value("OPENAI_API_KEY"))
        create_kwargs: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": build_json_mode_input(input_text) if response_text_format else input_text,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": "high"},
        }
        if response_text_format:
            create_kwargs["text"] = {"format": response_text_format}
        response = client.responses.create(**create_kwargs)
        return extract_response_text(response)


def build_json_mode_input(input_text: str) -> str:
    """JSONモード要件を満たすため、input側にもjson語を含める。"""
    return "Return a valid JSON object only.\n\n" + input_text


def generate_bayes_model(
    records: list[dict[str, Any]],
    *,
    generator: TextGenerator,
    model: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    """小コーパスからベイズモデルJSONを生成する。"""
    output_text = generator.generate(
        instructions=build_analysis_instructions(),
        input_text=build_corpus_text(records),
        model=model,
        max_output_tokens=max_output_tokens,
        response_text_format={"type": "json_object"},
    )
    payload = extract_json_object(output_text)
    parse_bayes_model(payload)
    return payload


def write_json(payload: dict[str, Any], path: Path | str) -> None:
    """JSONを整形して書き出す。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    records = read_jsonl(args.input)
    summary = summarize_corpus(records)
    if args.dry_run:
        print("small corpus dry-run")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print(f"  model: {args.model}")
        return 0

    payload = generate_bayes_model(
        records,
        generator=OpenAIResponsesGenerator(),
        model=args.model,
        max_output_tokens=args.max_output_tokens,
    )
    write_json(payload, args.output)
    print(f"ベイズモデルJSONを書き出しました: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
