"""小さい会話コーパスをLLMで分析し、ベイズモデルJSONを生成する。"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol


DEFAULT_INPUT_PATH = "data/small_corpus.jsonl"
DEFAULT_OUTPUT_PATH = "artifacts/bayes_models/generated_bayes_model.json"
DEFAULT_MODEL = os.getenv("ANALYSIS_LLM_MODEL", "gpt-5.4-pro")
DEFAULT_MAX_OUTPUT_TOKENS = 4096
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class TextGenerator(Protocol):
    """テキスト生成器の最小インターフェース。"""

    def generate(self, *, instructions: str, input_text: str, model: str, max_output_tokens: int) -> str:
        """LLMからテキストを生成する。"""


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="小コーパスからベイズモデルJSONを生成します。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help=f"入力JSONL（既定: {DEFAULT_INPUT_PATH}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"出力JSON（既定: {DEFAULT_OUTPUT_PATH}）。")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"分析LLMモデル（既定: {DEFAULT_MODEL}）。")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="最大出力トークン数。")
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、入力概要だけ表示します。")
    return parser.parse_args()


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
    return (
        "あなたは会話分析とベイズモデリングの研究支援者です。"
        "小さい会話コーパスから、その会話らしさを表す観測ラベル、会話戦略、"
        "positive_state/negative_stateの尤度を設計してください。"
        "出力はJSONオブジェクトのみです。"
        "必須キーは name, positive_state, negative_state, observations, likelihoods, prior, "
        "strategy_descriptions です。"
        "observations は3〜8個、likelihoods は各状態ごとに observations 全要素を含め、"
        "各状態の合計が概ね1.0になるようにしてください。"
        "prior は0より大きく1より小さい数値にしてください。"
    )


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


class OpenAIResponsesGenerator:
    """OpenAIまたはAzure OpenAI Responses APIを呼び出す生成器。"""

    def __init__(self) -> None:
        try:
            from dotenv import load_dotenv
            from openai import AzureOpenAI, OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI API利用に必要な `openai` と `python-dotenv` をインストールしてください。") from exc
        load_dotenv()
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview").strip()
        if azure_endpoint and azure_api_key:
            self.client = AzureOpenAI(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
            )
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, *, instructions: str, input_text: str, model: str, max_output_tokens: int) -> str:
        """Responses APIでテキストを生成する。"""
        response = self.client.responses.create(
            model=model,
            instructions=instructions,
            input=input_text,
            max_output_tokens=max_output_tokens,
            reasoning={"effort": "high"},
        )
        return (response.output_text or "").strip()


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
    )
    return extract_json_object(output_text)


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
