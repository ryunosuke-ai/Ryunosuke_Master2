"""生成ベイズモデルを使い、大きな対話データの各応答を評価する。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.generated_bayes_model import BayesModel, ObservationScore, load_bayes_model, score_observation


DEFAULT_INPUT_PATH = "data/large_dialogue.jsonl"
DEFAULT_MODEL_PATH = "artifacts/bayes_models/generated_bayes_model.json"
DEFAULT_OUTPUT_PATH = "artifacts/scored_dialogues/bayes_scored_dialogue.jsonl"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_MAX_OUTPUT_TOKENS = 1024
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
    parser = argparse.ArgumentParser(description="対話データを生成ベイズモデルでスコアリングします。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help=f"入力JSONL/CSV（既定: {DEFAULT_INPUT_PATH}）。")
    parser.add_argument("--bayes-model", default=DEFAULT_MODEL_PATH, help=f"ベイズモデルJSON（既定: {DEFAULT_MODEL_PATH}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"出力JSONL（既定: {DEFAULT_OUTPUT_PATH}）。")
    default_model = resolve_scoring_model()
    parser.add_argument("--model", default=default_model, help=f"評価LLMモデルまたはAzure deployment名（既定: {default_model}）。")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="最大出力トークン数。")
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、入力件数だけ確認します。")
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


def resolve_scoring_model() -> str:
    """評価用モデルまたはAzure deployment名を解決する。"""
    return read_env_value_with_fallback(
        "SCORING_LLM_MODEL",
        "AZURE_OPENAI_GPT54_DEPLOYMENT_NAME",
        "OPENAI_GPT54_MODEL",
        default=DEFAULT_MODEL,
    )


def resolve_scoring_azure_api_key(model: str = "") -> str:
    """評価用Azure OpenAI APIキーを解決する。"""
    if model.startswith("gpt-5.6") or model in {
        read_env_value("AZURE_OPENAI_GPT56_SOL_DEPLOYMENT"),
        read_env_value("AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT"),
    }:
        return read_env_value_with_fallback(
            "AZURE_OPENAI_GPT56_API_KEY",
            "AZURE_OPENAI_API_KEY",
        )
    return read_env_value_with_fallback(
        "AZURE_OPENAI_GPT54_API_KEY",
        "OPENAI_GPT54_API_KEY",
        "AZURE_OPENAI_API_KEY",
    )


def resolve_scoring_azure_endpoint(model: str) -> str:
    """モデル世代に対応するAzure endpointを解決する。"""
    sol = read_env_value("AZURE_OPENAI_GPT56_SOL_DEPLOYMENT")
    terra = read_env_value("AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT")
    if model in {sol, terra} or model.startswith("gpt-5.6"):
        return read_env_value_with_fallback("AZURE_OPENAI_GPT56_ENDPOINT", "AZURE_OPENAI_ENDPOINT")
    return read_env_value("AZURE_OPENAI_ENDPOINT")


def resolve_scoring_azure_api_version(model: str) -> str:
    """モデル世代に対応するAzure API versionを解決する。"""
    if model.startswith("gpt-5.6") or model in {
        read_env_value("AZURE_OPENAI_GPT56_SOL_DEPLOYMENT"),
        read_env_value("AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT"),
    }:
        return read_env_value_with_fallback(
            "AZURE_OPENAI_GPT56_API_VERSION",
            "AZURE_OPENAI_API_VERSION",
            default="2025-04-01-preview",
        )
    return read_env_value("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")


def read_dialogue_records(path: Path | str) -> list[dict[str, Any]]:
    """JSONLまたはCSVの対話データを読み込む。"""
    input_path = Path(path)
    if input_path.suffix.lower() == ".csv":
        with input_path.open("r", newline="", encoding="utf-8") as file:
            return [validate_dialogue_record(dict(row), line_number=index) for index, row in enumerate(csv.DictReader(file), start=2)]

    records: list[dict[str, Any]] = []
    try:
        with input_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{line_number}行目をJSONとして読めません: {exc}") from exc
                records.append(validate_dialogue_record(payload, line_number=line_number))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"対話データが見つかりません: {input_path}") from exc
    if not records:
        raise ValueError("対話データに有効なレコードがありません。")
    return records


def validate_dialogue_record(record: dict[str, Any], *, line_number: int) -> dict[str, Any]:
    """評価対象レコードの必須列を検証する。"""
    for key in ("conversation_id", "turn_index", "prompt", "response"):
        if key not in record:
            raise ValueError(f"{line_number}行目に `{key}` がありません。")
        if not str(record[key]).strip():
            raise ValueError(f"{line_number}行目の `{key}` が空です。")
    return record


def build_scoring_instructions(model: BayesModel) -> str:
    """観測ラベル判定用のLLM指示を作る。"""
    observation_lines = "\n".join(f"- {name}" for name in model.observations)
    strategy_lines = "\n".join(f"- {key}: {value}" for key, value in model.strategy_descriptions.items())
    return (
        "あなたは会話データの評価者です。"
        "promptに対するresponseが、指定された観測ラベルのどれに最も近いかを1つ選びます。"
        "出力はJSONのみで、observation, score, reason を含めてください。"
        "scoreは0.0〜1.0で、その観測ラベルらしさの強さです。\n\n"
        f"観測ラベル:\n{observation_lines}\n\n"
        f"戦略説明:\n{strategy_lines}"
    )


def build_scoring_input(record: dict[str, Any]) -> str:
    """LLM評価用の入力を作る。"""
    return (
        f"conversation_id: {record['conversation_id']}\n"
        f"turn_index: {record['turn_index']}\n"
        f"prompt:\n{record['prompt']}\n\n"
        f"response:\n{record['response']}"
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


def parse_observation_score(payload: dict[str, Any], model: BayesModel) -> ObservationScore:
    """LLM出力を観測評価へ変換する。"""
    observation = str(payload.get("observation", "")).strip()
    if observation not in model.observations:
        raise ValueError(f"未知の観測ラベルです: {observation}")
    raw_score = payload.get("score", 0.0)
    if not isinstance(raw_score, (int, float)):
        raise ValueError("`score` は数値である必要があります。")
    score = max(0.0, min(1.0, float(raw_score)))
    reason = str(payload.get("reason", "")).strip()
    return ObservationScore(observation=observation, score=score, reason=reason)


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
        azure_endpoint = resolve_scoring_azure_endpoint(model)
        azure_api_key = resolve_scoring_azure_api_key(model)
        azure_api_version = resolve_scoring_azure_api_version(model)
        if azure_endpoint and azure_api_key:
            client = self.azure_client_class(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
                max_retries=2,
            )
        else:
            client = self.openai_client_class(
                api_key=read_env_value("OPENAI_API_KEY"), max_retries=2
            )
        create_kwargs: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": build_json_mode_input(input_text) if response_text_format else input_text,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": "medium"},
        }
        if response_text_format:
            create_kwargs["text"] = {"format": response_text_format}
        response = client.responses.create(**create_kwargs)
        return extract_response_text(response)


def build_json_mode_input(input_text: str) -> str:
    """JSONモード要件を満たすため、input側にもjson語を含める。"""
    return "Return a valid JSON object only.\n\n" + input_text


def score_records(
    records: list[dict[str, Any]],
    *,
    bayes_model: BayesModel,
    generator: TextGenerator,
    model: str,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    """対話レコード群をスコアリングする。"""
    results: list[dict[str, Any]] = []
    prior_by_conversation: dict[str, float] = {}
    instructions = build_scoring_instructions(bayes_model)
    sorted_records = sorted(records, key=lambda item: (str(item["conversation_id"]), int(item["turn_index"])))
    for record in sorted_records:
        conversation_id = str(record["conversation_id"])
        prior = prior_by_conversation.get(conversation_id, bayes_model.prior)
        output_text = generator.generate(
            instructions=instructions,
            input_text=build_scoring_input(record),
            model=model,
            max_output_tokens=max_output_tokens,
            response_text_format={"type": "json_object"},
        )
        observation_score = parse_observation_score(extract_json_object(output_text), bayes_model)
        bayes_result = score_observation(bayes_model, observation_score, prior=prior)
        prior_by_conversation[conversation_id] = float(bayes_result["posterior"])
        results.append({**record, **bayes_result})
    return results


def write_jsonl(records: list[dict[str, Any]], path: Path | str) -> None:
    """JSONLを書き出す。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    records = read_dialogue_records(args.input)
    bayes_model = load_bayes_model(args.bayes_model)
    if args.dry_run:
        print("bayes scoring dry-run")
        print(f"  records: {len(records)}")
        print(f"  bayes_model: {bayes_model.name}")
        print(f"  model: {args.model}")
        return 0
    scored = score_records(
        records,
        bayes_model=bayes_model,
        generator=OpenAIResponsesGenerator(),
        model=args.model,
        max_output_tokens=args.max_output_tokens,
    )
    write_jsonl(scored, args.output)
    print(f"スコア済みJSONLを書き出しました: {args.output} ({len(scored)} 件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
