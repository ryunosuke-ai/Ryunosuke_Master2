"""Oracle正解応答を100点満点としてDPO前後モデルを評価する。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
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


DEFAULT_PROMPTS_PATH = "configs/evaluation_prompts/reminiscence_oracle_eval_v1.jsonl"
DEFAULT_SMALL_CORPUS_PATH = "data/small_corpus.jsonl"
DEFAULT_BAYES_MODEL_PATH = "artifacts/bayes_models/generated_transition_bayes_model.json"
DEFAULT_OUTPUT_DIR = "artifacts/evaluations/oracle_eval_runs/reminiscence_oracle_eval_v1"
DEFAULT_ORACLE_MAX_OUTPUT_TOKENS = 4096
PROMPT_TEMPLATE_VERSION = "oracle_eval.v1"
REFERENCE_TEMPLATE_VERSION = "oracle_reference_generation.v1"
JUDGE_TEMPLATE_VERSION = "oracle_score_against_reference.v1"


@dataclass(frozen=True)
class EvaluationPrompt:
    """Oracle評価用の1入力。"""

    prompt_id: str
    category: str
    prompt: str


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
    parser.add_argument("--max-prompts", type=int, default=None, help="評価prompt件数の上限。")
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


def read_evaluation_prompts(path: Path | str, *, max_prompts: int | None = None) -> list[EvaluationPrompt]:
    """評価prompt JSONLを検証して読み込む。"""
    prompts: list[EvaluationPrompt] = []
    seen_ids: set[str] = set()
    for line_number, record in enumerate(read_jsonl(path), start=1):
        prompt_id = str(record.get("id", "")).strip()
        category = str(record.get("category", "")).strip()
        prompt = str(record.get("prompt", "")).strip()
        if not prompt_id:
            raise ValueError(f"{line_number}行目の `id` が空です。")
        if prompt_id in seen_ids:
            raise ValueError(f"評価prompt idが重複しています: {prompt_id}")
        if not category:
            raise ValueError(f"{line_number}行目の `category` が空です。")
        if not prompt:
            raise ValueError(f"{line_number}行目の `prompt` が空です。")
        seen_ids.add(prompt_id)
        prompts.append(EvaluationPrompt(prompt_id=prompt_id, category=category, prompt=prompt))
    return prompts[:max_prompts] if max_prompts is not None else prompts


def load_small_corpus_context(path: Path | str, *, max_chars: int) -> str:
    """Oracleが参照する小コーパス本文を読み込む。"""
    records = read_small_corpus_jsonl(path)
    return build_corpus_text(records, max_chars=max_chars)


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


def build_reference_instructions(model: TransitionBayesModel, *, small_corpus_text: str = "") -> str:
    """Oracle正解応答生成の指示を作る。"""
    return (
        "あなたは会話評価実験のOracleです。"
        "与えられたユーザー発話に対して、以下の小コーパス由来スタイルを最もよく満たす理想的なAI応答を1つ作ってください。"
        "この応答は後で100点満点の正解応答として使われます。\n\n"
        "制約:\n"
        "- 出力はJSONのみです。\n"
        "- responseは日本語で1〜2文にしてください。\n"
        "- 相手が話し続けやすいように、発話内の具体語を拾い、必要な場合だけ質問を1つ添えてください。\n"
        "- 一般論、助言、長い説明、話題逸らし、過剰な推測は避けてください。\n"
        "- データセット名や評価用語を返答本文に出さないでください。\n\n"
        "必須キー: oracle_response, reason\n\n"
        f"{build_model_style_summary(model, small_corpus_text=small_corpus_text)}"
    )


def build_reference_input(prompt: EvaluationPrompt) -> str:
    """Oracle正解応答生成の入力を作る。"""
    return (
        "json output only.\n"
        f"prompt_id: {prompt.prompt_id}\n"
        f"category: {prompt.category}\n\n"
        f"user_prompt:\n{prompt.prompt}"
    )


def parse_reference_payload(payload: dict[str, Any]) -> dict[str, str]:
    """Oracle正解応答JSONを検証する。"""
    response = str(payload.get("oracle_response", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    if not response:
        raise ValueError("`oracle_response` が空です。")
    return {"oracle_response": response, "oracle_reason": reason}


def build_judge_instructions(model: TransitionBayesModel, *, small_corpus_text: str = "") -> str:
    """Oracle採点指示を作る。"""
    return (
        "あなたは会話評価実験のOracle採点者です。"
        "oracle_responseを100点満点の正解応答とみなし、response_aとresponse_bを個別に0〜100点で採点してください。"
        "モデル名は伏せられているため、どちらかを優遇せず、応答本文だけで評価してください。\n\n"
        "採点基準:\n"
        "- 100点: oracle_responseと同等に、小コーパス由来スタイルを満たす。\n"
        "- 80点: かなり良いが、具体性・感情・会話継続性のどれかが少し弱い。\n"
        "- 60点: 自然だが、文脈の拾い方や深め方が浅い。\n"
        "- 40点: 一般論、助言、話題逸らし、早い終結が目立つ。\n"
        "- 20点以下: 文脈不一致、不自然、会話を続けにくい。\n\n"
        "rubric_scoresは、context_understanding, concrete_pickup, experiential_deepening, emotion_and_scene, "
        "conversation_continuity, avoids_generic_advice, japanese_naturalness の各項目を0〜100点で出してください。"
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
    return (
        "json output only.\n"
        f"prompt_id: {prompt.prompt_id}\n"
        f"category: {prompt.category}\n\n"
        f"user_prompt:\n{prompt.prompt}\n\n"
        f"oracle_response_100_points:\n{oracle_response}\n\n"
        f"response_a:\n{response_a}\n\n"
        f"response_b:\n{response_b}"
    )


def _clamp_score(value: Any, *, key: str) -> float:
    """0〜100点の数値を検証して返す。"""
    if not isinstance(value, (int, float)):
        raise ValueError(f"`{key}` は数値である必要があります。")
    return max(0.0, min(100.0, float(value)))


def parse_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Oracle採点JSONを検証する。"""
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
) -> list[dict[str, Any]]:
    """base/DPO応答を同一prompt条件で生成する。"""
    bundle = load_compare_bundle(base_model_id, lora_path, use_4bit=use_4bit)
    records: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        print(f"[Oracle Eval] local generation {index}/{len(prompts)} {prompt.prompt_id}", flush=True)
        prompt_text = build_dpo_compare_prompt(prompt.prompt)
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
        records.append(
            {
                "prompt_id": prompt.prompt_id,
                "category": prompt.category,
                "prompt": prompt.prompt,
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
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                },
            }
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
    generator: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Oracle正解応答と採点結果を生成する。"""
    responses_with_oracle: list[dict[str, Any]] = []
    judgments: list[dict[str, Any]] = []
    reference_instructions = build_reference_instructions(bayes_model, small_corpus_text=small_corpus_text)
    judge_instructions = build_judge_instructions(bayes_model, small_corpus_text=small_corpus_text)
    prompt_lookup = {
        record["prompt_id"]: EvaluationPrompt(
            prompt_id=record["prompt_id"],
            category=record["category"],
            prompt=record["prompt"],
        )
        for record in response_records
    }
    for index, record in enumerate(response_records, start=1):
        prompt = prompt_lookup[record["prompt_id"]]
        print(f"[Oracle Eval] oracle reference {index}/{len(response_records)} {prompt.prompt_id}", flush=True)
        reference_text = generator.generate(
            instructions=reference_instructions,
            input_text=build_reference_input(prompt),
            model=oracle_model,
            max_output_tokens=max_output_tokens,
            response_text_format={"type": "json_object"},
        )
        reference = parse_reference_payload(extract_json_object(reference_text))
        first_label, second_label = model_order_for_prompt(prompt.prompt_id, seed=seed)
        response_by_label = {
            "base": record["base_response"],
            "dpo": record["dpo_response"],
        }
        response_a = response_by_label[first_label]
        response_b = response_by_label[second_label]
        print(f"[Oracle Eval] oracle judgment {index}/{len(response_records)} {prompt.prompt_id}", flush=True)
        judge_text = generator.generate(
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
        judgment = parse_judge_payload(extract_json_object(judge_text))
        score_by_label = {
            first_label: judgment["score_a"],
            second_label: judgment["score_b"],
        }
        if judgment["winner"] == "tie":
            winner_label = "tie"
        else:
            winner_label = first_label if judgment["winner"] == "response_a" else second_label
        responses_with_oracle.append(
            {
                **record,
                "oracle_response": reference["oracle_response"],
                "oracle_reason": reference["oracle_reason"],
                "oracle_model": oracle_model,
                "oracle_reference_template_version": REFERENCE_TEMPLATE_VERSION,
            }
        )
        judgments.append(
            {
                "prompt_id": prompt.prompt_id,
                "category": prompt.category,
                "prompt": prompt.prompt,
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
                "raw_winner": judgment["winner"],
                "rubric_scores": judgment["rubric_scores"],
                "reason": judgment["reason"],
                "oracle_model": oracle_model,
                "oracle_judge_template_version": JUDGE_TEMPLATE_VERSION,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            }
        )
    return responses_with_oracle, judgments


def summarize_judgments(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    """Oracle採点結果を集計する。"""
    if not judgments:
        raise ValueError("集計対象のjudgmentがありません。")
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


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    prompts = read_evaluation_prompts(args.prompts, max_prompts=args.max_prompts)
    bayes_model = load_transition_bayes_model(args.bayes_model)
    small_corpus_text = load_small_corpus_context(args.small_corpus, max_chars=args.small_corpus_max_chars)
    output_dir = Path(args.output_dir)
    responses_path = output_dir / "responses.jsonl"
    judgments_path = output_dir / "judgments.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    if args.dry_run:
        print("Oracle評価 dry-run")
        print(f"  prompts: {args.prompts} ({len(prompts)} 件)")
        print(f"  small_corpus: {args.small_corpus} ({len(small_corpus_text)} chars)")
        print(f"  bayes_model: {bayes_model.name}")
        print(f"  base_model_id: {args.base_model_id}")
        print(f"  lora_path: {args.lora_path}")
        print(f"  oracle_model: {args.oracle_model}")
        print(f"  output_dir: {output_dir}")
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
    )
    responses_with_oracle, judgments = run_oracle_judgment(
        response_records,
        bayes_model=bayes_model,
        small_corpus_text=small_corpus_text,
        oracle_model=args.oracle_model,
        max_output_tokens=args.oracle_max_output_tokens,
        seed=args.seed,
        generator=OpenAIResponsesGenerator(),
    )
    summary = summarize_judgments(judgments)
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
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "use_4bit": args.use_4bit,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "oracle_reference_template_version": REFERENCE_TEMPLATE_VERSION,
            "oracle_judge_template_version": JUDGE_TEMPLATE_VERSION,
        },
        manifest_path,
    )
    print(f"Oracle評価responsesを書き出しました: {responses_path}")
    print(f"Oracle評価judgmentsを書き出しました: {judgments_path}")
    print(f"Oracle評価summaryを書き出しました: {summary_path}")
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
