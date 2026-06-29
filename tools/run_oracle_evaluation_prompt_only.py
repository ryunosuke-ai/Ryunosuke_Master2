"""few-shot prompt制御だけのQwenをOracle評価する。"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.dpo_compare_text_chat import (  # noqa: E402
    DEFAULT_ENV_MAX_MEMORY,
    cleanup_generated_text,
    parse_max_memory_env,
    suppress_external_warnings,
)
from apps.dpo_text_chat import (  # noqa: E402
    ChatBundle,
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    build_dpo_generation_prompt,
    load_tokenizer,
    load_training_modules,
)
from core.transition_bayes_model import load_transition_bayes_model  # noqa: E402
from tools.analyze_small_corpus import OpenAIResponsesGenerator  # noqa: E402
from tools.run_oracle_evaluation import (  # noqa: E402
    DEFAULT_ORACLE_MAX_OUTPUT_TOKENS,
    ESCONV_CORE_WEIGHTS,
    ESCONV_STRATEGY_V3_AXIS_KEYS,
    ESCONV_STRATEGY_V3_PRESET,
    WEIGHTED_ESCONV_OVERALL_WEIGHTS,
    WIN_TIE_THRESHOLD,
    EvaluationPrompt,
    append_jsonl_record,
    build_model_style_summary,
    build_reference_input,
    build_reference_instructions,
    format_prompt_context,
    load_small_corpus_context,
    parse_axis_focus,
    parse_category_filter,
    parse_prompt_history,
    parse_v3_axis_scores,
    read_evaluation_prompts,
    read_jsonl,
    read_jsonl_lenient,
    records_by_sample_key,
    reference_template_version,
    retry_config_from_env,
    run_with_retry,
    sample_key,
    weighted_score,
    write_json,
    write_jsonl,
)
from tools.score_dialogue_with_bayes_model import (  # noqa: E402
    extract_json_object,
    load_env_file,
    resolve_scoring_model,
)


DEFAULT_PROMPTS_PATH = "configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl"
DEFAULT_SMALL_CORPUS_PATH = "data/esconv_analysis_corpus_reminiscence_5000_to_2000.jsonl"
DEFAULT_BAYES_MODEL_PATH = (
    "artifacts/bayes_models/generated_transition_bayes_model_esconv_reminiscence_5000_to_2000.json"
)
DEFAULT_FEWSHOT_PATH = "artifacts/datasets/esconv_gold_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl"
DEFAULT_OUTPUT_DIR = (
    "artifacts/evaluations/oracle_eval_runs/"
    "esconv_prompt_only_fewshot_oracle_esconv_v3_strategy_gpt54"
)
DEFAULT_BASELINE_SUMMARY_PATH = (
    "artifacts/evaluations/oracle_eval_runs/"
    "reminiscence_5000_to_2000_oracle_esconv_v3_strategy_gpt54/summary.json"
)
DEFAULT_BASELINE_SUMMARY_FALLBACK_PATH = (
    "docs/results/oracle_eval_runs/"
    "reminiscence_5000_to_2000_oracle_esconv_v3_strategy_gpt54/summary.json"
)
DEFAULT_FEWSHOT_COUNT = 8
PROMPT_ONLY_TEMPLATE_VERSION = "prompt_only_fewshot.esconv_strategy.v1"
SINGLE_JUDGE_TEMPLATE_VERSION = "oracle_score_single_response.esconv_strategy.v1"


@dataclass(frozen=True)
class FewShotExample:
    """prompt-only制御に入れるESConv few-shot例。"""

    example_id: str
    source_strategy: str
    prompt: str
    chosen: str


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    load_env_file()
    default_oracle_model = resolve_scoring_model()
    parser = argparse.ArgumentParser(description="prompt-only few-shot QwenをESConv Oracle v3で単独評価します。")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--small-corpus", default=DEFAULT_SMALL_CORPUS_PATH)
    parser.add_argument("--bayes-model", default=DEFAULT_BAYES_MODEL_PATH)
    parser.add_argument("--fewshot-examples", default=DEFAULT_FEWSHOT_PATH)
    parser.add_argument("--fewshot-count", type=int, default=DEFAULT_FEWSHOT_COUNT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--baseline-summary",
        default=DEFAULT_BASELINE_SUMMARY_PATH,
        help="比較に使う既存Bayes-DPO summary。存在しない場合は比較を省略します。",
    )
    parser.add_argument(
        "--base-model-id",
        default=os.getenv("LOCAL_QWEN_MODEL_ID", DEFAULT_BASE_MODEL_ID).strip() or DEFAULT_BASE_MODEL_ID,
    )
    parser.add_argument("--oracle-model", default=default_oracle_model)
    parser.add_argument("--oracle-workers", type=int, default=1)
    parser.add_argument("--style-preset", choices=(ESCONV_STRATEGY_V3_PRESET,), default=ESCONV_STRATEGY_V3_PRESET)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--skip-prompts", type=int, default=0)
    parser.add_argument("--categories", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--oracle-max-output-tokens", type=int, default=DEFAULT_ORACLE_MAX_OUTPUT_TOKENS)
    parser.add_argument("--small-corpus-max-chars", type=int, default=20000)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_fewshot_examples(path: Path | str) -> list[FewShotExample]:
    """ESConv gold DPO JSONLからfew-shot正例だけを読む。"""
    examples: list[FewShotExample] = []
    for line_number, record in enumerate(read_jsonl(path), start=1):
        prompt = str(record.get("prompt", "")).strip()
        chosen = str(record.get("chosen", "")).strip()
        if not prompt or not chosen:
            raise ValueError(f"{path}:{line_number} の `prompt` または `chosen` が空です。")
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        source_strategy = str(
            metadata.get("source_strategy") or metadata.get("strategy") or "unknown"
        ).strip() or "unknown"
        source_dialogue_id = str(record.get("source_dialogue_id") or record.get("conversation_id") or "unknown")
        turn_index = str(record.get("turn_index") or line_number)
        examples.append(
            FewShotExample(
                example_id=f"{source_dialogue_id}:{turn_index}",
                source_strategy=source_strategy,
                prompt=prompt,
                chosen=chosen,
            )
        )
    if not examples:
        raise ValueError(f"few-shot例がありません: {path}")
    return examples


def select_balanced_fewshot_examples(
    examples: list[FewShotExample],
    *,
    count: int,
    seed: int,
) -> list[FewShotExample]:
    """source_strategyが偏りにくいようfew-shot例をseed固定で選ぶ。"""
    if count <= 0:
        return []
    grouped: dict[str, list[FewShotExample]] = defaultdict(list)
    for example in examples:
        grouped[example.source_strategy].append(example)

    rng = random.Random(seed)
    strategy_keys = sorted(grouped)
    for key in strategy_keys:
        rng.shuffle(grouped[key])
    rng.shuffle(strategy_keys)

    selected: list[FewShotExample] = []
    offsets = {key: 0 for key in strategy_keys}
    limit = min(count, len(examples))
    while len(selected) < limit:
        changed = False
        for key in strategy_keys:
            offset = offsets[key]
            bucket = grouped[key]
            if offset >= len(bucket):
                continue
            selected.append(bucket[offset])
            offsets[key] += 1
            changed = True
            if len(selected) >= limit:
                break
        if not changed:
            break
    return selected


def extract_generation_context(prompt_text: str) -> str:
    """DPO promptからfew-shot例に見せる会話部分だけを取り出す。"""
    marker = "これまでの会話:"
    text = str(prompt_text).strip()
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    return text


def format_fewshot_example(example: FewShotExample, *, index: int) -> str:
    """few-shot例1件をprompt本文へ整形する。"""
    return (
        f"### 例{index}\n"
        f"会話:\n{extract_generation_context(example.prompt)}\n\n"
        f"良いAI返答:\n{example.chosen}"
    )


def build_prompt_only_fewshot_prompt(
    prompt: EvaluationPrompt,
    fewshot_examples: list[FewShotExample],
) -> str:
    """ESConv few-shotだけでbase Qwenを制御するpromptを作る。"""
    example_text = "\n\n".join(
        format_fewshot_example(example, index=index)
        for index, example in enumerate(fewshot_examples, start=1)
    )
    context_lines = [f"{turn['speaker']}: {turn['text']}" for turn in prompt.history]
    context_lines.append(f"User: {prompt.prompt}")
    context_lines.append("AI:")
    target_context = "\n".join(context_lines)
    return (
        "以下の支援対話の例を参考に、最後の会話に続くAI返答を生成してください。\n"
        "返答は日本語で1〜2文にしてください。\n"
        "相手の感情や具体的な状況を先に受け止め、必要な場合だけ短い確認質問や小さな次の一歩を添えてください。\n"
        "早すぎる助言、断定、一般論、長い説明、評価用語やデータセット名の言及は避けてください。\n\n"
        "参考例:\n"
        f"{example_text}\n\n"
        "生成対象:\n"
        f"{target_context}"
    )


def load_base_qwen_bundle(base_model_id: str, *, use_4bit: bool) -> ChatBundle:
    """LoRAなしのbase Qwenを読み込む。"""
    suppress_external_warnings()
    deps = load_training_modules()
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応GPUが見つかりません。Qwen3.5-27B の評価にはGPU環境が必要です。")

    tokenizer = load_tokenizer(base_model_id, deps)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    max_memory = parse_max_memory_env(os.getenv(DEFAULT_ENV_MAX_MEMORY, ""))
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "4bit量子化に必要な bitsandbytes が見つかりません。"
                " `--use-4bit` を外すか、依存関係を見直してください。"
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        model_kwargs["torch_dtype"] = dtype

    model = deps["ModelClass"].from_pretrained(base_model_id, **model_kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = True
    model.eval()
    return ChatBundle(tokenizer=tokenizer, model=model, torch=torch)


def generate_prompt_only_reply(
    bundle: ChatBundle,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> str:
    """few-shot promptだけでbase Qwen応答を生成する。"""
    generation_prompt = build_dpo_generation_prompt(bundle.tokenizer, prompt_text)
    bundle.torch.manual_seed(seed)
    if bundle.torch.cuda.is_available():
        bundle.torch.cuda.manual_seed_all(seed)

    model_inputs = bundle.tokenizer(generation_prompt, return_tensors="pt")
    model_inputs = {
        key: value.to(bundle.input_device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }
    input_ids = model_inputs["input_ids"]
    if "attention_mask" not in model_inputs:
        model_inputs["attention_mask"] = bundle.torch.ones_like(input_ids, device=bundle.input_device)

    with bundle.torch.no_grad():
        output_ids = bundle.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=bundle.tokenizer.eos_token_id,
            pad_token_id=bundle.tokenizer.eos_token_id,
        )

    generated = output_ids[0][input_ids.shape[1]:]
    decoded = bundle.tokenizer.decode(generated, skip_special_tokens=False).strip()
    return cleanup_generated_text(decoded, generation_prompt)


def generate_prompt_only_responses(
    prompts: list[EvaluationPrompt],
    *,
    base_model_id: str,
    fewshot_examples: list[FewShotExample],
    fewshot_examples_path: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    use_4bit: bool,
    existing_response_records: list[dict[str, Any]] | None = None,
    responses_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """prompt-only応答を生成する。"""
    existing_by_key = records_by_sample_key(existing_response_records or [])
    if existing_by_key:
        print(f"[Prompt-only Oracle] found existing responses: {len(existing_by_key)}", flush=True)
    missing_prompts = [prompt for prompt in prompts if prompt.prompt_id not in existing_by_key]
    bundle = None
    if missing_prompts:
        bundle = load_base_qwen_bundle(base_model_id, use_4bit=use_4bit)

    records: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        if prompt.prompt_id in existing_by_key:
            print(f"[Prompt-only Oracle] skip local generation {index}/{len(prompts)} {prompt.prompt_id}", flush=True)
            records.append(existing_by_key[prompt.prompt_id])
            continue
        print(f"[Prompt-only Oracle] local generation {index}/{len(prompts)} {prompt.prompt_id}", flush=True)
        if bundle is None:
            raise RuntimeError("prompt-only generation bundleが初期化されていません。")
        prompt_text = build_prompt_only_fewshot_prompt(prompt, fewshot_examples)
        response = generate_prompt_only_reply(
            bundle,
            prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        record = {
            "prompt_id": prompt.prompt_id,
            "category": prompt.category,
            "prompt": prompt.prompt,
            "history": [dict(turn) for turn in prompt.history],
            "axis_focus": list(prompt.axis_focus),
            "model_prompt": prompt_text,
            "prompt_only_response": response,
            "comparison_kind": "prompt_only_fewshot",
            "fewshot_example_ids": [example.example_id for example in fewshot_examples],
            "fewshot_source_strategies": [example.source_strategy for example in fewshot_examples],
            "generation": {
                "comparison_kind": "prompt_only_fewshot",
                "base_model_id": base_model_id,
                "fewshot_examples": fewshot_examples_path,
                "fewshot_count": len(fewshot_examples),
                "fewshot_example_ids": [example.example_id for example in fewshot_examples],
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "seed": seed,
                "use_4bit": use_4bit,
                "thinking": "disabled",
                "prompt_template_version": PROMPT_ONLY_TEMPLATE_VERSION,
            },
        }
        if responses_path is not None:
            append_jsonl_record(record, responses_path)
        records.append(record)
    return records


def single_judge_rubric_text() -> str:
    """prompt-only単独採点のESConv v3 rubricを返す。"""
    axis_keys = ", ".join(ESCONV_STRATEGY_V3_AXIS_KEYS)
    return (
        "採点基準:\n"
        "- prompt_only_responseを、ESConvらしさを測る複数軸で0〜100点評価してください。\n"
        "- oracle_responseは高品質な参照例ですが、唯一の正解文ではありません。"
        "prompt_only_responseを文脈と評価軸に照らして独立に評価してください。\n"
        "- ESConvらしさの主要軸は、ESConv strategy adherence、Emotional reflection / validation、"
        "Avoidance of premature adviceです。\n"
        "- 確認質問、問題探索、情報提供、次の一歩は Conversational progression として別軸で評価し、"
        "質問があるだけで主要軸やweighted評価を過大評価しないでください。\n"
        "- 感情の具体的な反映、受容、相談者の自己否定や不安を急いで直そうとしない姿勢を高く評価してください。\n"
        "- 早い助言、断定、一般論、説教、ラベル付け、文脈から外れた情報提供は、"
        "premature_advice_avoidance と strategy_adherence を下げてください。\n"
        "- responseが温かく自然でも、ユーザーの具体語や感情語に沿っていない場合は contextual_grounding を下げてください。\n\n"
        "出力JSONは次の形にしてください:\n"
        "{\n"
        "  \"scores\": {各評価軸: 0〜100},\n"
        "  \"reason\": \"短い理由\"\n"
        "}\n"
        f"必須の評価軸: {axis_keys}"
    )


def build_single_judge_instructions(
    model: Any,
    *,
    small_corpus_text: str = "",
) -> str:
    """prompt-only単独採点のOracle指示を作る。"""
    return (
        "あなたは会話評価実験のOracle採点者です。"
        "prompt_only_responseを、ESConvらしさを測る複数軸で個別に0〜100点評価してください。"
        "モデル名や条件名は採点に使わず、応答本文だけで評価してください。\n\n"
        f"{single_judge_rubric_text()}\n"
        "出力はJSONのみです。必須キーは scores, reason です。\n\n"
        f"{build_model_style_summary(model, small_corpus_text=small_corpus_text)}"
    )


def build_single_judge_input(
    *,
    prompt: EvaluationPrompt,
    oracle_response: str,
    prompt_only_response: str,
) -> str:
    """prompt-only単独採点のOracle入力を作る。"""
    axis_focus_section = ""
    if prompt.axis_focus:
        axis_focus_section = "\naxis_focus:\n" + "\n".join(f"- {item}" for item in prompt.axis_focus) + "\n"
    return (
        "json output only.\n"
        f"prompt_id: {prompt.prompt_id}\n"
        f"category: {prompt.category}\n\n"
        f"{axis_focus_section}"
        f"{format_prompt_context(prompt)}\n\n"
        f"oracle_response_100_points:\n{oracle_response}\n\n"
        f"prompt_only_response:\n{prompt_only_response}"
    )


def parse_single_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """単独採点JSONを検証し、集計用scoreを計算する。"""
    axis_scores = parse_v3_axis_scores(payload.get("scores"), key="scores")
    return {
        "axis_scores": axis_scores,
        "esconv_core_score": weighted_score(axis_scores, ESCONV_CORE_WEIGHTS),
        "weighted_esconv_overall_score": weighted_score(axis_scores, WEIGHTED_ESCONV_OVERALL_WEIGHTS),
        "reason": str(payload.get("reason", "")).strip(),
    }


def run_prompt_only_oracle_judgment(
    response_records: list[dict[str, Any]],
    *,
    bayes_model: Any,
    small_corpus_text: str,
    oracle_model: str,
    max_output_tokens: int,
    generator: Any,
    oracle_workers: int = 1,
    existing_judgment_records: list[dict[str, Any]] | None = None,
    judgments_path: Path | str | None = None,
    responses_path: Path | str | None = None,
    failures_path: Path | str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Oracle参照応答とprompt-only単独採点を生成する。"""
    retry_config = retry_config_from_env()
    existing_judgment_by_key = records_by_sample_key(existing_judgment_records or [])
    if existing_judgment_by_key:
        print(f"[Prompt-only Oracle] found existing judgments: {len(existing_judgment_by_key)}", flush=True)
    reference_instructions = build_reference_instructions(
        bayes_model,
        small_corpus_text=small_corpus_text,
        style_preset=ESCONV_STRATEGY_V3_PRESET,
    )
    judge_instructions = build_single_judge_instructions(
        bayes_model,
        small_corpus_text=small_corpus_text,
    )
    prompt_lookup = {
        record["prompt_id"]: EvaluationPrompt(
            prompt_id=record["prompt_id"],
            category=record["category"],
            prompt=record["prompt"],
            history=parse_prompt_history(record.get("history", []), line_number=0),
            axis_focus=parse_axis_focus(record.get("axis_focus", []), line_number=0),
        )
        for record in response_records
    }

    def judge_one(index: int, record: dict[str, Any]) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        prompt = prompt_lookup[record["prompt_id"]]
        stage = "reference"
        try:
            if record.get("oracle_response"):
                reference = {
                    "oracle_response": str(record["oracle_response"]),
                    "oracle_reason": str(record.get("oracle_reason", "")),
                }
            else:
                print(f"[Prompt-only Oracle] oracle reference {index}/{len(response_records)} {prompt.prompt_id}", flush=True)
                reference = run_with_retry(
                    lambda: _parse_reference_for_prompt_only(
                        extract_json_object(
                            generator.generate(
                                instructions=reference_instructions,
                                input_text=build_reference_input(prompt),
                                model=oracle_model,
                                max_output_tokens=max_output_tokens,
                                response_text_format={"type": "json_object"},
                            )
                        )
                    ),
                    prompt_id=prompt.prompt_id,
                    stage=stage,
                    retry_config=retry_config,
                )
            stage = "judgment"
            print(f"[Prompt-only Oracle] oracle judgment {index}/{len(response_records)} {prompt.prompt_id}", flush=True)
            judgment = run_with_retry(
                lambda: parse_single_judge_payload(
                    extract_json_object(
                        generator.generate(
                            instructions=judge_instructions,
                            input_text=build_single_judge_input(
                                prompt=prompt,
                                oracle_response=reference["oracle_response"],
                                prompt_only_response=str(record["prompt_only_response"]),
                            ),
                            model=oracle_model,
                            max_output_tokens=max_output_tokens,
                            response_text_format={"type": "json_object"},
                        )
                    )
                ),
                prompt_id=prompt.prompt_id,
                stage=stage,
                retry_config=retry_config,
            )
        except Exception as exc:
            return index, None, None, {
                "sample_id": prompt.prompt_id,
                "prompt_id": prompt.prompt_id,
                "status": "failed",
                "stage": stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "attempts": retry_config.max_retries + 1,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }

        response_with_oracle = {
            **record,
            "oracle_response": reference["oracle_response"],
            "oracle_reason": reference["oracle_reason"],
            "oracle_model": oracle_model,
            "oracle_reference_template_version": reference_template_version(ESCONV_STRATEGY_V3_PRESET),
            "style_preset": ESCONV_STRATEGY_V3_PRESET,
        }
        judgment_record = {
            "prompt_id": prompt.prompt_id,
            "category": prompt.category,
            "prompt": prompt.prompt,
            "history": [dict(turn) for turn in prompt.history],
            "axis_focus": list(prompt.axis_focus),
            "oracle_response": reference["oracle_response"],
            "prompt_only_response": record["prompt_only_response"],
            "axis_scores": judgment["axis_scores"],
            "esconv_core_score": judgment["esconv_core_score"],
            "weighted_esconv_overall_score": judgment["weighted_esconv_overall_score"],
            "score_prompt_only": judgment["weighted_esconv_overall_score"],
            "reason": judgment["reason"],
            "oracle_model": oracle_model,
            "oracle_judge_template_version": SINGLE_JUDGE_TEMPLATE_VERSION,
            "oracle_reference_template_version": reference_template_version(ESCONV_STRATEGY_V3_PRESET),
            "prompt_template_version": PROMPT_ONLY_TEMPLATE_VERSION,
            "style_preset": ESCONV_STRATEGY_V3_PRESET,
            "comparison_kind": "prompt_only_fewshot",
        }
        return index, response_with_oracle, judgment_record, None

    completed_results: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    pending_records: list[tuple[int, dict[str, Any]]] = []
    for index, record in enumerate(response_records, start=1):
        key = sample_key(record)
        if key in existing_judgment_by_key:
            print(f"[Prompt-only Oracle] skip oracle judgment {index}/{len(response_records)} {key}", flush=True)
            judgment = existing_judgment_by_key[key]
            response_with_oracle = dict(record)
            if "oracle_response" not in response_with_oracle and "oracle_response" in judgment:
                response_with_oracle.update(
                    {
                        "oracle_response": judgment["oracle_response"],
                        "oracle_model": oracle_model,
                        "oracle_reference_template_version": reference_template_version(ESCONV_STRATEGY_V3_PRESET),
                        "style_preset": ESCONV_STRATEGY_V3_PRESET,
                    }
                )
            completed_results.append((index, response_with_oracle, judgment))
        else:
            pending_records.append((index, record))

    def record_success(index: int, response_with_oracle: dict[str, Any], judgment: dict[str, Any]) -> None:
        completed_results.append((index, response_with_oracle, judgment))
        if judgments_path is not None:
            append_jsonl_record(judgment, judgments_path)
        print(
            f"[Prompt-only Oracle] completed {len(completed_results)}/{len(response_records)} "
            f"{judgment['prompt_id']} core={judgment['esconv_core_score']:.1f} "
            f"overall={judgment['weighted_esconv_overall_score']:.1f}",
            flush=True,
        )

    def record_failure(failure: dict[str, Any]) -> None:
        if failures_path is not None:
            append_jsonl_record(failure, failures_path)
        print(
            f"[Prompt-only Oracle] failed {failure['stage']} {failure['prompt_id']} "
            f"{failure['error_type']}: {failure['error_message']}",
            flush=True,
        )

    if oracle_workers <= 1:
        for index, record in pending_records:
            _, response_with_oracle, judgment, failure = judge_one(index, record)
            if failure is not None:
                record_failure(failure)
                continue
            if response_with_oracle is not None and judgment is not None:
                record_success(index, response_with_oracle, judgment)
    else:
        print(f"[Prompt-only Oracle] oracle parallel workers={oracle_workers}", flush=True)
        with ThreadPoolExecutor(max_workers=oracle_workers) as executor:
            futures = {executor.submit(judge_one, index, record): index for index, record in pending_records}
            for future in as_completed(futures):
                index, response_with_oracle, judgment, failure = future.result()
                if failure is not None:
                    record_failure(failure)
                    continue
                if response_with_oracle is not None and judgment is not None:
                    record_success(index, response_with_oracle, judgment)

    ordered_results = sorted(completed_results, key=lambda item: item[0])
    responses_with_oracle = [response for _, response, _ in ordered_results]
    judgments = [judgment for _, _, judgment in ordered_results]
    if responses_path is not None and len(judgments) == len(response_records):
        write_jsonl(responses_with_oracle, responses_path)
    return responses_with_oracle, judgments


def _parse_reference_for_prompt_only(payload: dict[str, Any]) -> dict[str, str]:
    """Oracle参照応答JSONを検証する。"""
    response = str(payload.get("oracle_response", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    if not response:
        raise ValueError("`oracle_response` が空です。")
    return {"oracle_response": response, "oracle_reason": reason}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    """dictリスト内の数値平均を返す。"""
    return sum(float(row[key]) for row in rows) / len(rows)


def _axis_mean(rows: list[dict[str, Any]], axis_key: str) -> float:
    """単独採点の軸別平均を返す。"""
    return sum(float(row["axis_scores"][axis_key]) for row in rows) / len(rows)


def summarize_prompt_only_judgments(
    judgments: list[dict[str, Any]],
    *,
    baseline_summary: dict[str, Any] | None = None,
    baseline_summary_path: str = "",
) -> dict[str, Any]:
    """prompt-only単独採点結果を集計する。"""
    if not judgments:
        raise ValueError("集計対象のjudgmentがありません。")
    axis_scores = {
        axis_key: {"mean_prompt_only": _axis_mean(judgments, axis_key)}
        for axis_key in ESCONV_STRATEGY_V3_AXIS_KEYS
    }
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({str(row["category"]) for row in judgments}):
        rows = [row for row in judgments if row["category"] == category]
        by_category[category] = {
            "count": len(rows),
            "esconv_core_score": {"mean_prompt_only": _mean(rows, "esconv_core_score")},
            "weighted_esconv_overall": {"mean_prompt_only": _mean(rows, "weighted_esconv_overall_score")},
            "axis_scores": {
                axis_key: {"mean_prompt_only": _axis_mean(rows, axis_key)}
                for axis_key in ESCONV_STRATEGY_V3_AXIS_KEYS
            },
        }
    summary = {
        "records": len(judgments),
        "comparison_kind": "prompt_only_fewshot",
        "score_definition": {
            "esconv_core_score": ESCONV_CORE_WEIGHTS,
            "weighted_esconv_overall": WEIGHTED_ESCONV_OVERALL_WEIGHTS,
            "win_tie_threshold": WIN_TIE_THRESHOLD,
            "note": "prompt-only単独採点のためwin rateは算出しません。",
        },
        "esconv_core_score": {"mean_prompt_only": _mean(judgments, "esconv_core_score")},
        "weighted_esconv_overall": {
            "mean_prompt_only": _mean(judgments, "weighted_esconv_overall_score")
        },
        "axis_scores": axis_scores,
        "by_category": by_category,
    }
    if baseline_summary:
        summary["baseline_comparison"] = build_baseline_comparison(
            summary,
            baseline_summary=baseline_summary,
            baseline_summary_path=baseline_summary_path,
        )
    return summary


def build_baseline_comparison(
    prompt_only_summary: dict[str, Any],
    *,
    baseline_summary: dict[str, Any],
    baseline_summary_path: str,
) -> dict[str, Any]:
    """既存Bayes-DPO summaryとの差分を作る。"""
    comparison = {
        "baseline_label": "bayes_dpo",
        "baseline_summary_path": baseline_summary_path,
        "score_direction": "prompt_only_minus_bayes_dpo",
        "esconv_core_score": _metric_comparison(
            prompt_only_summary["esconv_core_score"],
            baseline_summary.get("esconv_core_score", {}),
            "mean_prompt_only",
            "mean_dpo",
        ),
        "weighted_esconv_overall": _metric_comparison(
            prompt_only_summary["weighted_esconv_overall"],
            baseline_summary.get("weighted_esconv_overall", {}),
            "mean_prompt_only",
            "mean_dpo",
        ),
        "axis_scores": {},
    }
    axis_comparison: dict[str, Any] = {}
    baseline_axes = baseline_summary.get("axis_scores", {})
    for axis_key, prompt_axis in prompt_only_summary["axis_scores"].items():
        axis_comparison[axis_key] = _metric_comparison(
            prompt_axis,
            baseline_axes.get(axis_key, {}),
            "mean_prompt_only",
            "mean_dpo",
        )
    comparison["axis_scores"] = axis_comparison
    return comparison


def _metric_comparison(
    prompt_metric: dict[str, Any],
    baseline_metric: dict[str, Any],
    prompt_key: str,
    baseline_key: str,
) -> dict[str, float | None]:
    """prompt-onlyとbaselineの1指標差分を返す。"""
    prompt_value = _optional_float(prompt_metric.get(prompt_key))
    baseline_value = _optional_float(baseline_metric.get(baseline_key))
    gap = None
    if prompt_value is not None and baseline_value is not None:
        gap = prompt_value - baseline_value
    return {
        prompt_key: prompt_value,
        baseline_key: baseline_value,
        "gap": gap,
    }


def _optional_float(value: Any) -> float | None:
    """Noneを許容してfloat化する。"""
    if value is None:
        return None
    return float(value)


def load_baseline_summary(path: str) -> tuple[dict[str, Any] | None, str]:
    """既存Bayes-DPO summaryを読む。見つからなければfallbackも試す。"""
    candidates = [Path(path)] if str(path).strip() else []
    fallback = Path(DEFAULT_BASELINE_SUMMARY_FALLBACK_PATH)
    if fallback not in candidates:
        candidates.append(fallback)
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8")), str(candidate)
    return None, ""


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    category_filter = parse_category_filter(args.categories)
    prompts = read_evaluation_prompts(
        args.prompts,
        max_prompts=args.max_prompts,
        skip_prompts=args.skip_prompts,
        categories=category_filter,
    )
    bayes_model = load_transition_bayes_model(args.bayes_model)
    small_corpus_text = load_small_corpus_context(args.small_corpus, max_chars=args.small_corpus_max_chars)
    all_fewshot_examples = read_fewshot_examples(args.fewshot_examples)
    fewshot_examples = select_balanced_fewshot_examples(
        all_fewshot_examples,
        count=args.fewshot_count,
        seed=args.seed,
    )
    baseline_summary, baseline_summary_path = load_baseline_summary(args.baseline_summary)
    output_dir = Path(args.output_dir)
    responses_path = output_dir / "responses.jsonl"
    judgments_path = output_dir / "judgments.jsonl"
    summary_path = output_dir / "summary.json"
    failures_path = output_dir / "failures.jsonl"
    manifest_path = output_dir / "manifest.json"

    if args.dry_run:
        print("Prompt-only few-shot Oracle評価 dry-run")
        print(f"  prompts: {args.prompts} ({len(prompts)} 件)")
        print(f"  small_corpus: {args.small_corpus} ({len(small_corpus_text)} chars)")
        print(f"  bayes_model: {bayes_model.name}")
        print(f"  base_model_id: {args.base_model_id}")
        print(f"  fewshot_examples: {args.fewshot_examples}")
        print(f"  fewshot_count: {len(fewshot_examples)} / {len(all_fewshot_examples)}")
        print(f"  fewshot_strategies: {json.dumps([item.source_strategy for item in fewshot_examples], ensure_ascii=False)}")
        print(f"  oracle_model: {args.oracle_model}")
        print(f"  oracle_workers: {max(1, args.oracle_workers)}")
        print(f"  style_preset: {args.style_preset}")
        print(f"  baseline_summary: {baseline_summary_path or 'not found'}")
        print(f"  output_dir: {output_dir}")
        return 0

    existing_responses = read_jsonl_lenient(responses_path)
    existing_judgments = read_jsonl_lenient(judgments_path)
    existing_judgment_keys = set(records_by_sample_key(existing_judgments))
    prompt_keys = {prompt.prompt_id for prompt in prompts}
    if summary_path.exists() and prompt_keys and prompt_keys <= existing_judgment_keys:
        print(f"[Prompt-only Oracle] 完了済みsummaryを検出したため既存成果物を上書きしません: {summary_path}")
        print(f"[Prompt-only Oracle] completed judgments: {len(existing_judgment_keys)}/{len(prompt_keys)}")
        return 0

    response_records = generate_prompt_only_responses(
        prompts,
        base_model_id=args.base_model_id,
        fewshot_examples=fewshot_examples,
        fewshot_examples_path=args.fewshot_examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        use_4bit=args.use_4bit,
        existing_response_records=existing_responses,
        responses_path=responses_path,
    )
    responses_with_oracle, judgments = run_prompt_only_oracle_judgment(
        response_records,
        bayes_model=bayes_model,
        small_corpus_text=small_corpus_text,
        oracle_model=args.oracle_model,
        max_output_tokens=args.oracle_max_output_tokens,
        generator=OpenAIResponsesGenerator(),
        oracle_workers=max(1, args.oracle_workers),
        existing_judgment_records=existing_judgments,
        judgments_path=judgments_path,
        responses_path=responses_path,
        failures_path=failures_path,
    )
    if not judgments:
        raise RuntimeError("Oracle評価で成功したjudgmentがありません。failures.jsonlを確認してください。")
    summary = summarize_prompt_only_judgments(
        judgments,
        baseline_summary=baseline_summary,
        baseline_summary_path=baseline_summary_path,
    )
    if len(judgments) == len(prompts):
        write_jsonl(responses_with_oracle, responses_path)
    write_jsonl(judgments, judgments_path)
    write_json(summary, summary_path)
    write_json(
        {
            "comparison_kind": "prompt_only_fewshot",
            "prompts": args.prompts,
            "small_corpus": args.small_corpus,
            "small_corpus_chars": len(small_corpus_text),
            "bayes_model": args.bayes_model,
            "fewshot_examples": args.fewshot_examples,
            "fewshot_count": len(fewshot_examples),
            "fewshot_example_ids": [example.example_id for example in fewshot_examples],
            "fewshot_source_strategies": [example.source_strategy for example in fewshot_examples],
            "output_dir": args.output_dir,
            "base_model_id": args.base_model_id,
            "oracle_model": args.oracle_model,
            "oracle_workers": max(1, args.oracle_workers),
            "style_preset": args.style_preset,
            "skip_prompts": args.skip_prompts,
            "categories": sorted(category_filter),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "use_4bit": args.use_4bit,
            "baseline_summary": baseline_summary_path,
            "prompt_template_version": PROMPT_ONLY_TEMPLATE_VERSION,
            "oracle_reference_template_version": reference_template_version(args.style_preset),
            "oracle_judge_template_version": SINGLE_JUDGE_TEMPLATE_VERSION,
        },
        manifest_path,
    )
    print(f"Prompt-only few-shot Oracle評価responsesを書き出しました: {responses_path}")
    print(f"Prompt-only few-shot Oracle評価judgmentsを書き出しました: {judgments_path}")
    print(f"Prompt-only few-shot Oracle評価summaryを書き出しました: {summary_path}")
    print(
        "結果: "
        f"core={summary['esconv_core_score']['mean_prompt_only']:.2f} "
        f"overall={summary['weighted_esconv_overall']['mean_prompt_only']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
