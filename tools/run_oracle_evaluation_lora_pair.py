"""2つのLoRA adapterを直接比較するOracle評価runner。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.dpo_compare_text_chat import (  # noqa: E402
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_ENV_MAX_MEMORY,
    build_dpo_generation_prompt,
    cleanup_generated_text,
    disable_peft_bitsandbytes_dispatch,
    load_tokenizer,
    load_training_modules,
    parse_max_memory_env,
    suppress_external_warnings,
)
from apps.dpo_text_chat import ChatBundle  # noqa: E402
from core.transition_bayes_model import load_transition_bayes_model  # noqa: E402
from tools.analyze_small_corpus import OpenAIResponsesGenerator, resolve_analysis_model  # noqa: E402
from tools.run_oracle_evaluation import (  # noqa: E402
    DEFAULT_ORACLE_MAX_OUTPUT_TOKENS,
    ESCONV_STRATEGY_V3_PRESET,
    PROMPT_TEMPLATE_VERSION,
    append_jsonl_record,
    build_local_model_prompt,
    judge_template_version,
    load_small_corpus_context,
    parse_category_filter,
    read_evaluation_prompts,
    read_jsonl_lenient,
    reference_template_version,
    records_by_sample_key,
    retry_config_from_env,
    run_oracle_judgment,
    summarize_judgments,
    write_json,
    write_jsonl,
)
from tools.score_dialogue_with_bayes_model import load_env_file  # noqa: E402


DEFAULT_PROMPTS_PATH = "configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl"
DEFAULT_SMALL_CORPUS_PATH = "data/esconv_analysis_corpus_reminiscence_5000_to_2000.jsonl"
DEFAULT_BAYES_MODEL_PATH = (
    "artifacts/bayes_models/generated_transition_bayes_model_esconv_reminiscence_5000_to_2000.json"
)
DEFAULT_BAYES_LORA_PATH = (
    "artifacts/training_runs/"
    "qwen35_bayes_dpo_lora_reminiscence_5000_to_2000_ep1_lr5e-6_r8_a16_no4bit"
)
DEFAULT_RANDOM_LORA_PATH = (
    "artifacts/training_runs/"
    "qwen35_random2500_dailydialog_dpo_lora_esconv_5000_to_2000_random2500_ep1_lr5e-6_r8_a16_no4bit"
)
DEFAULT_OUTPUT_DIR = (
    "artifacts/evaluations/oracle_eval_runs/"
    "esconv_5000_to_2000_bayes_vs_random2500_oracle_esconv_v3_strategy"
)
BASE_FIELD_LABEL = "bayes_dpo"
DPO_FIELD_LABEL = "random_dpo"
BASE_ADAPTER_NAME = "bayes_dpo"
DPO_ADAPTER_NAME = "random_dpo"


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    load_env_file()
    default_oracle_model = resolve_analysis_model()
    parser = argparse.ArgumentParser(
        description="Bayes-DPO LoRAとRandom-DPO LoRAをOracle評価で直接比較します。"
    )
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--small-corpus", default=DEFAULT_SMALL_CORPUS_PATH)
    parser.add_argument("--bayes-model", default=DEFAULT_BAYES_MODEL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--base-lora-path", default=DEFAULT_BAYES_LORA_PATH)
    parser.add_argument("--dpo-lora-path", default=DEFAULT_RANDOM_LORA_PATH)
    parser.add_argument("--oracle-model", default=default_oracle_model)
    parser.add_argument("--oracle-workers", type=int, default=1)
    parser.add_argument(
        "--style-preset",
        choices=(ESCONV_STRATEGY_V3_PRESET,),
        default=ESCONV_STRATEGY_V3_PRESET,
    )
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--skip-prompts", type=int, default=0)
    parser.add_argument("--categories", default="")
    parser.add_argument(
        "--local-prompt-mode",
        choices=("instruction", "context_only"),
        default="instruction",
    )
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


def require_lora_path(path: Path | str, *, label: str) -> None:
    """LoRA adapterの存在を事前検証する。"""
    lora_path = Path(path)
    if not lora_path.exists():
        raise FileNotFoundError(f"{label} LoRA not found: {lora_path}")
    if not (lora_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"{label} LoRA adapter_config.json not found: {lora_path}")


def load_lora_pair_bundle(
    base_model_id: str,
    *,
    base_lora_path: str,
    dpo_lora_path: str,
    use_4bit: bool,
) -> ChatBundle:
    """1つのベースモデルへ2つのLoRA adapterを読み込む。"""
    suppress_external_warnings()
    deps = load_training_modules()
    disable_peft_bitsandbytes_dispatch()
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応GPUが見つかりません。Qwen3.5-27B の比較にはGPU環境が必要です。")
    require_lora_path(base_lora_path, label="Bayes-DPO")
    require_lora_path(dpo_lora_path, label="Random-DPO")

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

    base_model = deps["ModelClass"].from_pretrained(base_model_id, **model_kwargs)
    if hasattr(base_model, "config"):
        base_model.config.use_cache = True
    model = deps["PeftModel"].from_pretrained(
        base_model,
        base_lora_path,
        adapter_name=BASE_ADAPTER_NAME,
    )
    model.load_adapter(dpo_lora_path, adapter_name=DPO_ADAPTER_NAME)
    model.eval()
    return ChatBundle(tokenizer=tokenizer, model=model, torch=torch)


def generate_reply_with_adapter(
    bundle: ChatBundle,
    prompt_text: str,
    *,
    adapter_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> str:
    """指定したLoRA adapterで返答を生成する。"""
    set_adapter = getattr(bundle.model, "set_adapter", None)
    if set_adapter is None:
        raise RuntimeError("読み込んだモデルがLoRA adapterの切り替えに対応していません。")
    set_adapter(adapter_name)

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


def generate_lora_pair_responses(
    prompts: list[Any],
    *,
    base_model_id: str,
    base_lora_path: str,
    dpo_lora_path: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    use_4bit: bool,
    local_prompt_mode: str,
    existing_response_records: list[dict[str, Any]] | None = None,
    responses_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Bayes-DPOをbase field、Random-DPOをdpo fieldとして応答生成する。"""
    existing_by_key = records_by_sample_key(existing_response_records or [])
    if existing_by_key:
        print(f"[Oracle Eval] found existing responses: {len(existing_by_key)}", flush=True)
    missing_prompts = [prompt for prompt in prompts if prompt.prompt_id not in existing_by_key]
    bundle = None
    if missing_prompts:
        bundle = load_lora_pair_bundle(
            base_model_id,
            base_lora_path=base_lora_path,
            dpo_lora_path=dpo_lora_path,
            use_4bit=use_4bit,
        )
    records: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        if prompt.prompt_id in existing_by_key:
            print(f"[Oracle Eval] skip local generation {index}/{len(prompts)} {prompt.prompt_id}", flush=True)
            records.append(existing_by_key[prompt.prompt_id])
            continue
        print(
            f"[Oracle Eval LoRA Pair] local generation {index}/{len(prompts)} "
            f"{prompt.prompt_id} base=bayes_dpo dpo=random_dpo",
            flush=True,
        )
        if bundle is None:
            raise RuntimeError("LoRA pair generation bundleが初期化されていません。")
        prompt_text = build_local_model_prompt(prompt, mode=local_prompt_mode)
        base_response = generate_reply_with_adapter(
            bundle,
            prompt_text,
            adapter_name=BASE_ADAPTER_NAME,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        dpo_response = generate_reply_with_adapter(
            bundle,
            prompt_text,
            adapter_name=DPO_ADAPTER_NAME,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        records.append(
            {
                "prompt_id": prompt.prompt_id,
                "category": prompt.category,
                "prompt": prompt.prompt,
                "history": [dict(turn) for turn in prompt.history],
                "axis_focus": list(prompt.axis_focus),
                "model_prompt": prompt_text,
                "base_response": base_response,
                "dpo_response": dpo_response,
                "comparison_kind": "lora_pair",
                "base_field_label": BASE_FIELD_LABEL,
                "dpo_field_label": DPO_FIELD_LABEL,
                "base_lora_path": base_lora_path,
                "dpo_lora_path": dpo_lora_path,
                "generation": {
                    "comparison_kind": "lora_pair",
                    "base_model_id": base_model_id,
                    "base_field_label": BASE_FIELD_LABEL,
                    "dpo_field_label": DPO_FIELD_LABEL,
                    "base_lora_path": base_lora_path,
                    "dpo_lora_path": dpo_lora_path,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "seed": seed,
                    "use_4bit": use_4bit,
                    "thinking": "disabled",
                    "local_prompt_mode": local_prompt_mode,
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                },
            }
        )
        if responses_path is not None:
            append_jsonl_record(records[-1], responses_path)
    if existing_by_key:
        print(
            f"[Oracle Eval] skipping completed response generation for "
            f"{sum(1 for prompt in prompts if prompt.prompt_id in existing_by_key)}/{len(prompts)} samples",
            flush=True,
        )
    return records


def add_lora_pair_summary_labels(
    summary: dict[str, Any],
    *,
    base_lora_path: str,
    dpo_lora_path: str,
) -> dict[str, Any]:
    """既存summaryにLoRA pair比較のラベル情報を追加する。"""
    enriched = dict(summary)
    enriched.update(
        {
            "comparison_kind": "lora_pair",
            "base_field_label": BASE_FIELD_LABEL,
            "dpo_field_label": DPO_FIELD_LABEL,
            "base_lora_path": base_lora_path,
            "dpo_lora_path": dpo_lora_path,
            "bayes_dpo_win_rate": summary.get("base_win_rate"),
            "random_dpo_win_rate": summary.get("dpo_win_rate"),
            "label_note": (
                "互換性のため base field は Bayes-DPO、"
                "dpo field は Random-DPO を表します。"
            ),
        }
    )
    return enriched


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
    output_dir = Path(args.output_dir)
    responses_path = output_dir / "responses.jsonl"
    judgments_path = output_dir / "judgments.jsonl"
    summary_path = output_dir / "summary.json"
    partial_summary_path = output_dir / "summary.partial.json"
    failures_path = output_dir / "failures.jsonl"
    manifest_path = output_dir / "manifest.json"

    if args.dry_run:
        print("LoRA pair Oracle評価 dry-run")
        print(f"  prompts: {args.prompts} ({len(prompts)} 件)")
        print(f"  small_corpus: {args.small_corpus} ({len(small_corpus_text)} chars)")
        print(f"  bayes_model: {bayes_model.name}")
        print(f"  base_model_id: {args.base_model_id}")
        print(f"  comparison_kind: lora_pair")
        print(f"  base_field_label: {BASE_FIELD_LABEL}")
        print(f"  dpo_field_label: {DPO_FIELD_LABEL}")
        print(f"  base_lora_path: {args.base_lora_path}")
        print(f"  dpo_lora_path: {args.dpo_lora_path}")
        print(f"  oracle_model: {args.oracle_model}")
        print(f"  oracle_workers: {max(1, args.oracle_workers)}")
        print(f"  style_preset: {args.style_preset}")
        print(f"  local_prompt_mode: {args.local_prompt_mode}")
        print(f"  output_dir: {output_dir}")
        return 0

    require_lora_path(args.base_lora_path, label="Bayes-DPO")
    require_lora_path(args.dpo_lora_path, label="Random-DPO")
    existing_responses = read_jsonl_lenient(responses_path)
    existing_judgments = read_jsonl_lenient(judgments_path)
    existing_judgment_keys = set(records_by_sample_key(existing_judgments))
    prompt_keys = {prompt.prompt_id for prompt in prompts}
    if summary_path.exists() and prompt_keys and prompt_keys <= existing_judgment_keys:
        print(f"[Oracle Eval] 完了済みsummaryを検出したため既存成果物を上書きしません: {summary_path}")
        print(f"[Oracle Eval] completed judgments: {len(existing_judgment_keys)}/{len(prompt_keys)}")
        return 0

    response_records = generate_lora_pair_responses(
        prompts,
        base_model_id=args.base_model_id,
        base_lora_path=args.base_lora_path,
        dpo_lora_path=args.dpo_lora_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        use_4bit=args.use_4bit,
        local_prompt_mode=args.local_prompt_mode,
        existing_response_records=existing_responses,
        responses_path=responses_path,
    )
    partial_summary_metadata = {
        "comparison_kind": "lora_pair",
        "base_field_label": BASE_FIELD_LABEL,
        "dpo_field_label": DPO_FIELD_LABEL,
        "base_lora_path": args.base_lora_path,
        "dpo_lora_path": args.dpo_lora_path,
    }
    responses_with_oracle, judgments = run_oracle_judgment(
        response_records,
        bayes_model=bayes_model,
        small_corpus_text=small_corpus_text,
        oracle_model=args.oracle_model,
        max_output_tokens=args.oracle_max_output_tokens,
        seed=args.seed,
        style_preset=args.style_preset,
        generator=OpenAIResponsesGenerator(),
        oracle_workers=max(1, args.oracle_workers),
        existing_judgment_records=existing_judgments,
        judgments_path=judgments_path,
        responses_path=responses_path,
        failures_path=failures_path,
        partial_summary_path=partial_summary_path,
        partial_summary_metadata=partial_summary_metadata,
        retry_config=retry_config_from_env(),
    )
    if not judgments:
        raise RuntimeError("Oracle評価で成功したjudgmentがありません。failures.jsonlを確認してください。")
    summary = add_lora_pair_summary_labels(
        summarize_judgments(judgments),
        base_lora_path=args.base_lora_path,
        dpo_lora_path=args.dpo_lora_path,
    )
    if len(judgments) == len(prompts):
        write_jsonl(responses_with_oracle, responses_path)
    write_jsonl(judgments, judgments_path)
    write_json(summary, summary_path)
    write_json(
        {
            "comparison_kind": "lora_pair",
            "base_field_label": BASE_FIELD_LABEL,
            "dpo_field_label": DPO_FIELD_LABEL,
            "base_model_id": args.base_model_id,
            "base_lora_path": args.base_lora_path,
            "dpo_lora_path": args.dpo_lora_path,
            "bayes_dpo_win_rate": summary.get("bayes_dpo_win_rate"),
            "random_dpo_win_rate": summary.get("random_dpo_win_rate"),
            "prompts": args.prompts,
            "small_corpus": args.small_corpus,
            "small_corpus_chars": len(small_corpus_text),
            "bayes_model": args.bayes_model,
            "output_dir": args.output_dir,
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
            "local_prompt_mode": args.local_prompt_mode,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "oracle_reference_template_version": reference_template_version(args.style_preset),
            "oracle_judge_template_version": judge_template_version(args.style_preset),
        },
        manifest_path,
    )
    print(f"LoRA pair Oracle評価responsesを書き出しました: {responses_path}")
    print(f"LoRA pair Oracle評価judgmentsを書き出しました: {judgments_path}")
    print(f"LoRA pair Oracle評価summaryを書き出しました: {summary_path}")
    print(
        "結果: "
        f"bayes_dpo_win_rate=base_win_rate={summary.get('bayes_dpo_win_rate', 0):.2%} "
        f"random_dpo_win_rate=dpo_win_rate={summary.get('random_dpo_win_rate', 0):.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
