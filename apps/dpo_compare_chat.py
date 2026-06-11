"""DPO学習前後のQwen3.5返答を比較するStreamlitアプリ。"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.local_llm_utils import extract_qwen_final_text  # noqa: E402
from apps.dpo_text_chat import (  # noqa: E402
    append_history_line,
    append_prompt_history_turn,
    build_dpo_generation_prompt,
    build_dpo_prompt,
    create_run_dir,
)


DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3.5-27B"
DEFAULT_LORA_PATH = (
    "artifacts/training_runs/"
    "qwen35_bayes_dpo_lora_reminiscence_5000_to_2000_ep1_lr5e-6_r8_a16_no4bit"
)
TRAINING_RUNS_DIR = Path("artifacts/training_runs")
DEFAULT_MAX_NEW_TOKENS = 192
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_REPETITION_PENALTY = 1.0


@dataclass
class CompareModels:
    """比較UIで使うモデル一式。"""

    tokenizer: object
    model: object
    device: str


def read_env_value(name: str, default: str) -> str:
    """環境変数から設定値を読む。"""
    return os.getenv(name, default).strip() or default


def list_lora_adapter_paths(training_runs_dir: Path = TRAINING_RUNS_DIR) -> list[str]:
    """training_runs配下からLoRA adapter候補を列挙する。"""
    if not training_runs_dir.exists():
        return []
    adapter_paths: list[str] = []
    for path in sorted(training_runs_dir.iterdir(), key=lambda item: item.name, reverse=True):
        if not path.is_dir():
            continue
        if (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists():
            adapter_paths.append(path.as_posix())
    return adapter_paths


def build_dpo_compare_prompt(
    user_text: str,
    history_turns: list[dict[str, str]] | None = None,
) -> str:
    """学習データと同じ形式の比較用promptを作る。"""
    return build_dpo_prompt(user_text, history_turns)


def build_prompt_history_from_turns(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """比較UIの会話ターンからDPO prompt用の履歴を作る。"""
    history: list[dict[str, str]] = []
    for turn in turns:
        user_text = str(turn.get("user", "")).strip()
        ai_text = str(turn.get("assistant", "")).strip()
        if user_text:
            append_prompt_history_turn(history, "User", user_text)
        if ai_text:
            append_prompt_history_turn(history, "AI", ai_text)
    return history


def write_streamlit_session_header(history_file: str, *, base_model_id: str, lora_path: str) -> None:
    """Streamlit比較UIの会話ログへセッション情報を残す。"""
    with open(history_file, "a", encoding="utf-8") as file:
        file.write(f"# session_start: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write("# mode: streamlit_compare\n")
        file.write(f"# base_model_id: {base_model_id}\n")
        file.write(f"# lora_path: {lora_path}\n")
        file.write("# thinking: disabled\n")
        file.write("# conversation_mode: independent\n")
        file.write("# prompt_history: independent_per_model\n")
        file.write("\n")


def independent_history_file(run_dir: str, model_label: str) -> str:
    """独立会話ごとのログファイルパスを返す。"""
    return (Path(run_dir) / f"{model_label}_conversation.txt").as_posix()


def write_independent_session_header(
    history_file: str,
    *,
    model_label: str,
    title: str,
    base_model_id: str,
    lora_path: str,
) -> None:
    """左右それぞれの独立会話ログへセッション情報を残す。"""
    with open(history_file, "a", encoding="utf-8") as file:
        file.write(f"# session_start: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write("# mode: streamlit_compare_independent_chat\n")
        file.write(f"# model_label: {model_label}\n")
        file.write(f"# title: {title}\n")
        file.write(f"# base_model_id: {base_model_id}\n")
        file.write(f"# lora_path: {lora_path}\n")
        file.write("# thinking: disabled\n")
        file.write("# prompt_template: dpo\n")
        file.write("\n")


def setup_independent_history_files(run_dir: str, *, base_model_id: str, lora_path: str) -> tuple[str, str]:
    """base/DPOそれぞれの独立会話ログを初期化する。"""
    base_history_file = independent_history_file(run_dir, "base")
    dpo_history_file = independent_history_file(run_dir, "dpo")
    write_independent_session_header(
        base_history_file,
        model_label="base",
        title="学習前: Qwen3.5",
        base_model_id=base_model_id,
        lora_path=lora_path,
    )
    write_independent_session_header(
        dpo_history_file,
        model_label="dpo",
        title="学習後: DPO LoRA",
        base_model_id=base_model_id,
        lora_path=lora_path,
    )
    return base_history_file, dpo_history_file


def ensure_streamlit_session(base_model_id: str, lora_path: str) -> None:
    """Streamlitの会話履歴とログ保存先を初期化する。"""
    if "dpo_compare_base_turns" not in st.session_state:
        st.session_state.dpo_compare_base_turns = []
    if "dpo_compare_dpo_turns" not in st.session_state:
        st.session_state.dpo_compare_dpo_turns = []
    model_changed = (
        st.session_state.get("dpo_compare_base_model_id") != base_model_id
        or st.session_state.get("dpo_compare_lora_path") != lora_path
    )
    if "dpo_compare_history_file" not in st.session_state or model_changed:
        run_dir, history_file = create_run_dir(
            code_id="dpo_compare",
            base_model_id=base_model_id,
            lora_path=lora_path,
        )
        st.session_state.dpo_compare_run_dir = run_dir
        st.session_state.dpo_compare_history_file = history_file
        base_history_file, dpo_history_file = setup_independent_history_files(
            run_dir,
            base_model_id=base_model_id,
            lora_path=lora_path,
        )
        st.session_state.dpo_compare_base_history_file = base_history_file
        st.session_state.dpo_compare_dpo_history_file = dpo_history_file
        st.session_state.dpo_compare_base_model_id = base_model_id
        st.session_state.dpo_compare_lora_path = lora_path
        write_streamlit_session_header(history_file, base_model_id=base_model_id, lora_path=lora_path)
        append_history_line(history_file, "log(base)", base_history_file)
        append_history_line(history_file, "log(dpo)", dpo_history_file)


def reset_streamlit_session(base_model_id: str, lora_path: str) -> None:
    """Streamlitの会話履歴をリセットし、新しいログを作る。"""
    run_dir, history_file = create_run_dir(
        code_id="dpo_compare",
        base_model_id=base_model_id,
        lora_path=lora_path,
    )
    st.session_state.dpo_compare_base_turns = []
    st.session_state.dpo_compare_dpo_turns = []
    st.session_state.dpo_compare_run_dir = run_dir
    st.session_state.dpo_compare_history_file = history_file
    base_history_file, dpo_history_file = setup_independent_history_files(
        run_dir,
        base_model_id=base_model_id,
        lora_path=lora_path,
    )
    st.session_state.dpo_compare_base_history_file = base_history_file
    st.session_state.dpo_compare_dpo_history_file = dpo_history_file
    st.session_state.dpo_compare_base_model_id = base_model_id
    st.session_state.dpo_compare_lora_path = lora_path
    write_streamlit_session_header(history_file, base_model_id=base_model_id, lora_path=lora_path)
    append_history_line(history_file, "log(base)", base_history_file)
    append_history_line(history_file, "log(dpo)", dpo_history_file)


def reset_independent_chat(model_label: str) -> None:
    """片側の独立会話だけをリセットする。"""
    turns_key = f"dpo_compare_{model_label}_turns"
    st.session_state[turns_key] = []
    history_file = st.session_state.get(f"dpo_compare_{model_label}_history_file")
    if history_file:
        append_history_line(history_file, f"system({model_label})", "conversation reset")


def strip_prompt_prefix(decoded_text: str, prompt_text: str) -> str:
    """decode結果にpromptが含まれる場合、生成部分だけを取り出す。"""
    if decoded_text.startswith(prompt_text):
        return decoded_text[len(prompt_text):].strip()
    marker = "AI:"
    marker_index = decoded_text.rfind(marker)
    if marker_index >= 0:
        return decoded_text[marker_index + len(marker):].strip()
    return decoded_text.strip()


def cleanup_generated_text(decoded_text: str, prompt_text: str) -> str:
    """生成結果を表示用の返答本文へ整える。"""
    generated_text = strip_prompt_prefix(decoded_text, prompt_text)
    reply = extract_qwen_final_text(generated_text)
    if reply:
        return reply
    fallback = extract_qwen_final_text(generated_text, show_thinking=True)
    return fallback or generated_text


@contextmanager
def adapter_disabled(model: object) -> Iterator[None]:
    """PEFT adapterを一時的に無効化する。"""
    disable_adapter = getattr(model, "disable_adapter", None)
    if disable_adapter is None:
        yield
        return
    with disable_adapter():
        yield


def load_training_modules() -> dict[str, object]:
    """重い依存を遅延読み込みする。"""
    from core.hf_kernel_compat import disable_hub_kernel_integration

    disable_hub_kernel_integration()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        try:
            from transformers import Qwen3_5ForConditionalGeneration as ModelClass
        except ImportError:
            ModelClass = AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "比較UIの実行に必要な依存関係が不足しています。"
            "`python3 -m pip install -r requirements.txt` を実行してください。"
        ) from exc
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoProcessor": AutoProcessor,
        "AutoTokenizer": AutoTokenizer,
        "ModelClass": ModelClass,
    }


def disable_peft_bitsandbytes_dispatch() -> None:
    """PEFTのLoRA挿入時にbitsandbytes backendを使わせない。"""
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.model as peft_lora_model
    except ImportError:
        return

    def _always_false() -> bool:
        return False

    for module in (peft_import_utils, peft_lora_model):
        for name in ("is_bnb_available", "is_bnb_4bit_available"):
            detector = getattr(module, name, None)
            if hasattr(detector, "cache_clear"):
                detector.cache_clear()
            setattr(module, name, _always_false)


def load_tokenizer(model_id: str, deps: dict[str, object]):
    """Qwen tokenizerを読み込む。"""
    try:
        processor = deps["AutoProcessor"].from_pretrained(model_id, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None) or processor
    except Exception:
        tokenizer = deps["AutoTokenizer"].from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


@st.cache_resource(show_spinner=False)
def load_compare_models(base_model_id: str, lora_path: str) -> CompareModels:
    """ベースモデルにDPO LoRA adapterを載せて読み込む。"""
    deps = load_training_modules()
    disable_peft_bitsandbytes_dispatch()
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応GPUが見つかりません。Qwen3.5-27B の比較にはGPU環境が必要です。")
    if not Path(lora_path).exists():
        raise RuntimeError(f"LoRA adapter が見つかりません: {lora_path}")

    tokenizer = load_tokenizer(base_model_id, deps)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = deps["ModelClass"].from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if hasattr(base_model, "config"):
        base_model.config.use_cache = True
    model = deps["PeftModel"].from_pretrained(base_model, lora_path)
    model.eval()
    return CompareModels(tokenizer=tokenizer, model=model, device="cuda")


def generate_reply(
    compare_models: CompareModels,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    use_adapter: bool,
) -> str:
    """ベースまたはDPO後モデルで返答を生成する。"""
    torch = load_training_modules()["torch"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = compare_models.tokenizer
    generation_prompt = build_dpo_generation_prompt(tokenizer, prompt_text)
    model_inputs = tokenizer(generation_prompt, return_tensors="pt")
    model_inputs = {
        key: value.to(compare_models.device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }
    input_ids = model_inputs["input_ids"]
    if "attention_mask" not in model_inputs:
        model_inputs["attention_mask"] = torch.ones_like(input_ids, device=compare_models.device)

    def _run_generate():
        with torch.no_grad():
            return compare_models.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

    if use_adapter:
        output_ids = _run_generate()
    else:
        with adapter_disabled(compare_models.model):
            output_ids = _run_generate()

    generated = output_ids[0][input_ids.shape[1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=False).strip()
    return cleanup_generated_text(decoded, generation_prompt)


def render_independent_chat_column(
    *,
    title: str,
    model_label: str,
    use_adapter: bool,
    compare_models: CompareModels | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> None:
    """base/DPOそれぞれの独立したチャット欄を描画する。"""
    turns_key = f"dpo_compare_{model_label}_turns"
    form_key = f"dpo_compare_{model_label}_form"
    input_key = f"dpo_compare_{model_label}_input"

    st.subheader(title)
    if st.button("この会話をリセット", key=f"dpo_compare_{model_label}_reset"):
        reset_independent_chat(model_label)
        st.rerun()

    turns = st.session_state[turns_key]
    for turn in turns:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant"):
            st.markdown(turn["assistant"])

    prompt_history = build_prompt_history_from_turns(turns)
    with st.expander("生成に使うprompt", expanded=False):
        preview_text = st.session_state.get(input_key, "")
        preview_prompt = build_dpo_compare_prompt(preview_text, prompt_history) if str(preview_text).strip() else ""
        st.code(preview_prompt or "入力後に表示されます。", language="text")

    with st.form(form_key, clear_on_submit=True):
        user_text = st.text_area(
            f"{title}への入力",
            key=input_key,
            height=120,
            placeholder="ここにメッセージを入力してください",
        )
        submitted = st.form_submit_button("送信", disabled=compare_models is None)

    if not submitted:
        return

    clean_text = user_text.strip()
    if not clean_text:
        st.warning("メッセージを入力してください。")
        return
    if compare_models is None:
        st.error("モデルが読み込まれていません。")
        return

    prompt_text = build_dpo_compare_prompt(clean_text, prompt_history)
    try:
        with st.spinner(f"{title} が返答を生成しています。"):
            reply = generate_reply(
                compare_models,
                prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                use_adapter=use_adapter,
            )
    except Exception as exc:
        st.error(f"生成に失敗しました: {exc}")
        return

    final_reply = reply or "（空の返答）"
    turns.append({"user": clean_text, "assistant": final_reply})
    history_file = st.session_state[f"dpo_compare_{model_label}_history_file"]
    append_history_line(history_file, "User", clean_text)
    append_history_line(history_file, "AI", final_reply)
    st.rerun()


def render_app() -> None:
    """Streamlit UIを描画する。"""
    st.set_page_config(page_title="DPO前後比較チャット", layout="wide")
    st.title("DPO前後 独立比較チャット")

    with st.sidebar:
        st.header("モデル設定")
        base_model_id = st.text_input(
            "ベースモデル",
            value=read_env_value("DPO_COMPARE_BASE_MODEL", DEFAULT_BASE_MODEL_ID),
        )
        lora_candidates = list_lora_adapter_paths()
        default_lora_path = read_env_value("DPO_COMPARE_LORA_PATH", DEFAULT_LORA_PATH)
        if lora_candidates:
            default_index = lora_candidates.index(default_lora_path) if default_lora_path in lora_candidates else 0
            selected_lora_path = st.selectbox(
                "DPO LoRA adapter候補",
                lora_candidates,
                index=default_index,
            )
        else:
            selected_lora_path = default_lora_path
        lora_path = st.text_input(
            "DPO LoRA adapter",
            value=selected_lora_path,
        )
        st.header("生成設定")
        max_new_tokens = st.slider("max_new_tokens", 32, 512, DEFAULT_MAX_NEW_TOKENS, step=16)
        temperature = st.slider("temperature", 0.0, 1.5, DEFAULT_TEMPERATURE, step=0.05)
        top_p = st.slider("top_p", 0.1, 1.0, DEFAULT_TOP_P, step=0.05)
        repetition_penalty = st.slider(
            "repetition_penalty",
            0.8,
            1.5,
            DEFAULT_REPETITION_PENALTY,
            step=0.05,
        )
        seed = st.number_input("seed", value=42, min_value=0, max_value=2_147_483_647)
        ensure_streamlit_session(base_model_id, lora_path)
        st.caption(f"ログ出力先: {st.session_state.dpo_compare_run_dir}/")
        st.caption(f"baseログ: `{st.session_state.dpo_compare_base_history_file}`")
        st.caption(f"DPOログ: `{st.session_state.dpo_compare_dpo_history_file}`")
        if st.button("両方の会話をリセット"):
            reset_streamlit_session(base_model_id, lora_path)
            st.rerun()
        st.caption("左右の会話履歴と入力は独立しています。")

    try:
        with st.spinner("モデルを読み込んでいます。初回は時間がかかります。"):
            compare_models: CompareModels | None = load_compare_models(base_model_id, lora_path)
    except Exception as exc:
        compare_models = None
        st.error(f"モデル読み込みに失敗しました: {exc}")

    left, right = st.columns(2, gap="large")
    with left:
        render_independent_chat_column(
            title="学習前: Qwen3.5",
            model_label="base",
            use_adapter=False,
            compare_models=compare_models,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=int(seed),
        )
    with right:
        render_independent_chat_column(
            title="学習後: DPO LoRA",
            model_label="dpo",
            use_adapter=True,
            compare_models=compare_models,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=int(seed),
        )


if __name__ == "__main__":
    render_app()
