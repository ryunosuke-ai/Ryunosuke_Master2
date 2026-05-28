"""DPO学習後のQwen3.5だけと会話するStreamlitアプリ。"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.dpo_compare_chat import list_lora_adapter_paths  # noqa: E402
from apps.dpo_text_chat import (  # noqa: E402
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    ChatBundle,
    append_history_line,
    append_prompt_history_turn,
    build_dpo_generation_prompt,
    build_dpo_log_model_id,
    build_dpo_prompt,
    cleanup_generated_text,
    disable_peft_bitsandbytes_dispatch,
    load_tokenizer,
    load_training_modules,
)
from core.log_manager import create_log_run_dir  # noqa: E402


DEFAULT_LORA_PATH = "artifacts/training_runs/qwen35_dpo_lora_300samples_ep1_lr5e-6_r8_a16_no4bit"
DEFAULT_ENV_BASE_MODEL = "DPO_TRAINED_CHAT_BASE_MODEL"
DEFAULT_ENV_LORA_PATH = "DPO_TRAINED_CHAT_LORA_PATH"


def read_env_value(name: str, default: str) -> str:
    """環境変数から設定値を読む。"""
    return os.getenv(name, default).strip() or default


def create_run_dir(
    *,
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    lora_path: str = DEFAULT_LORA_PATH,
) -> tuple[str, str]:
    """DPO後モデル会話ログ保存用のrunディレクトリを作る。"""
    run_dir, history_file, _ts = create_log_run_dir(
        "dpo_trained_chat",
        build_dpo_log_model_id(base_model_id, lora_path),
        metadata={
            "base_model_id": base_model_id,
            "lora_path": lora_path,
        },
    )
    return run_dir, history_file


def write_streamlit_session_header(history_file: str, *, base_model_id: str, lora_path: str) -> None:
    """Streamlit DPO後モデル会話ログへセッション情報を残す。"""
    with open(history_file, "a", encoding="utf-8") as file:
        file.write(f"# session_start: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write("# mode: streamlit_dpo_trained_chat\n")
        file.write(f"# base_model_id: {base_model_id}\n")
        file.write(f"# lora_path: {lora_path}\n")
        file.write("# thinking: disabled\n")
        file.write("# prompt_template: dpo\n")
        file.write("\n")


def ensure_streamlit_session(base_model_id: str, lora_path: str) -> None:
    """Streamlitの会話履歴とログ保存先を初期化する。"""
    if "dpo_trained_prompt_history" not in st.session_state:
        st.session_state.dpo_trained_prompt_history = []
    if "dpo_trained_turns" not in st.session_state:
        st.session_state.dpo_trained_turns = []
    if "dpo_trained_history_file" not in st.session_state:
        run_dir, history_file = create_run_dir(base_model_id=base_model_id, lora_path=lora_path)
        st.session_state.dpo_trained_run_dir = run_dir
        st.session_state.dpo_trained_history_file = history_file
        write_streamlit_session_header(history_file, base_model_id=base_model_id, lora_path=lora_path)


def reset_streamlit_session(base_model_id: str, lora_path: str) -> None:
    """Streamlitの会話履歴をリセットし、新しいログを作る。"""
    run_dir, history_file = create_run_dir(base_model_id=base_model_id, lora_path=lora_path)
    st.session_state.dpo_trained_prompt_history = []
    st.session_state.dpo_trained_turns = []
    st.session_state.dpo_trained_run_dir = run_dir
    st.session_state.dpo_trained_history_file = history_file
    write_streamlit_session_header(history_file, base_model_id=base_model_id, lora_path=lora_path)


@st.cache_resource(show_spinner=False)
def load_trained_chat_bundle(base_model_id: str, lora_path: str) -> ChatBundle:
    """ベースモデルにDPO LoRA adapterを載せて読み込む。"""
    deps = load_training_modules()
    disable_peft_bitsandbytes_dispatch()
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応GPUが見つかりません。Qwen3.5-27B の実行にはGPU環境が必要です。")
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
    return ChatBundle(tokenizer=tokenizer, model=model, torch=torch)


def generate_reply(
    bundle: ChatBundle,
    user_text: str,
    *,
    history_turns: list[dict[str, str]] | None = None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> str:
    """DPO学習後モデルで返答を生成する。"""
    prompt_text = build_dpo_prompt(user_text, history_turns)
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


def render_app() -> None:
    """Streamlit UIを描画する。"""
    st.set_page_config(page_title="DPO後モデル単独チャット", layout="wide")
    st.title("DPO後モデル単独チャット")

    with st.sidebar:
        st.header("モデル設定")
        base_model_id = st.text_input(
            "ベースモデル",
            value=read_env_value(DEFAULT_ENV_BASE_MODEL, DEFAULT_BASE_MODEL_ID),
        )
        lora_candidates = list_lora_adapter_paths()
        default_lora_path = read_env_value(DEFAULT_ENV_LORA_PATH, DEFAULT_LORA_PATH)
        if lora_candidates:
            default_index = lora_candidates.index(default_lora_path) if default_lora_path in lora_candidates else 0
            selected_lora_path = st.selectbox(
                "DPO LoRA adapter候補",
                lora_candidates,
                index=default_index,
            )
        else:
            selected_lora_path = default_lora_path
        lora_path = st.text_input("DPO LoRA adapter", value=selected_lora_path)

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
        st.caption(f"ログ出力先: {st.session_state.dpo_trained_run_dir}/")
        if st.button("会話をリセット"):
            reset_streamlit_session(base_model_id, lora_path)
            st.rerun()

    prompt_history = st.session_state.dpo_trained_prompt_history
    for turn in st.session_state.dpo_trained_turns:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant"):
            st.markdown(turn["ai"])

    user_text = st.chat_input("DPO後モデルに送る発話を入力してください")
    if not user_text:
        return

    append_history_line(st.session_state.dpo_trained_history_file, "User", user_text)
    with st.chat_message("user"):
        st.markdown(user_text)

    try:
        with st.spinner("モデルを読み込んでいます。初回は時間がかかります。"):
            bundle = load_trained_chat_bundle(base_model_id, lora_path)
        with st.spinner("返答を生成しています。"):
            reply = generate_reply(
                bundle,
                user_text,
                history_turns=prompt_history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=int(seed),
            )
    except Exception as exc:
        st.error(f"生成に失敗しました: {exc}")
        return

    final_reply = reply or "（空の返答）"
    st.session_state.dpo_trained_turns.append({"user": user_text, "ai": final_reply})
    append_prompt_history_turn(prompt_history, "User", user_text)
    append_prompt_history_turn(prompt_history, "AI", final_reply)
    append_history_line(st.session_state.dpo_trained_history_file, "AI(dpo)", final_reply)

    with st.chat_message("assistant"):
        st.markdown(final_reply)


if __name__ == "__main__":
    render_app()
