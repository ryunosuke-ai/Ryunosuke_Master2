"""DPO比較用のベースQwen3.5だけと会話するStreamlitアプリ。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.dpo_text_chat import (  # noqa: E402
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    append_history_line,
    append_prompt_history_turn,
    build_dpo_generation_prompt,
    build_dpo_prompt,
    cleanup_generated_text,
    load_tokenizer,
    load_training_modules,
)
from core.log_manager import build_model_segment, create_log_run_dir  # noqa: E402


DEFAULT_ENV_BASE_MODEL = "DPO_BASE_CHAT_BASE_MODEL"


@dataclass(frozen=True)
class BaseChatBundle:
    """ベースモデル生成に必要な一式。"""

    tokenizer: object
    model: object
    torch: object

    @property
    def input_device(self):
        """入力テンソルを載せる先のデバイス。"""
        try:
            return next(self.model.parameters()).device
        except Exception:
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")


def read_env_value(name: str, default: str) -> str:
    """環境変数から設定値を読む。"""
    return os.getenv(name, default).strip() or default


def build_base_log_model_id(base_model_id: str) -> str:
    """ベースモデル用のログ分類IDを作る。"""
    return build_model_segment("base", base_model_id)


def create_run_dir(*, base_model_id: str = DEFAULT_BASE_MODEL_ID) -> tuple[str, str]:
    """ベースモデル会話ログ保存用のrunディレクトリを作る。"""
    run_dir, history_file, _ts = create_log_run_dir(
        "dpo_base_chat",
        build_base_log_model_id(base_model_id),
        metadata={"base_model_id": base_model_id},
    )
    return run_dir, history_file


def write_streamlit_session_header(history_file: str, *, base_model_id: str) -> None:
    """Streamlitベースモデル会話ログへセッション情報を残す。"""
    with open(history_file, "a", encoding="utf-8") as file:
        file.write(f"# session_start: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write("# mode: streamlit_base_chat\n")
        file.write(f"# base_model_id: {base_model_id}\n")
        file.write("# thinking: disabled\n")
        file.write("# prompt_template: dpo\n")
        file.write("\n")


def ensure_streamlit_session(base_model_id: str) -> None:
    """Streamlitの会話履歴とログ保存先を初期化する。"""
    if "dpo_base_prompt_history" not in st.session_state:
        st.session_state.dpo_base_prompt_history = []
    if "dpo_base_turns" not in st.session_state:
        st.session_state.dpo_base_turns = []
    if "dpo_base_history_file" not in st.session_state:
        run_dir, history_file = create_run_dir(base_model_id=base_model_id)
        st.session_state.dpo_base_run_dir = run_dir
        st.session_state.dpo_base_history_file = history_file
        write_streamlit_session_header(history_file, base_model_id=base_model_id)


def reset_streamlit_session(base_model_id: str) -> None:
    """Streamlitの会話履歴をリセットし、新しいログを作る。"""
    run_dir, history_file = create_run_dir(base_model_id=base_model_id)
    st.session_state.dpo_base_prompt_history = []
    st.session_state.dpo_base_turns = []
    st.session_state.dpo_base_run_dir = run_dir
    st.session_state.dpo_base_history_file = history_file
    write_streamlit_session_header(history_file, base_model_id=base_model_id)


@st.cache_resource(show_spinner=False)
def load_base_chat_bundle(base_model_id: str) -> BaseChatBundle:
    """LoRA adapterを載せないベースモデルを読み込む。"""
    deps = load_training_modules()
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応GPUが見つかりません。Qwen3.5-27B の実行にはGPU環境が必要です。")

    tokenizer = load_tokenizer(base_model_id, deps)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = deps["ModelClass"].from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if hasattr(model, "config"):
        model.config.use_cache = True
    model.eval()
    return BaseChatBundle(tokenizer=tokenizer, model=model, torch=torch)


def generate_reply(
    bundle: BaseChatBundle,
    user_text: str,
    *,
    history_turns: list[dict[str, str]] | None = None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> str:
    """ベースモデルで返答を生成する。"""
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
    st.set_page_config(page_title="ベースモデル単独チャット", layout="wide")
    st.title("ベースモデル単独チャット")

    with st.sidebar:
        st.header("モデル設定")
        base_model_id = st.text_input(
            "ベースモデル",
            value=read_env_value(DEFAULT_ENV_BASE_MODEL, DEFAULT_BASE_MODEL_ID),
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
        ensure_streamlit_session(base_model_id)
        st.caption(f"ログ出力先: {st.session_state.dpo_base_run_dir}/")
        if st.button("会話をリセット"):
            reset_streamlit_session(base_model_id)
            st.rerun()

    prompt_history = st.session_state.dpo_base_prompt_history
    for turn in st.session_state.dpo_base_turns:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant"):
            st.markdown(turn["ai"])

    user_text = st.chat_input("ベースモデルに送る発話を入力してください")
    if not user_text:
        return

    append_history_line(st.session_state.dpo_base_history_file, "User", user_text)
    with st.chat_message("user"):
        st.markdown(user_text)

    try:
        with st.spinner("モデルを読み込んでいます。初回は時間がかかります。"):
            bundle = load_base_chat_bundle(base_model_id)
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
    st.session_state.dpo_base_turns.append({"user": user_text, "ai": final_reply})
    append_prompt_history_turn(prompt_history, "User", user_text)
    append_prompt_history_turn(prompt_history, "AI", final_reply)
    append_history_line(st.session_state.dpo_base_history_file, "AI(base)", final_reply)

    with st.chat_message("assistant"):
        st.markdown(final_reply)


if __name__ == "__main__":
    render_app()
