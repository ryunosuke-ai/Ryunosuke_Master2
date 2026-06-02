"""Azure OpenAI の GPT-5.4 と GPT-5.4 pro を比較するStreamlitチャットアプリ。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    st.error("`python-dotenv` パッケージが見つかりません。`python3 -m pip install python-dotenv` を実行してください。")
    st.stop()

try:
    from openai import AzureOpenAI
except ModuleNotFoundError:
    st.error("`openai` パッケージが見つかりません。`python3 -m pip install openai` を実行してください。")
    st.stop()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.log_manager import build_model_segment, create_log_run_dir  # noqa: E402


load_dotenv()

DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_REASONING_EFFORT = "medium"
REASONING_EFFORTS = ("medium", "high", "xhigh")

SYSTEM_INSTRUCTIONS = (
    "あなたは日本語で自然に会話するチャットボットです。"
    "これまでの会話履歴を踏まえて、文脈に合う返答をしてください。"
    "返答は簡潔にし、必要な場合だけ質問を1つ添えてください。"
)


@dataclass(frozen=True)
class BotConfig:
    """モデルごとの設定。"""

    state_prefix: str
    title: str
    code_id: str
    api_key_env: str
    fallback_api_key_env: str
    deployment_env: str
    fallback_deployment_env: str
    default_deployment: str


GPT54_CONFIG = BotConfig(
    state_prefix="openai_gpt54",
    title="GPT-5.4",
    code_id="azure_openai_gpt54_chat",
    api_key_env="AZURE_OPENAI_GPT54_API_KEY",
    fallback_api_key_env="OPENAI_GPT54_API_KEY",
    deployment_env="AZURE_OPENAI_GPT54_DEPLOYMENT_NAME",
    fallback_deployment_env="OPENAI_GPT54_MODEL",
    default_deployment="gpt-5.4",
)

GPT54_PRO_CONFIG = BotConfig(
    state_prefix="openai_gpt54_pro",
    title="GPT-5.4 pro",
    code_id="azure_openai_gpt54_pro_chat",
    api_key_env="AZURE_OPENAI_GPT54_PRO_API_KEY",
    fallback_api_key_env="OPENAI_GPT54_PRO_API_KEY",
    deployment_env="AZURE_OPENAI_GPT54_PRO_DEPLOYMENT_NAME",
    fallback_deployment_env="OPENAI_GPT54_PRO_MODEL",
    default_deployment="gpt-5.4-pro",
)


def read_env_value(name: str, default: str = "") -> str:
    """環境変数から設定値を読む。"""
    return os.getenv(name, default).strip() or default


def read_env_value_with_fallback(name: str, fallback_name: str, default: str = "") -> str:
    """新しい環境変数名を優先し、旧名があればフォールバックする。"""
    primary = read_env_value(name)
    if primary:
        return primary
    return read_env_value(fallback_name, default)


def build_azure_responses_target_url(azure_endpoint: str, api_version: str) -> str:
    """Azure OpenAI Responses APIのターゲットURLを表示用に作る。"""
    endpoint = azure_endpoint.strip().rstrip("/")
    return f"{endpoint}/openai/responses?api-version={api_version}"


def append_history_line(history_file: str, role: str, text: str) -> None:
    """履歴ファイルへ1行追記する。"""
    safe_text = str(text).replace("\n", " ").strip()
    if not safe_text:
        return
    with open(history_file, "a", encoding="utf-8") as file:
        file.write(f"[{datetime.now().strftime('%H:%M:%S')}] {role}: {safe_text}\n")


def create_openai_run_dir(config: BotConfig, deployment: str) -> tuple[str, str]:
    """モデル別の会話ログ保存先を作る。"""
    run_dir, history_file, _ts = create_log_run_dir(
        config.code_id,
        build_model_segment("azure_openai", deployment),
        metadata={
            "provider": "azure_openai",
            "api": "responses",
            "deployment": deployment,
        },
    )
    return run_dir, history_file


def write_session_header(history_file: str, *, config: BotConfig, deployment: str) -> None:
    """会話ログへセッション情報を残す。"""
    with open(history_file, "a", encoding="utf-8") as file:
        file.write(f"# session_start: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write("# mode: streamlit_azure_openai_compare_chat\n")
        file.write(f"# bot: {config.title}\n")
        file.write("# provider: azure_openai\n")
        file.write("# api: responses\n")
        file.write(f"# deployment: {deployment}\n")
        file.write("\n")


def ensure_bot_session(config: BotConfig, deployment: str) -> None:
    """モデル別のStreamlit状態を初期化する。"""
    turns_key = f"{config.state_prefix}_turns"
    run_dir_key = f"{config.state_prefix}_run_dir"
    history_file_key = f"{config.state_prefix}_history_file"
    deployment_key = f"{config.state_prefix}_deployment"

    if turns_key not in st.session_state:
        st.session_state[turns_key] = []

    if history_file_key not in st.session_state or st.session_state.get(deployment_key) != deployment:
        run_dir, history_file = create_openai_run_dir(config, deployment)
        st.session_state[run_dir_key] = run_dir
        st.session_state[history_file_key] = history_file
        st.session_state[deployment_key] = deployment
        write_session_header(history_file, config=config, deployment=deployment)


def reset_bot_session(config: BotConfig, deployment: str) -> None:
    """指定モデルの会話履歴だけをリセットする。"""
    turns_key = f"{config.state_prefix}_turns"
    run_dir_key = f"{config.state_prefix}_run_dir"
    history_file_key = f"{config.state_prefix}_history_file"
    deployment_key = f"{config.state_prefix}_deployment"

    run_dir, history_file = create_openai_run_dir(config, deployment)
    st.session_state[turns_key] = []
    st.session_state[run_dir_key] = run_dir
    st.session_state[history_file_key] = history_file
    st.session_state[deployment_key] = deployment
    write_session_header(history_file, config=config, deployment=deployment)


def build_response_input(turns: list[dict[str, str]], user_text: str) -> list[dict[str, str]]:
    """Responses APIへ渡す会話履歴を作る。"""
    messages: list[dict[str, str]] = []
    for turn in turns:
        user = str(turn.get("user", "")).strip()
        assistant = str(turn.get("assistant", "")).strip()
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_text})
    return messages


def generate_reply(
    *,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
    deployment: str,
    turns: list[dict[str, str]],
    user_text: str,
    max_output_tokens: int,
    reasoning_effort: str,
) -> str:
    """Azure OpenAI Responses APIで返答を生成する。"""
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )
    response = client.responses.create(
        model=deployment,
        instructions=SYSTEM_INSTRUCTIONS,
        input=build_response_input(turns, user_text),
        max_output_tokens=max_output_tokens,
        reasoning={"effort": reasoning_effort},
    )
    return (response.output_text or "").strip()


def render_bot_column(
    config: BotConfig,
    *,
    max_output_tokens: int,
    reasoning_effort: str,
) -> None:
    """1モデル分のチャットUIを描画する。"""
    api_key = read_env_value_with_fallback(config.api_key_env, config.fallback_api_key_env)
    azure_endpoint = read_env_value("AZURE_OPENAI_ENDPOINT")
    api_version = read_env_value("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    deployment = read_env_value_with_fallback(
        config.deployment_env,
        config.fallback_deployment_env,
        config.default_deployment,
    )

    ensure_bot_session(config, deployment)

    turns_key = f"{config.state_prefix}_turns"
    run_dir_key = f"{config.state_prefix}_run_dir"
    history_file_key = f"{config.state_prefix}_history_file"
    input_key = f"{config.state_prefix}_input"

    st.subheader(config.title)
    st.caption(f"deployment: `{deployment}` / API: `Azure OpenAI Responses`")
    if azure_endpoint:
        st.caption(f"target: `{build_azure_responses_target_url(azure_endpoint, api_version)}`")
    st.caption(f"ログ出力先: `{st.session_state[run_dir_key]}/`")

    missing_envs = []
    if not api_key:
        missing_envs.append(f"{config.api_key_env} または {config.fallback_api_key_env}")
    if not azure_endpoint:
        missing_envs.append("AZURE_OPENAI_ENDPOINT")
    if not api_version:
        missing_envs.append("AZURE_OPENAI_API_VERSION")
    if missing_envs:
        st.error(f"`.env` の設定が不足しています: {', '.join(missing_envs)}")

    if st.button("会話をリセット", key=f"{config.state_prefix}_reset"):
        reset_bot_session(config, deployment)
        st.rerun()

    for turn in st.session_state[turns_key]:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant"):
            st.markdown(turn["assistant"])

    with st.form(f"{config.state_prefix}_form", clear_on_submit=True):
        user_text = st.text_area(
            f"{config.title} への入力",
            key=input_key,
            height=100,
            placeholder="ここにメッセージを入力してください",
        )
        submitted = st.form_submit_button("送信", disabled=bool(missing_envs))

    if not submitted:
        return

    clean_text = user_text.strip()
    if not clean_text:
        st.warning("メッセージを入力してください。")
        return

    append_history_line(st.session_state[history_file_key], "User", clean_text)

    try:
        with st.spinner(f"{config.title} が返答を生成しています。"):
            reply = generate_reply(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                deployment=deployment,
                turns=st.session_state[turns_key],
                user_text=clean_text,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
            )
    except Exception as exc:
        st.error(f"生成に失敗しました: {exc}")
        return

    final_reply = reply or "（空の返答）"
    st.session_state[turns_key].append({"user": clean_text, "assistant": final_reply})
    append_history_line(st.session_state[history_file_key], "AI", final_reply)
    st.rerun()


def render_app() -> None:
    """Streamlit UIを描画する。"""
    st.set_page_config(page_title="Azure OpenAI GPT-5.4 / GPT-5.4 pro 出力確認チャット", layout="wide")
    st.title("Azure OpenAI GPT-5.4 / GPT-5.4 pro 出力確認チャット")

    with st.sidebar:
        st.header("共通生成設定")
        max_output_tokens = st.slider(
            "max_output_tokens",
            min_value=128,
            max_value=4096,
            value=DEFAULT_MAX_OUTPUT_TOKENS,
            step=128,
        )
        reasoning_effort = st.selectbox(
            "reasoning.effort",
            REASONING_EFFORTS,
            index=REASONING_EFFORTS.index(DEFAULT_REASONING_EFFORT),
        )
        st.caption("2つのモデルは同じ設定値で呼び出します。会話履歴と入力はモデルごとに独立しています。")
        st.caption(f"Azure API version: `{read_env_value('AZURE_OPENAI_API_VERSION', '2025-04-01-preview')}`")

    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        render_bot_column(
            GPT54_CONFIG,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
    with right_col:
        render_bot_column(
            GPT54_PRO_CONFIG,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )


if __name__ == "__main__":
    render_app()
