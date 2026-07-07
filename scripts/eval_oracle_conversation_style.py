"""会話スタイル模倣のOracle評価CLI。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.oracle_eval_common import (  # noqa: E402
    EvaluationSpec,
    RubricAxis,
    add_common_cli_args,
    run_score_category_cli,
)


SPEC = EvaluationSpec(
    category_key="conversation_style",
    category_title="会話スタイル模倣評価",
    output_subdir="oracle_conversation_style",
    prompt_version="oracle_conversation_style.v1",
    reference_note="PersonaChatのpersona consistency/engagingnessを、ESConv支援者スタイルとの一貫性・類似性に読み替えた評価。",
    axes=(
        RubricAxis(
            key="fluency",
            title="Fluency",
            description="応答が流暢で、会話文として自然に読めるか。",
            high="文法、語順、語調が自然で、相談場面の応答として読みやすい。",
            low="不自然、途切れがち、意味が取りづらい、翻訳調や機械的表現が目立つ。",
        ),
        RubricAxis(
            key="engagingness",
            title="Engagingness",
            description="相談者がさらに話し続けたいと思える応答か。",
            high="相談者に寄り添い、返信しやすく、必要に応じて穏やかに問いかけ、会話を閉じすぎない。",
            low="返答しにくい、一方的、会話が途切れる、相談者の気持ちを広げにくい。",
        ),
        RubricAxis(
            key="style_consistency",
            title="Style Consistency",
            description="会話履歴全体を通してESConvらしい支援対話スタイルを一貫して保てているか。",
            high="共感、受容、探索、整理、助言のバランスが自然で、支援者としての話し方がぶれない。",
            low="支援対話らしさが崩れる、急に一般雑談調・説教調・冷たい調子になる。",
        ),
        RubricAxis(
            key="style_similarity",
            title="Style Similarity",
            description="応答がESConvの支援者応答スタイルにどの程度近いか。",
            high="感情を受け止め、必要に応じて状況を探索し、焦らず支援する自然なトーンと戦略選択。",
            low="DailyDialogの一般雑談に近い、または支援対話としての特徴が弱い。",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(description="ESConv会話スタイル模倣をLLM Oracleで評価します。")
    add_common_cli_args(
        parser,
        default_output_dir=f"artifacts/evaluations/oracle_eval_runs/{SPEC.output_subdir}",
    )
    return parser.parse_args()


def main() -> int:
    """CLIエントリポイント。"""
    return run_score_category_cli(parse_args(), SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
