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
            ten_point_guidance=(
                "1〜2: 意味が取りづらい、文が破綻している、相談応答として成立しにくい。\n"
                "3〜4: 読めるが不自然さや機械的表現が目立ち、会話の流れを妨げる。\n"
                "5〜6: 大きな破綻はないが、一般的・硬い・ぎこちない表現が残る。\n"
                "7〜8: 自然で読みやすく、相談場面の応答として十分に滑らか。\n"
                "9〜10: 語調、文のつながり、間合いが非常に自然。10点はほぼ人間らしく理想的な流暢さに限る。"
            ),
        ),
        RubricAxis(
            key="engagingness",
            title="Engagingness",
            description="相談者がさらに話し続けたいと思える応答か。",
            high="相談者に寄り添い、返信しやすく、必要に応じて穏やかに問いかけ、会話を閉じすぎない。",
            low="返答しにくい、一方的、会話が途切れる、相談者の気持ちを広げにくい。",
            ten_point_guidance=(
                "1〜2: 一方的、拒絶的、または会話を止める応答で、相談者が続けにくい。\n"
                "3〜4: 返信は可能だが、寄り添いが弱く、問いかけや受け止めが表面的。\n"
                "5〜6: 最低限会話は続くが、一般的で相談者の話を深める力は中程度。\n"
                "7〜8: 気持ちを受け止め、自然に次の発話を促し、相談者が続けやすい。\n"
                "9〜10: 相談者の感情や状況に深く沿い、安心して話を広げられる。10点はほぼ理想的な促しに限る。"
            ),
        ),
        RubricAxis(
            key="style_consistency",
            title="Style Consistency",
            description="会話履歴全体を通してESConvらしい支援対話スタイルを一貫して保てているか。",
            high="共感、受容、探索、整理、助言のバランスが自然で、支援者としての話し方がぶれない。",
            low="支援対話らしさが崩れる、急に一般雑談調・説教調・冷たい調子になる。",
            ten_point_guidance=(
                "1〜2: 支援対話スタイルから大きく外れ、冷たい・説教的・雑談的な崩れが明確。\n"
                "3〜4: 一部支援的だが、トーンや戦略が不安定で文脈との一貫性が弱い。\n"
                "5〜6: 支援的ではあるが一般的で、共感・探索・整理のバランスは中程度。\n"
                "7〜8: 履歴に沿って共感的・受容的な支援者スタイルを明確に維持している。\n"
                "9〜10: 会話全体に非常によく調和し、支援者としての姿勢が一貫している。10点はほぼ理想的な一貫性に限る。"
            ),
        ),
        RubricAxis(
            key="style_similarity",
            title="Style Similarity",
            description="応答がESConvの支援者応答スタイルにどの程度近いか。",
            high="感情を受け止め、必要に応じて状況を探索し、焦らず支援する自然なトーンと戦略選択。",
            low="DailyDialogの一般雑談に近い、または支援対話としての特徴が弱い。",
            ten_point_guidance=(
                "1〜2: ESConv支援者応答とは大きく異なり、一般雑談・断定・早すぎる助言が中心。\n"
                "3〜4: 支援対話らしさは弱く、感情の受け止めや探索が表面的。\n"
                "5〜6: 支援的ではあるが一般的で、ESConvらしさは中程度。\n"
                "7〜8: 感情受容、穏やかな探索、必要に応じた整理があり、ESConvらしさが明確。\n"
                "9〜10: 相談者の状態に深く沿い、ESConvの支援者応答に非常に近い。10点はほぼ理想的な類似性に限る。"
            ),
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
