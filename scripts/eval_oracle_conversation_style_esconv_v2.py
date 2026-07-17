"""ESConvらしさ重視の会話スタイルOracle評価CLI。"""

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
    category_key="conversation_style_esconv_v2",
    category_title="ESConv支援スタイル模倣評価 v2",
    output_subdir="oracle_conversation_style_esconv_v2",
    prompt_version="oracle_conversation_style_esconv_v2.v1",
    reference_note=(
        "目的コーパスらしさを重視し、一般的な会話継続性・engagingnessを主スコアから分離した評価。"
    ),
    axes=(
        RubricAxis(
            key="esconv_tone_similarity",
            title="ESConv Tone Similarity",
            description="応答のトーンがESConv支援者らしい共感的・受容的・非断定的な話し方に近いか。",
            high="相談者の感情を丁寧に受け止め、穏やかで非評価的・非断定的な支援者トーンに近い。",
            low="一般雑談調、事務的、説教的、断定的、または感情を十分に受け止めない調子が目立つ。",
            ten_point_guidance=(
                "1〜2: ESConv支援者トーンから大きく外れ、冷たい・説教的・一般雑談的。\n"
                "3〜4: 一部支援的だが、受容や非断定性が弱く、トーンが表面的。\n"
                "5〜6: 支援的ではあるが一般的で、ESConvらしい共感的トーンは中程度。\n"
                "7〜8: 感情を受け止める穏やかな支援者トーンが明確で、文脈に合う。\n"
                "9〜10: 相談者の状態に深く沿い、ESConv支援者応答に非常に近い。10点はほぼ理想的な場合に限る。"
            ),
        ),
        RubricAxis(
            key="supporter_role_consistency",
            title="Supporter Role Consistency",
            description="会話履歴の中で、ESConvの支援者としての役割・態度を一貫して保てているか。",
            high="共感、受容、探索、整理、必要に応じた助言という支援者の役割がぶれずに保たれている。",
            low="急に雑談相手、説教者、問題解決者、評価者のような役割に崩れる。",
            ten_point_guidance=(
                "1〜2: 支援者役割が崩れ、相談支援として不適切な立場や態度が明確。\n"
                "3〜4: 支援的な要素はあるが、雑談調・説教調・問題解決者調へのぶれが目立つ。\n"
                "5〜6: 支援者らしさは最低限あるが、役割一貫性は中程度で一般的。\n"
                "7〜8: 履歴に沿って支援者としての態度を安定して維持している。\n"
                "9〜10: 会話全体に非常によく調和し、ESConv支援者としての役割がほぼ理想的に一貫している。"
            ),
        ),
        RubricAxis(
            key="non_directive_support_style",
            title="Non-directive Support Style",
            description="感情開示や混乱に対して、助言へ急がず、受け止め・整理・探索を優先するESConvらしい支援スタイルか。",
            high="強い感情や不安をまず受容し、必要に応じて穏やかに整理・探索してから次の支援へ進む。",
            low="感情受容を挟まず助言や解決策に飛ぶ、断定する、行動指示を急ぐ。",
            ten_point_guidance=(
                "1〜2: 感情開示に対して受容なく助言・指示・解決策に飛び、ESConvらしい支援過程から外れる。\n"
                "3〜4: 共感や受容は薄く、助言・一般論・判断が前に出やすい。\n"
                "5〜6: 最低限受け止めているが、整理や探索は浅く、支援過程としては中程度。\n"
                "7〜8: 助言に急がず、受容・整理・探索を優先するESConvらしい流れが明確。\n"
                "9〜10: 相談者の感情段階に非常によく沿い、非指示的支援スタイルがほぼ理想的。"
            ),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(description="ESConv支援スタイル模倣v2をLLM Oracleで評価します。")
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
