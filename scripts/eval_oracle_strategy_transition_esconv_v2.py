"""ESConv支援過程重視の戦略・状態遷移Oracle評価CLI。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.oracle_eval_common import (  # noqa: E402
    RubricAxis,
    StrategyEvaluationSpec,
    add_strategy_cli_args,
    run_strategy_cli,
)


SPEC = StrategyEvaluationSpec(
    category_key="strategy_transition_esconv_v2",
    category_title="ESConv支援過程としての戦略・状態遷移評価 v2",
    output_subdir="oracle_strategy_transition_esconv_v2",
    prompt_version="oracle_strategy_transition_esconv_v2.v1",
    reference_note=(
        "一般的な会話継続性ではなく、ESConvでよく見られる支援プロセスとしての戦略選択・状態遷移を評価する。"
    ),
    score_axes=(
        RubricAxis(
            key="strategy_stage_alignment",
            title="Strategy Stage Alignment",
            description="相談者状態に対して、ESConvでよく見られる支援戦略の段階に合っているか。",
            high="感情開示・混乱・状況説明・解決検討などの状態に応じ、共感、探索、整理、助言を適切な段階で選んでいる。",
            low="相談者の段階に合わない戦略を選び、支援過程として早すぎる・遅すぎる・ずれている。",
            ten_point_guidance=(
                "1〜2: 相談者状態と戦略段階が大きくずれ、ESConv支援過程として不適切。\n"
                "3〜4: 一部合うが、状態に対する戦略選択が表面的または段階違い。\n"
                "5〜6: 最低限は合うが一般的で、ESConvの支援段階としては中程度。\n"
                "7〜8: 相談者状態に合う支援戦略を明確に選び、ESConvらしい段階に沿う。\n"
                "9〜10: 状態把握と戦略段階が非常によく合い、ほぼ理想的なESConv支援過程になっている。"
            ),
        ),
        RubricAxis(
            key="premature_advice_avoidance",
            title="Premature Advice Avoidance",
            description="感情開示・混乱・強い不安に対して、共感・受容・探索を挟まず助言へ飛んでいないか。",
            high="強い感情にはまず共感・受容・整理を行い、助言や行動提案を急がない。",
            low="相談者の感情を十分に受け止めず、一般論、指示、助言、解決策を早く出しすぎる。",
            ten_point_guidance=(
                "1〜2: 感情開示に対し受容なく助言・指示へ飛び、支援過程として明確に不適切。\n"
                "3〜4: 受け止めが弱く、助言や解決策が早すぎる印象が強い。\n"
                "5〜6: 最低限の共感はあるが、支援過程としてはやや急ぎ気味または一般的。\n"
                "7〜8: 助言へ急がず、共感・受容・探索を適切に挟んでいる。\n"
                "9〜10: 相談者の感情段階に非常によく沿い、助言のタイミングがほぼ理想的。"
            ),
        ),
        RubricAxis(
            key="esconv_transition_plausibility",
            title="ESConv Transition Plausibility",
            description="応答前状態、応答戦略、応答後状態の遷移がESConvらしい支援過程として自然か。",
            high="before -> strategy -> after が、感情受容、整理、探索、解決検討へ進むESConv的な支援過程として妥当。",
            low="一般的には会話が続いても、ESConvの支援過程としては唐突、段階飛ばし、または状態変化が不自然。",
            ten_point_guidance=(
                "1〜2: 状態遷移がESConv支援過程として破綻している、または段階を大きく飛ばしている。\n"
                "3〜4: 遷移の一部は理解できるが、感情受容・整理・探索の流れが弱く唐突。\n"
                "5〜6: 大きな破綻はないが、ESConvらしい支援過程としては一般的で中程度。\n"
                "7〜8: before -> strategy -> after が支援過程として自然で、ESConvらしさが明確。\n"
                "9〜10: 相談者状態の変化が非常に妥当で、ESConvの支援プロセスにほぼ理想的に沿っている。"
            ),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(description="ESConv支援過程v2の戦略・状態遷移をLLM Oracleで評価します。")
    add_strategy_cli_args(
        parser,
        default_output_dir="artifacts/evaluations/oracle_eval_runs/oracle_strategy_transition_esconv_v2",
    )
    return parser.parse_args()


def main() -> int:
    """CLIエントリポイント。"""
    return run_strategy_cli(parse_args(), SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
