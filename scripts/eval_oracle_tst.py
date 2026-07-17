"""テキストスタイル転移のOracle評価CLI。"""

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
    category_key="text_style_transfer",
    category_title="テキストスタイル転移評価",
    output_subdir="oracle_tst",
    prompt_version="oracle_tst.v1",
    reference_note="Mir et al. (2019) のstyle strength/content preservation/naturalnessをESConv支援対話に読み替えた評価。",
    axes=(
        RubricAxis(
            key="style_strength",
            title="Style Strength",
            description="応答がESConvらしい感情支援対話スタイルをどの程度反映しているか。",
            high="相談者の感情を受け止め、共感的・受容的で、すぐに解決策を押し付けない温かい応答。",
            low="一般雑談的、事務的、冷たい、感情を無視する、早すぎる助言や断定が目立つ応答。",
            ten_point_guidance=(
                "1〜2: 感情支援スタイルがほぼなく、冷たい・事務的・断定的・文脈無視が明確。\n"
                "3〜4: 支援的な語はあるが表面的で、感情の受け止めや受容が弱い。\n"
                "5〜6: 支援的ではあるが一般的で、ESConvらしさは中程度。\n"
                "7〜8: 感情を受け止め、共感的・受容的で、ESConvらしさが明確。\n"
                "9〜10: 相談者の感情や状況を深く受け止め、助言に急がず、ESConv支援者応答に非常に近い。10点はほぼ理想的な場合に限る。"
            ),
        ),
        RubricAxis(
            key="content_preservation",
            title="Content Preservation",
            description="会話履歴やユーザ発話の内容を保ち、文脈に合った応答をしているか。",
            high="ユーザの悩み、状況、感情を正しく踏まえ、話題を逸らさず、過剰な追加をしない応答。",
            low="文脈と関係ない、悩みを誤解する、一般論だけで具体的文脈を反映しない応答。",
            ten_point_guidance=(
                "1〜2: 文脈を大きく誤解する、話題を逸らす、ユーザ発話と矛盾する。\n"
                "3〜4: 一部は合うが、重要な感情・状況を拾えておらず一般論が目立つ。\n"
                "5〜6: 大枠は保つが浅く、具体的な悩みや感情への反映は中程度。\n"
                "7〜8: ユーザの状況・感情を正しく踏まえ、話題を保った自然な応答。\n"
                "9〜10: 履歴の細部と感情の両方を的確に保持し、過剰な追加も少ない。10点はほぼ理想的な文脈保持に限る。"
            ),
        ),
        RubricAxis(
            key="naturalness",
            title="Naturalness",
            description="応答が自然で読みやすく、言語的に破綻していないか。",
            high="文法的に自然で読みやすく、会話として違和感がなく、過度に冗長でない応答。",
            low="不自然な表現、文の破綻、機械的・テンプレート的すぎる表現が目立つ応答。",
            ten_point_guidance=(
                "1〜2: 文が破綻している、意味が通らない、会話応答として成立しにくい。\n"
                "3〜4: 理解はできるが、不自然・機械的・テンプレート的な表現が目立つ。\n"
                "5〜6: 大きな破綻はないが、やや硬い・冗長・一般的な表現が残る。\n"
                "7〜8: 自然で読みやすく、会話として違和感が少ない。\n"
                "9〜10: 語調、簡潔さ、文脈への自然な接続が非常に優れている。10点はほぼ理想的な自然さに限る。"
            ),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(description="ESConvらしいテキストスタイル転移をLLM Oracleで評価します。")
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
