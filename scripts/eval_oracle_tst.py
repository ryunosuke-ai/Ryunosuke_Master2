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
        ),
        RubricAxis(
            key="content_preservation",
            title="Content Preservation",
            description="会話履歴やユーザ発話の内容を保ち、文脈に合った応答をしているか。",
            high="ユーザの悩み、状況、感情を正しく踏まえ、話題を逸らさず、過剰な追加をしない応答。",
            low="文脈と関係ない、悩みを誤解する、一般論だけで具体的文脈を反映しない応答。",
        ),
        RubricAxis(
            key="naturalness",
            title="Naturalness",
            description="応答が自然で読みやすく、言語的に破綻していないか。",
            high="文法的に自然で読みやすく、会話として違和感がなく、過度に冗長でない応答。",
            low="不自然な表現、文の破綻、機械的・テンプレート的すぎる表現が目立つ応答。",
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
