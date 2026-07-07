"""USR風の一般対話品質Oracle評価CLI。"""

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
    category_key="usr_quality",
    category_title="一般的な対話品質評価",
    output_subdir="oracle_usr_quality",
    prompt_version="oracle_usr_quality.v1",
    reference_note="USRのreference-free品質観点から、知識接地を除いたUnderstandable/Natural/Maintains Context/Interesting/Overallを評価。",
    axes=(
        RubricAxis(
            key="understandable",
            title="Understandable",
            description="応答が理解可能で、意味が通っているか。",
            high="何を伝えたいか明確で、矛盾や意味不明な箇所がない。",
            low="意味が取りづらい、論理が破綻している、曖昧すぎて返答意図が分からない。",
        ),
        RubricAxis(
            key="natural",
            title="Natural",
            description="応答が自然な会話文として書かれているか。",
            high="人間の会話として自然で、過度に硬すぎず、読みやすい。",
            low="不自然、機械的、テンプレート的、文法や語調に違和感がある。",
        ),
        RubricAxis(
            key="maintains_context",
            title="Maintains Context",
            description="会話履歴を踏まえ、文脈を維持しているか。",
            high="直前発話や履歴の内容を正しく踏まえ、話題を保っている。",
            low="文脈を無視する、話題を逸らす、ユーザの意図や感情を誤解している。",
        ),
        RubricAxis(
            key="interesting_or_engaging",
            title="Interesting Or Engaging",
            description="応答が相談者にとって会話を続けやすく、関心を持てるものか。",
            high="返答しやすく、適度に具体的で、会話を自然に続けられる。",
            low="一方的、平板、会話が閉じる、相談者が次に何を返せばよいか分かりにくい。",
        ),
        RubricAxis(
            key="overall_quality",
            title="Overall Quality",
            description="一般的な対話応答としての総合品質。",
            high="理解可能性、自然さ、文脈維持、会話継続性のバランスがよい。",
            low="複数の重要な欠点があり、対話応答として有用性が低い。",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(description="USR風の一般対話品質をLLM Oracleで評価します。")
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
