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
            ten_point_guidance=(
                "1〜2: 意味不明、矛盾、文脈上成立しない内容が明確。\n"
                "3〜4: 大意は取れるが曖昧さや論理の飛躍が目立つ。\n"
                "5〜6: 最低限理解できるが、意図が一般的・弱い・やや不明瞭。\n"
                "7〜8: 意図が明確で、文脈上も理解しやすい。\n"
                "9〜10: 伝えたいことが非常に明確で、矛盾や曖昧さがほぼない。10点はほぼ理想的な明瞭さに限る。"
            ),
        ),
        RubricAxis(
            key="natural",
            title="Natural",
            description="応答が自然な会話文として書かれているか。",
            high="人間の会話として自然で、過度に硬すぎず、読みやすい。",
            low="不自然、機械的、テンプレート的、文法や語調に違和感がある。",
            ten_point_guidance=(
                "1〜2: 文法や語調が大きく不自然で、会話文として成立しにくい。\n"
                "3〜4: 不自然・機械的・テンプレート的な表現が目立つ。\n"
                "5〜6: 大きな破綻はないが、硬さ・冗長さ・一般的な言い回しが残る。\n"
                "7〜8: 人間の会話として自然で、読みやすく違和感が少ない。\n"
                "9〜10: 文体と語調が非常に自然で、相談場面にもよく合う。10点はほぼ理想的な自然さに限る。"
            ),
        ),
        RubricAxis(
            key="maintains_context",
            title="Maintains Context",
            description="会話履歴を踏まえ、文脈を維持しているか。",
            high="直前発話や履歴の内容を正しく踏まえ、話題を保っている。",
            low="文脈を無視する、話題を逸らす、ユーザの意図や感情を誤解している。",
            ten_point_guidance=(
                "1〜2: 文脈を無視する、話題が大きく逸れる、ユーザ意図を明確に誤解する。\n"
                "3〜4: 一部文脈に触れるが、重要な感情や状況を拾えていない。\n"
                "5〜6: 大枠は維持するが、具体性や履歴への接続は中程度。\n"
                "7〜8: 直前発話と履歴を正しく踏まえ、自然に話題を保っている。\n"
                "9〜10: 履歴の細部とユーザ意図を非常によく踏まえている。10点はほぼ理想的な文脈維持に限る。"
            ),
        ),
        RubricAxis(
            key="interesting_or_engaging",
            title="Interesting Or Engaging",
            description="応答が相談者にとって会話を続けやすく、関心を持てるものか。",
            high="返答しやすく、適度に具体的で、会話を自然に続けられる。",
            low="一方的、平板、会話が閉じる、相談者が次に何を返せばよいか分かりにくい。",
            ten_point_guidance=(
                "1〜2: 一方的・閉鎖的で、相談者が次に返しにくい。\n"
                "3〜4: 会話は続けられるが、平板で関心を引きにくく、促しが弱い。\n"
                "5〜6: 最低限返答しやすいが、一般的で深まりは中程度。\n"
                "7〜8: 適度に具体的で、相談者が自然に話を続けやすい。\n"
                "9〜10: 相談者の関心や感情に沿い、非常に自然に次の発話を促す。10点はほぼ理想的な会話継続性に限る。"
            ),
        ),
        RubricAxis(
            key="overall_quality",
            title="Overall Quality",
            description="一般的な対話応答としての総合品質。",
            high="理解可能性、自然さ、文脈維持、会話継続性のバランスがよい。",
            low="複数の重要な欠点があり、対話応答として有用性が低い。",
            ten_point_guidance=(
                "1〜2: 複数の重大な欠点があり、対話応答として有用性が非常に低い。\n"
                "3〜4: 一部は使えるが、理解可能性・自然さ・文脈維持の不足が目立つ。\n"
                "5〜6: 最低限の品質はあるが、一般的・浅い・改善点が多い。\n"
                "7〜8: 主要な品質要素のバランスがよく、明確に良い対話応答。\n"
                "9〜10: 文脈適合性、自然さ、会話継続性の総合バランスが非常に高い。10点はほぼ理想的な応答に限る。"
            ),
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
