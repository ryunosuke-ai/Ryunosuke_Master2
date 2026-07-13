#!/usr/bin/env python3
"""MathDial日本語応答を10段階Oracle評価する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.oracle_eval_common import EvaluationSpec, RubricAxis, add_common_cli_args, run_score_category_cli


GUIDANCE = "1〜2: ほぼ満たさない。3〜4: 重要な不足がある。5〜6: 最低限だが一般的。7〜8: 明確に良い。9〜10: ほぼ理想的で、10は改善点がほとんどない場合に限る。"

PEDAGOGICAL = EvaluationSpec(
    category_key="mathdial_pedagogy_v1", category_title="MathDial個別指導スタイル", output_subdir="pedagogical", prompt_version="mathdial_oracle_pedagogy_v1",
    reference_note="MathDial、MRBench、mistake remediation研究をMathDial向けに再定義した評価。",
    axes=tuple(RubricAxis(key=key, title=title, description=description, high=high, low=low, ten_point_guidance=GUIDANCE) for key, title, description, high, low in (
        ("tutoring_style_strength", "Tutoring Style Strength", "答えを渡すだけでなく学習者を個別に導くMathDial型の指導スタイルか。", "学習者の状態を拾い、診断、質問、ヒント、確認を適切に選ぶ。", "解答提示だけ、一般論、または一方的説明に留まる。"),
        ("misconception_diagnosis", "Misconception Diagnosis", "直前の誤り・混乱を正確に認識し、必要な箇所へ焦点を当てるか。", "実際の誤りや不足を特定し、それに応答する。", "誤りを見逃す、存在しない誤りを決めつける、焦点がずれる。"),
        ("scaffolding_quality", "Scaffolding Quality", "学習者が自力で次へ進める適量の質問・ヒント・説明か。", "次の一歩が明確で、支援量が段階的かつ適切。", "曖昧すぎるか、全手順・答えを一度に与える。"),
        ("premature_answer_avoidance", "Premature Answer Avoidance", "十分な診断や試行前に最終解答を明かしていないか。", "必要な場合を除き答えを保留し、自己解決を促す。", "学習機会を奪う形で答えや全手順を即座に提示する。"),
        ("pedagogical_transition_plausibility", "Pedagogical Transition", "現在の学習者状態から次の理解状態へ進める戦略として自然か。", "会話段階と理解状態に合う次の指導手を選ぶ。", "段階を飛ばす、停滞させる、または会話を不自然に閉じる。"),
        ("teacher_move_alignment", "Teacher Move Alignment", "文脈に対してprobing/focus/telling/generic相当の機能を適切に選ぶか。", "必要なteacher move機能が明確で文脈に合う。", "戦略機能が不明確または文脈と不一致。"),
        ("learner_self_correction_support", "Self-correction Support", "学習者が誤りに気づき自分で修正する余地を作るか。", "考えを言語化・再検討できる具体的な働きかけがある。", "受動的に答えを受け取らせるだけ。"),
    )),
)

GENERAL = EvaluationSpec(
    category_key="mathdial_general_quality_v1", category_title="MathDial一般品質", output_subdir="general", prompt_version="mathdial_oracle_general_v1", reference_note="目的スタイルと一般品質のトレードオフを分離する。",
    axes=tuple(RubricAxis(key=key, title=title, description=description, high=high, low=low, ten_point_guidance=GUIDANCE) for key, title, description, high, low in (
        ("correctness", "Correctness", "問題・履歴・ground truthに照らして数学的に正しいか。", "誤った主張をせず、必要な内容が正しい。", "計算、概念、問題条件に誤りがある。"),
        ("understandable", "Understandable", "学習者が理解できる明瞭さか。", "簡潔で構造が明確。", "曖昧、飛躍、過度に複雑。"),
        ("natural_japanese", "Natural Japanese", "日本人の個別指導会話として自然な日本語か。", "自然で適切な語調。", "翻訳調、不自然、役割表記混入。"),
        ("maintains_context", "Maintains Context", "問題と直前までの履歴を維持しているか。", "具体的内容を正しく参照する。", "話題逸脱や履歴矛盾がある。"),
        ("overall_quality", "Overall Quality", "一般的な次の教師応答として総合的に良いか。", "正確、明瞭、自然で文脈に合う。", "重大な正確性・明瞭性・文脈上の問題がある。"),
    )),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="MathDial Oracle評価")
    add_common_cli_args(parser, default_output_dir="artifacts/mathdial_wildchat/oracle_eval/pedagogical")
    parser.add_argument("--category", choices=("pedagogical", "general"), default="pedagogical")
    args = parser.parse_args()
    return run_score_category_cli(args, PEDAGOGICAL if args.category == "pedagogical" else GENERAL)


if __name__ == "__main__":
    raise SystemExit(main())
