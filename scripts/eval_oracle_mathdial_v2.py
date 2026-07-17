#!/usr/bin/env python3
"""MathDial原論文とMRBenchに基づくv2 Oracle評価。"""

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
from scripts.eval_oracle_mathdial import GENERAL  # noqa: E402


CORE_V2 = EvaluationSpec(
    category_key="mathdial_pedagogy_v2",
    category_title="MathDial個別指導能力評価 v2",
    output_subdir="pedagogical_v2",
    prompt_version="mathdial_oracle_pedagogy_v2_confirmatory",
    reference_note=(
        "MathDial原論文のequitable tutoringとteacher move taxonomy、"
        "MRBench/NAACL 2025のmistake identification、mistake location、"
        "revealing the answer、providing guidance、actionabilityを、"
        "正答・誤答の両方を含むMathDial held-out文脈向けに事前定義した評価。"
    ),
    axes=(
        RubricAxis(
            key="equitable_tutoring",
            title="Equitable Tutoring",
            description=(
                "学習者に考え、説明し、解法を探索する実質的な余地を与える応答か。"
                "単に答えを隠すことや質問形にすることだけでは高得点にしない。"
            ),
            high=(
                "学習者の現在地点に合う反省、説明、具体的な問い、適切な挑戦を通じ、"
                "学習者自身が推論へ参加できる。"
            ),
            low=(
                "答え・全手順を一方的に渡すか、逆に有用な支援なしで考えさせるだけで、"
                "学習機会を作れていない。"
            ),
            ten_point_guidance=(
                "1〜2: 学習者の推論参加をほぼ奪うか、支援なしで突き放す。\n"
                "3〜4: 質問や励ましはあるが、思考・説明・探索の機会が実質的でない。\n"
                "5〜6: 最低限の学習機会はあるが、一般的または支援量が不均衡。\n"
                "7〜8: 文脈に合う具体的な学習機会を与え、学習者主体の推論を促す。\n"
                "9〜10: 支援と探索余地の均衡がほぼ理想的。10点は改善点がほとんどない場合のみ。"
            ),
        ),
        RubricAxis(
            key="learner_reasoning_diagnosis",
            title="Learner Reasoning Diagnosis",
            description=(
                "直前の学習者推論を、正しい、誤り、不完全、混乱・不確実のいずれかとして"
                "正確に把握して応答しているか。"
            ),
            high=(
                "誤答を正答と肯定せず、正答に存在しない誤りを作らず、"
                "不完全さや混乱も含めて学習者状態を正確に扱う。"
            ),
            low=(
                "誤答への誤った称賛、正答への誤訂正、主要な混乱の見落とし、"
                "または根拠のない状態判断がある。"
            ),
            ten_point_guidance=(
                "1〜2: 正誤を逆に扱うなど、学習者状態の診断が重大に誤る。\n"
                "3〜4: 一部を拾うが主要な誤り・正しい到達点・混乱を見誤る。\n"
                "5〜6: 大枠は合うが、診断が曖昧または表面的。\n"
                "7〜8: 直前の推論状態を正確に把握し、応答へ明確に反映する。\n"
                "9〜10: 正誤、不完全さ、確信度まで精密に把握する。10点はほぼ完全な診断のみ。"
            ),
        ),
        RubricAxis(
            key="mistake_location_and_targeting",
            title="Mistake Location and Targeting",
            description=(
                "誤りがある場合は真正な最初の誤り・概念・計算箇所へ焦点を当てるか。"
                "誤りがない場合は誤りを捏造せず、未完了の次段階や確認点へ焦点を当てるか。"
            ),
            high=(
                "修正すべき具体箇所、または正答後に確認・発展すべき具体箇所が明確で、"
                "応答内容がそこに直接対応する。"
            ),
            low=(
                "誤りの場所を外す、存在しない誤りを指摘する、"
                "すでに理解済みの箇所へ戻る、または一般論だけを返す。"
            ),
            ten_point_guidance=(
                "1〜2: 焦点が誤っており、学習者を誤方向へ導く。\n"
                "3〜4: 関連はあるが、本質的な誤り・未完了点を外す。\n"
                "5〜6: おおむね関連するが、対象箇所が曖昧または広すぎる。\n"
                "7〜8: 真正な誤り・未完了点を具体的に捉えて焦点化する。\n"
                "9〜10: 最初に直すべき箇所を精密に特定し、応答全体をそこへ合わせる。"
            ),
        ),
        RubricAxis(
            key="guidance_quality",
            title="Providing Guidance",
            description=(
                "学習者状態に関連し、数学的に正しく、有用なヒント、説明、例、"
                "または支援質問を提供しているか。"
            ),
            high=(
                "診断した箇所に直接効く正確な支援を、必要十分な具体性で提供する。"
                "質問だけ、説明だけを一律に優先しない。"
            ),
            low=(
                "誤った支援、無関係な説明、曖昧な「もう一度考えて」、"
                "または学習者状態に合わない情報過多・情報不足。"
            ),
            ten_point_guidance=(
                "1〜2: 支援が誤り・無関係で、理解を悪化させる。\n"
                "3〜4: 多少関連するが、不正確、不完全、または使いにくい。\n"
                "5〜6: 正しく最低限役立つが、一般的または調整不足。\n"
                "7〜8: 誤り・未完了点に合う正確で有用な支援。\n"
                "9〜10: 診断との対応、正確性、情報量がほぼ理想的。"
            ),
        ),
        RubricAxis(
            key="feedback_actionability",
            title="Feedback Actionability",
            description=(
                "応答後に学習者が何を考え、計算し、説明し、確認すべきかが明確か。"
                "問題が完了済みなら適切な検証・振り返り・終了へ進めるか。"
            ),
            high=(
                "学習者が自分で実行できる具体的な次の認知行動を一つ以上示し、"
                "応答可能な余地を残す。"
            ),
            low=(
                "次に何をすべきか不明、単なる称賛・訂正・一般論、"
                "または教師が全作業を終えて学習者の行動を残さない。"
            ),
            ten_point_guidance=(
                "1〜2: 次の行動がなく、学習者が推論を継続できない。\n"
                "3〜4: 行動を暗示するが曖昧で、具体的に実行しにくい。\n"
                "5〜6: 実行可能な次の一歩はあるが、一般的または大きすぎる。\n"
                "7〜8: 明確で適量の次の一歩を学習者へ返す。\n"
                "9〜10: 現在の理解から自己修正・検証へ直結するほぼ理想的な行動を示す。"
            ),
        ),
        RubricAxis(
            key="answer_revealing_calibration",
            title="Answer-Revealing Calibration",
            description=(
                "最終解答や残りの全手順を明かす量と時機が、学習者状態に合っているか。"
                "常に答えを隠すことを高評価にはしない。"
            ),
            high=(
                "まだ自力で進める学習者には解答を早く明かさず、"
                "反復して停滞した場合、明示的に説明を求めた場合、または解答済みの場合には、"
                "必要なtelling・説明・確認を適切に使う。"
            ),
            low=(
                "学習機会を奪う早すぎる解答提示、または必要な説明を不当に withheld して"
                "質問だけを反復する。"
            ),
            ten_point_guidance=(
                "1〜2: 明らかに早すぎる全解答、または明らかに必要な説明の拒否。\n"
                "3〜4: 情報量・時機が学習者状態と大きくずれる。\n"
                "5〜6: 大きな破綻はないが、やや教えすぎ・隠しすぎ。\n"
                "7〜8: 学習者状態に応じて情報開示を適切に調整する。\n"
                "9〜10: probing/focus/tellingの切替と情報量がほぼ理想的。"
            ),
        ),
        RubricAxis(
            key="teacher_move_stage_alignment",
            title="Teacher-Move Stage Alignment",
            description=(
                "MathDialのProbing、Focus、Telling、Genericに相当する機能のうち、"
                "現在の学習者状態と会話段階に適切な機能を実現しているか。"
            ),
            high=(
                "概念探索にはProbing、直接進展にはFocus、停滞時の必要部分にはTelling、"
                "導入・確認・終了にはGenericを、文脈に応じて使い分ける。"
            ),
            low=(
                "質問を常に優先する、説明を常に避ける、早すぎるTelling、"
                "停滞中も同じ問いを反復するなど、段階と機能がずれる。"
            ),
            ten_point_guidance=(
                "1〜2: teacher move機能が会話段階と明確に矛盾する。\n"
                "3〜4: 一部機能するが、早すぎる・遅すぎる・反復的。\n"
                "5〜6: 妥当だが一般的で、MathDialの段階適応は弱い。\n"
                "7〜8: 学習者状態と段階に合うteacher move機能が明確。\n"
                "9〜10: move選択、情報量、次状態への接続がほぼ理想的。"
            ),
        ),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="MathDial v2 Oracle評価")
    add_common_cli_args(
        parser,
        default_output_dir="artifacts/mathdial_wildchat/oracle_eval_v2/pedagogical",
    )
    parser.add_argument("--category", choices=("pedagogical", "general"), default="pedagogical")
    args = parser.parse_args()
    return run_score_category_cli(args, CORE_V2 if args.category == "pedagogical" else GENERAL)


if __name__ == "__main__":
    raise SystemExit(main())
