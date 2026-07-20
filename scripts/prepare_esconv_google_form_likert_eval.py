#!/usr/bin/env python3
"""ESConvの3モデル人手評価を7段階Likert形式で整形する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

# ファイルを直接実行した場合もリポジトリ内のscripts packageを解決する。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_oracle_three_model_significance import (
    effect_size,
    friedman_test,
    holm_adjust,
    paired_permutation_p,
    win_tie_loss,
)
from scripts.prepare_esconv_google_form_eval import (
    DEFAULT_TOPCONF_RUN,
    DEFAULT_V2_RUN,
    FORM_VERSIONS,
    MODEL_KEYS,
    MODEL_ORACLE_KEYS,
    QUESTIONS,
    REPRESENTATIVE_AXES,
    build_candidates,
    format_history,
    load_axis_scores,
    select_oracle_enriched,
    sha256_file,
    version_orders,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = Path(
    "artifacts/user_eval/google_forms/esconv_oracle_enriched_likert_v2"
)
DEFAULT_SEED = 20260720
LIKERT_COLUMNS = ("1", "2", "3", "4", "5", "6", "7")
LIKERT_ANCHORS = {
    "1": "全く当てはまらない",
    "2": "当てはまらない",
    "3": "あまり当てはまらない",
    "4": "どちらともいえない",
    "5": "やや当てはまる",
    "6": "当てはまる",
    "7": "非常によく当てはまる",
}
LIKERT_STATEMENTS = (
    {
        "key": "style_strength",
        "statement": "相談者を支える応答として、全体的に良い。",
    },
    {
        "key": "esconv_tone_similarity",
        "statement": "相談者の気持ちを受け止め、やさしく話している。",
    },
    {
        "key": "supporter_role_consistency",
        "statement": "これまでの会話に合った、支える立場の話し方を続けている。",
    },
    {
        "key": "non_directive_support_style",
        "statement": (
            "相手の話を理解・整理しようとし、指示や結論を"
            "すぐに押しつけていない。"
        ),
    },
    {
        "key": "premature_advice_avoidance",
        "statement": (
            "この会話の段階に対して、助言や提案を出すタイミングが適切である。"
        ),
    },
    {
        "key": "content_preservation",
        "statement": "これまでの話の内容によく合っている。",
    },
    {
        "key": "naturalness",
        "statement": "日本語の会話として自然で読みやすい。",
    },
)
FINAL_CHOICE_QUESTION = (
    "3つの応答のうち、相談者の気持ちを受け止め、"
    "会話の状況に合わせて支える応答として、"
    "最もふさわしいものはどれですか。"
)
FINAL_CHOICE_OPTIONS = (
    "応答A",
    "応答B",
    "応答C",
    "ほぼ同じ",
    "判断できない",
)


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="ESConvのGoogle Form用7段階Likert評価データを作成します。"
    )
    parser.add_argument("--v2-run", type=Path, default=DEFAULT_V2_RUN)
    parser.add_argument("--topconf-run", type=Path, default=DEFAULT_TOPCONF_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--diagnostic-permutations", type=int, default=100_000)
    return parser.parse_args()


def public_record(
    row: dict[str, Any],
    *,
    item_number: int,
    order: tuple[str, str, str],
) -> dict[str, Any]:
    """モデル名を除いたLikert評価レコードを作る。"""
    return {
        "item_id": f"item_{item_number:02d}",
        "item_number": item_number,
        "conversation": format_history(row["history"], row["prompt"]),
        "response_a": row["responses"][order[0]],
        "response_b": row["responses"][order[1]],
        "response_c": row["responses"][order[2]],
        "likert_statements": list(LIKERT_STATEMENTS),
        "likert_columns": list(LIKERT_COLUMNS),
        "likert_anchors": dict(LIKERT_ANCHORS),
        "final_choice_question": FINAL_CHOICE_QUESTION,
        "final_choice_options": list(FINAL_CHOICE_OPTIONS),
    }


def private_record(
    row: dict[str, Any],
    *,
    item_number: int,
    order: tuple[str, str, str],
) -> dict[str, Any]:
    """回答収集後の復号と監査に使う非公開レコードを作る。"""
    return {
        "item_id": f"item_{item_number:02d}",
        "prompt_id": row["prompt_id"],
        "category": row["category"],
        "selection_type": "oracle_enriched_category_stratified_posthoc",
        "position_to_model": {
            "A": order[0],
            "B": order[1],
            "C": order[2],
        },
        "representative_axis_scores": {
            model_key: {
                axis: row["oracle_axis_scores"][MODEL_ORACLE_KEYS[model_key]][axis]
                for axis in REPRESENTATIVE_AXES
            }
            for model_key in MODEL_KEYS
        },
        "representative_means": row["representative_means"],
        "basis_advantage_over_best_control": row[
            "basis_advantage_over_best_control"
        ],
        "response_sha256": {
            model_key: hashlib.sha256(
                row["responses"][model_key].encode("utf-8")
            ).hexdigest()
            for model_key in MODEL_KEYS
        },
    }


def selection_diagnostics(
    selected: list[dict[str, Any]],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """選定済みOracleスコアを事後的に診断する。"""
    model_values = {
        model: [row["representative_means"][model] for row in selected]
        for model in MODEL_KEYS
    }
    friedman_values = {
        "base": model_values["base"],
        "bayes_dpo": model_values["basis"],
        "random_dpo": model_values["random"],
    }
    chi2, p_value, kendalls_w = friedman_test(friedman_values)

    comparisons = (
        ("BASiS_vs_Base", "basis", "base"),
        ("BASiS_vs_Random-DPO", "basis", "random"),
        ("Base_vs_Random-DPO", "base", "random"),
    )
    pairwise_rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for index, (name, left, right) in enumerate(comparisons):
        diffs = [
            left_value - right_value
            for left_value, right_value in zip(
                model_values[left],
                model_values[right],
            )
        ]
        raw_p = paired_permutation_p(
            diffs,
            n_permutations=permutations,
            rng=random.Random(seed + index),
        )
        wins, ties, losses = win_tie_loss(diffs, 0.25)
        raw_p_values.append(raw_p)
        pairwise_rows.append(
            {
                "comparison": name,
                "mean_difference": mean(diffs),
                "effect_size_cohens_dz": effect_size(diffs),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "p_raw": raw_p,
            }
        )
    for row, adjusted in zip(pairwise_rows, holm_adjust(raw_p_values)):
        row["p_holm"] = adjusted
        row["significant_at_0_05"] = adjusted < 0.05

    axis_rows: list[dict[str, Any]] = []
    for axis_index, axis in enumerate(REPRESENTATIVE_AXES):
        for comparison_index, (name, left, right) in enumerate(comparisons[:2]):
            left_key = MODEL_ORACLE_KEYS[left]
            right_key = MODEL_ORACLE_KEYS[right]
            diffs = [
                row["oracle_axis_scores"][left_key][axis]
                - row["oracle_axis_scores"][right_key][axis]
                for row in selected
            ]
            raw_p = paired_permutation_p(
                diffs,
                n_permutations=permutations,
                rng=random.Random(seed + 100 + axis_index * 10 + comparison_index),
            )
            wins, ties, losses = win_tie_loss(diffs, 0.25)
            axis_rows.append(
                {
                    "axis": axis,
                    "comparison": name,
                    "mean_difference": mean(diffs),
                    "effect_size_cohens_dz": effect_size(diffs),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "p_raw_unadjusted": raw_p,
                }
            )

    return {
        "warning": (
            "この20件は同じOracleスコアを用いて選定しているため、"
            "以下のp値は選定条件付きの事後診断である。"
            "独立した有意差の証拠や人手評価結果として使用してはならない。"
        ),
        "selection_conditioned_posthoc": True,
        "n": len(selected),
        "representative_five_axis_means": {
            model: mean(values) for model, values in model_values.items()
        },
        "friedman": {
            "chi_square": chi2,
            "p_value": p_value,
            "kendalls_w": kendalls_w,
        },
        "pairwise_representative_mean": pairwise_rows,
        "per_axis_unadjusted_diagnostics": axis_rows,
        "permutations": permutations,
        "seed": seed,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Google Formの内容を1item 1行のCSVへ書く。"""
    fields = [
        "item_id",
        "conversation",
        "response_a",
        "response_b",
        "response_c",
        *[
            f"likert_statement_{index}"
            for index in range(1, len(LIKERT_STATEMENTS) + 1)
        ],
        "scale",
        "final_choice_question",
        "final_choice_options",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "item_id": row["item_id"],
                    "conversation": row["conversation"],
                    "response_a": row["response_a"],
                    "response_b": row["response_b"],
                    "response_c": row["response_c"],
                    **{
                        f"likert_statement_{index}": statement["statement"]
                        for index, statement in enumerate(
                            LIKERT_STATEMENTS,
                            start=1,
                        )
                    },
                    "scale": " / ".join(
                        f"{key}={value}" for key, value in LIKERT_ANCHORS.items()
                    ),
                    "final_choice_question": FINAL_CHOICE_QUESTION,
                    "final_choice_options": " / ".join(FINAL_CHOICE_OPTIONS),
                }
            )


def write_markdown(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    """フォーム確認用Markdownを書く。"""
    lines = [
        f"# {title}",
        "",
        "各応答について、次の基準で7項目を評価してください。",
        "",
        " / ".join(f"{key}={value}" for key, value in LIKERT_ANCHORS.items()),
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## 評価 {row['item_number']} / {len(rows)}",
                "",
                "### これまでの会話",
                "",
                row["conversation"],
                "",
            ]
        )
        for position in ("a", "b", "c"):
            lines.extend(
                [
                    f"### 応答{position.upper()}",
                    "",
                    row[f"response_{position}"],
                    "",
                ]
            )
            for index, statement in enumerate(LIKERT_STATEMENTS, start=1):
                lines.append(f"{index}. {statement['statement']} [1–7]")
            lines.append("")
        lines.extend(
            [
                "### 最後の質問",
                "",
                FINAL_CHOICE_QUESTION,
                "",
                f"選択肢: {' / '.join(FINAL_CHOICE_OPTIONS)}",
                "",
                "任意コメント:",
                "",
                "---",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_apps_script(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    """7段階グリッドを含むGoogle Form生成コードを書く。"""
    payload = json.dumps(rows, ensure_ascii=False)
    statements = json.dumps(
        [statement["statement"] for statement in LIKERT_STATEMENTS],
        ensure_ascii=False,
    )
    columns = json.dumps(list(LIKERT_COLUMNS), ensure_ascii=False)
    final_options = json.dumps(list(FINAL_CHOICE_OPTIONS), ensure_ascii=False)
    script = f"""// Google Apps Scriptへ貼り付け、createEsconvLikertFormを実行する。
function createEsconvLikertForm() {{
  const form = FormApp.create({json.dumps(title, ensure_ascii=False)});
  form.setDescription(
    'このアンケートでは、相談場面に対する3つの匿名応答を評価します。' +
    '各応答を1から7で評価した後、最もふさわしい応答を選んでください。' +
    'モデル名は表示されません。'
  );
  const consent = form.addMultipleChoiceItem()
    .setTitle('説明を読み、研究目的で回答を利用することに同意しますか。')
    .setRequired(true);
  const firstPage = form.addPageBreakItem().setTitle('参加者情報と評価 1');
  consent.setChoices([
    consent.createChoice('同意する', firstPage),
    consent.createChoice('同意しない', FormApp.PageNavigationType.SUBMIT)
  ]);
  form.addTextItem()
    .setTitle('参加者IDを入力してください。氏名は入力しないでください。')
    .setRequired(true);

  const items = {payload};
  const statements = {statements};
  const columns = {columns};
  const finalOptions = {final_options};
  const scaleHelp =
    '1=全く当てはまらない、2=当てはまらない、' +
    '3=あまり当てはまらない、4=どちらともいえない、' +
    '5=やや当てはまる、6=当てはまる、7=非常によく当てはまる';

  items.forEach((item, index) => {{
    if (index > 0) {{
      form.addPageBreakItem().setTitle(`評価 ${{index + 1}} / ${{items.length}}`);
    }}
    form.addSectionHeaderItem()
      .setTitle('これまでの会話')
      .setHelpText(item.conversation);
    ['A', 'B', 'C'].forEach((position) => {{
      const response = item['response_' + position.toLowerCase()];
      form.addSectionHeaderItem()
        .setTitle('応答' + position)
        .setHelpText(response);
      form.addGridItem()
        .setTitle('応答' + position + 'を評価してください。')
        .setHelpText(scaleHelp)
        .setRows(statements)
        .setColumns(columns)
        .setRequired(true);
    }});
    form.addMultipleChoiceItem()
      .setTitle(item.final_choice_question)
      .setChoiceValues(finalOptions)
      .setRequired(true);
    form.addParagraphTextItem().setTitle('この評価についてのコメント（任意）');
  }});
  Logger.log('編集URL: ' + form.getEditUrl());
  Logger.log('回答URL: ' + form.getPublishedUrl());
}}
"""
    path.write_text(script, encoding="utf-8")


def write_versions(
    *,
    output_dir: Path,
    selected: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """匿名位置を変えた3つのフォーム版を書く。"""
    orders = version_orders(len(selected), seed=seed)
    manifest: dict[str, Any] = {}
    for version in FORM_VERSIONS:
        version_dir = output_dir / f"form_version_{version.lower()}"
        version_dir.mkdir(parents=True, exist_ok=True)
        public_rows = [
            public_record(row, item_number=index, order=orders[version][index - 1])
            for index, row in enumerate(selected, start=1)
        ]
        private_rows = [
            private_record(
                row,
                item_number=index,
                order=orders[version][index - 1],
            )
            for index, row in enumerate(selected, start=1)
        ]
        title = f"相談支援応答の7段階評価（フォーム{version}）"
        write_jsonl(version_dir / "form_items_public.jsonl", public_rows)
        write_jsonl(version_dir / "private_model_mapping.jsonl", private_rows)
        write_csv(version_dir / "google_form_items.csv", public_rows)
        write_markdown(
            version_dir / "google_form_sections.md",
            public_rows,
            title,
        )
        write_apps_script(
            version_dir / "create_google_form.gs",
            public_rows,
            title,
        )
        position_counts = Counter(
            (position, model)
            for row in private_rows
            for position, model in row["position_to_model"].items()
        )
        manifest[version] = {
            "count": len(public_rows),
            "position_counts": {
                f"{position}:{model}": count
                for (position, model), count in sorted(position_counts.items())
            },
            "public_jsonl_sha256": sha256_file(
                version_dir / "form_items_public.jsonl"
            ),
            "private_mapping_sha256": sha256_file(
                version_dir / "private_model_mapping.jsonl"
            ),
        }
    return manifest


def main() -> int:
    """Likert形式のGoogle Form成果物を生成する。"""
    args = parse_args()
    response_path = args.v2_run / "three_model_responses.jsonl"
    scores = load_axis_scores(
        v2_run=args.v2_run,
        topconf_run=args.topconf_run,
    )
    candidates = build_candidates(
        response_path=response_path,
        axis_scores=scores,
    )
    selected = select_oracle_enriched(candidates, total=args.count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = selection_diagnostics(
        selected,
        permutations=args.diagnostic_permutations,
        seed=args.seed,
    )
    write_json(args.output_dir / "selection_conditioned_diagnostics.json", diagnostics)
    write_json(
        args.output_dir / "questionnaire_spec.json",
        {
            "version": "esconv_google_form_likert.v2",
            "rating_design": "three_responses_x_seven_axes_plus_final_choice",
            "likert_scale": {
                "minimum": 1,
                "maximum": 7,
                "anchors": LIKERT_ANCHORS,
            },
            "statements": list(LIKERT_STATEMENTS),
            "final_choice": {
                "question": FINAL_CHOICE_QUESTION,
                "options": list(FINAL_CHOICE_OPTIONS),
            },
            "axis_sources": list(QUESTIONS),
            "references": [
                {
                    "citation": (
                        "Mir et al. (2019). Evaluating Style Transfer for Text. "
                        "NAACL-HLT 2019."
                    ),
                    "url": "https://aclanthology.org/N19-1049/",
                },
                {
                    "citation": (
                        "Liu et al. (2021). Towards Emotional Support Dialog "
                        "Systems. ACL-IJCNLP 2021."
                    ),
                    "url": "https://aclanthology.org/2021.acl-long.269/",
                },
                {
                    "citation": (
                        "Zhang et al. (2018). Personalizing Dialogue Agents: "
                        "I have a dog, do you have pets too? ACL 2018."
                    ),
                    "url": "https://aclanthology.org/P18-1205/",
                },
            ],
        },
    )
    versions = write_versions(
        output_dir=args.output_dir,
        selected=selected,
        seed=args.seed,
    )
    write_json(
        args.output_dir / "selection_manifest.json",
        {
            "version": "esconv_oracle_enriched_likert_selection.v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "selection": (
                "代表5軸のBASiS平均と最良control平均の差を使い、"
                "10カテゴリの各カテゴリから上位2件を選ぶ。"
            ),
            "inference_scope": (
                "OracleでBASiS優位が確認されたESConv場面に限定した"
                "対象化ユーザ評価。ESConv全体の無条件な主評価ではない。"
            ),
            "posthoc_selection": True,
            "category_counts": dict(
                sorted(Counter(row["category"] for row in selected).items())
            ),
            "prompt_ids": [row["prompt_id"] for row in selected],
            "minimum_basis_advantage": min(
                row["basis_advantage_over_best_control"] for row in selected
            ),
            "mean_basis_advantage": mean(
                row["basis_advantage_over_best_control"] for row in selected
            ),
            "source": {
                "responses": response_path.as_posix(),
                "responses_sha256": sha256_file(response_path),
                "v2_run": args.v2_run.as_posix(),
                "topconf_run": args.topconf_run.as_posix(),
            },
            "form_versions": versions,
            "blinding": {
                "public_files_contain_model_identity": False,
                "private_mapping_must_not_be_shared_with_participants": True,
                "assignment": (
                    "参加者をフォームA/B/Cへできるだけ同数に割り付ける。"
                ),
            },
        },
    )
    print(
        "ESConv 7段階Likert Google Form用データを書き出しました: "
        f"{args.output_dir} ({len(selected)} items)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
