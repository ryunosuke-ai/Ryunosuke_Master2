#!/usr/bin/env python3
"""ESConvの3モデル人手評価をGoogle Form用に整形する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_V2_RUN = Path(
    "artifacts/evaluations/oracle_eval_runs/"
    "esconv_topconf_three_model_esconv_v2_100_gpt54_v1_"
    "topconf_three_model_esconv_v2_10pt"
)
DEFAULT_TOPCONF_RUN = Path(
    "artifacts/evaluations/oracle_eval_runs/"
    "esconv_topconf_three_model_gpt54_100_10pt_"
    "topconf_three_model_10pt"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/user_eval/google_forms/esconv_representative_v1")
DEFAULT_SEED = 20260720

MODEL_KEYS = ("base", "basis", "random")
MODEL_RESPONSE_KEYS = {
    "base": "base_response",
    "basis": "bayes_dpo_response",
    "random": "random_dpo_response",
}
MODEL_ORACLE_KEYS = {
    "base": "base",
    "basis": "bayes_dpo",
    "random": "random_dpo",
}
FORM_VERSIONS = ("A", "B", "C")
FORM_OPTIONS = ("応答A", "応答B", "応答C", "ほぼ同じ", "判断できない")

QUESTIONS = (
    {
        "key": "style_strength",
        "title": "相談している人を支える応答として、全体的に最も良いのはどれですか。",
        "short_label": "相談支援らしさ",
        "source": "Mir et al. (2019); Liu et al. (2021)",
    },
    {
        "key": "esconv_tone_similarity",
        "title": "相談している人の気持ちを受け止め、やさしく話しているのはどれですか。",
        "short_label": "気持ちの受け止めとやさしさ",
        "source": "Liu et al. (2021)",
    },
    {
        "key": "supporter_role_consistency",
        "title": "これまでの会話に合った、支える立場の話し方を続けているのはどれですか。",
        "short_label": "支える立場の一貫性",
        "source": "Zhang et al. (2018); Liu et al. (2021)",
    },
    {
        "key": "non_directive_support_style",
        "title": (
            "相手の話を理解・整理しようとし、すぐに指示や結論を"
            "押しつけていないのはどれですか。"
        ),
        "short_label": "押しつけない支援",
        "source": "Liu et al. (2021)",
    },
    {
        "key": "premature_advice_avoidance",
        "title": (
            "この会話の段階を考えたとき、助言や提案を出すタイミングが"
            "最も適切なのはどれですか。"
        ),
        "short_label": "助言のタイミング",
        "source": "Liu et al. (2021)",
    },
    {
        "key": "content_preservation",
        "title": "これまでの話の内容に最もよく合っているのはどれですか。",
        "short_label": "話の内容への合い方",
        "source": "Mir et al. (2019)",
    },
    {
        "key": "naturalness",
        "title": "日本語の会話として最も自然で読みやすいのはどれですか。",
        "short_label": "日本語の自然さ",
        "source": "Mir et al. (2019)",
    },
)

REPRESENTATIVE_AXES = (
    "style_strength",
    "esconv_tone_similarity",
    "supporter_role_consistency",
    "non_directive_support_style",
    "premature_advice_avoidance",
)


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="ESConvのGoogle Form用3モデル人手評価データを作成します。"
    )
    parser.add_argument("--v2-run", type=Path, default=DEFAULT_V2_RUN)
    parser.add_argument("--topconf-run", type=Path, default=DEFAULT_TOPCONF_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--main-count", type=int, default=20)
    parser.add_argument("--exploratory-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONLを厳密に読み込む。"""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} が不正なJSONです。") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} がJSON objectではありません。")
            rows.append(row)
    if not rows:
        raise ValueError(f"データが空です: {path}")
    return rows


def write_json(path: Path, payload: Any) -> None:
    """JSONを書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """JSONLを書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    """ファイルのSHA-256を返す。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_targets(categories: list[str], total: int) -> dict[str, int]:
    """カテゴリごとの均等な件数を返す。"""
    unique = sorted(set(categories))
    if not unique:
        raise ValueError("カテゴリがありません。")
    base, remainder = divmod(total, len(unique))
    return {
        category: base + (1 if index < remainder else 0)
        for index, category in enumerate(unique)
    }


def load_axis_scores(
    *,
    v2_run: Path,
    topconf_run: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    """sample -> model -> axisのOracleスコアを読む。"""
    specs = (
        (
            topconf_run / "oracle_tst_10pt" / "raw.jsonl",
            ("style_strength", "content_preservation", "naturalness"),
        ),
        (
            v2_run / "oracle_conversation_style_esconv_v2_10pt" / "raw.jsonl",
            (
                "esconv_tone_similarity",
                "supporter_role_consistency",
                "non_directive_support_style",
            ),
        ),
        (
            v2_run / "oracle_strategy_transition_esconv_v2_10pt" / "raw.jsonl",
            ("premature_advice_avoidance",),
        ),
    )
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for path, axes in specs:
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id") or "").strip()
            model_name = str(row.get("model_name") or "").strip()
            row_scores = row.get("scores")
            if not sample_id or model_name not in MODEL_ORACLE_KEYS.values():
                raise ValueError(f"{path}: sample_id/model_nameが不正です。")
            if not isinstance(row_scores, dict):
                raise ValueError(f"{path}: scoresがありません: {sample_id}")
            for axis in axes:
                if axis not in row_scores:
                    raise ValueError(f"{path}: {sample_id}に{axis}がありません。")
                scores[sample_id][model_name][axis] = float(row_scores[axis])
    return {
        sample_id: {model: dict(axis_scores) for model, axis_scores in models.items()}
        for sample_id, models in scores.items()
    }


def build_candidates(
    *,
    response_path: Path,
    axis_scores: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    """応答と代表軸スコアを結合する。"""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in read_jsonl(response_path):
        prompt_id = str(row.get("prompt_id") or "").strip()
        if not prompt_id or prompt_id in seen:
            raise ValueError(f"prompt_idが空または重複しています: {prompt_id}")
        seen.add(prompt_id)
        scores = axis_scores.get(prompt_id)
        if scores is None:
            raise ValueError(f"Oracleスコアがありません: {prompt_id}")
        for model_key, response_key in MODEL_RESPONSE_KEYS.items():
            if not str(row.get(response_key) or "").strip():
                raise ValueError(f"{prompt_id}: {response_key}が空です。")
            oracle_key = MODEL_ORACLE_KEYS[model_key]
            missing = [
                axis
                for axis in REPRESENTATIVE_AXES
                if axis not in scores.get(oracle_key, {})
            ]
            if missing:
                raise ValueError(f"{prompt_id}/{oracle_key}: 軸不足 {missing}")
        representative_means = {
            model_key: mean(
                scores[MODEL_ORACLE_KEYS[model_key]][axis]
                for axis in REPRESENTATIVE_AXES
            )
            for model_key in MODEL_KEYS
        }
        basis_advantage = representative_means["basis"] - max(
            representative_means["base"],
            representative_means["random"],
        )
        candidates.append(
            {
                "prompt_id": prompt_id,
                "category": str(row.get("category") or ""),
                "axis_focus": list(row.get("axis_focus") or []),
                "history": list(row.get("history") or []),
                "prompt": str(row.get("prompt") or "").strip(),
                "responses": {
                    model_key: str(row[response_key]).strip()
                    for model_key, response_key in MODEL_RESPONSE_KEYS.items()
                },
                "oracle_axis_scores": scores,
                "representative_means": representative_means,
                "basis_advantage_over_best_control": basis_advantage,
            }
        )
    return candidates


def select_model_blind(
    candidates: list[dict[str, Any]],
    *,
    total: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Oracleスコアを使わず、カテゴリ層化で主評価itemを選ぶ。"""
    targets = category_targets([row["category"] for row in candidates], total)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[row["category"]].append(row)
    selected: list[dict[str, Any]] = []
    for category in sorted(targets):
        rows = sorted(grouped[category], key=lambda row: row["prompt_id"])
        random.Random(f"{seed}:{category}").shuffle(rows)
        if len(rows) < targets[category]:
            raise ValueError(f"{category}の候補が不足しています。")
        selected.extend(rows[: targets[category]])
    return sorted(selected, key=lambda row: (row["category"], row["prompt_id"]))


def select_oracle_enriched(
    candidates: list[dict[str, Any]],
    *,
    total: int,
) -> list[dict[str, Any]]:
    """カテゴリを維持しつつ、BASiS優位の事例分析itemを選ぶ。"""
    targets = category_targets([row["category"] for row in candidates], total)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[row["category"]].append(row)
    selected: list[dict[str, Any]] = []
    for category in sorted(targets):
        rows = sorted(
            grouped[category],
            key=lambda row: (
                row["basis_advantage_over_best_control"],
                row["representative_means"]["basis"],
                row["prompt_id"],
            ),
            reverse=True,
        )
        selected.extend(rows[: targets[category]])
    if any(row["basis_advantage_over_best_control"] <= 0 for row in selected):
        raise ValueError("探索用セットにBASiS非優位itemが含まれます。")
    return sorted(selected, key=lambda row: (row["category"], row["prompt_id"]))


def format_history(history: list[dict[str, Any]], prompt: str) -> str:
    """Google Formへ表示する日本語会話を作る。"""
    lines: list[str] = []
    for turn in history:
        speaker = str(turn.get("speaker") or turn.get("role") or "")
        label = "相談者" if speaker.lower() in {"user", "相談者"} else "支援者"
        text = str(turn.get("text") or turn.get("content") or "").strip()
        if text:
            lines.append(f"{label}: {text}")
    lines.append(f"相談者: {prompt}")
    return "\n\n".join(lines)


def version_orders(
    count: int,
    *,
    seed: int,
) -> dict[str, list[tuple[str, str, str]]]:
    """3版を通して各モデルがA/B/Cへ均等に現れる順序を作る。"""
    permutations = list(itertools.permutations(MODEL_KEYS))
    rng = random.Random(seed)
    base_orders: list[tuple[str, str, str]] = []
    while len(base_orders) < count:
        block = list(permutations)
        rng.shuffle(block)
        base_orders.extend(block)
    base_orders = base_orders[:count]
    return {
        "A": base_orders,
        "B": [order[1:] + order[:1] for order in base_orders],
        "C": [order[2:] + order[:2] for order in base_orders],
    }


def public_record(
    row: dict[str, Any],
    *,
    item_number: int,
    order: tuple[str, str, str],
) -> dict[str, Any]:
    """モデル情報を除いたフォーム用レコードを作る。"""
    return {
        "item_id": f"item_{item_number:02d}",
        "item_number": item_number,
        "conversation": format_history(row["history"], row["prompt"]),
        "response_a": row["responses"][order[0]],
        "response_b": row["responses"][order[1]],
        "response_c": row["responses"][order[2]],
        "questions": [
            {"key": question["key"], "title": question["title"]}
            for question in QUESTIONS
        ],
        "options": list(FORM_OPTIONS),
    }


def private_record(
    row: dict[str, Any],
    *,
    item_number: int,
    order: tuple[str, str, str],
    selection_type: str,
) -> dict[str, Any]:
    """分析者だけが保持するモデル対応表を作る。"""
    return {
        "item_id": f"item_{item_number:02d}",
        "prompt_id": row["prompt_id"],
        "category": row["category"],
        "axis_focus": row["axis_focus"],
        "selection_type": selection_type,
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


def write_form_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Google Formへ転記しやすいCSVを書く。"""
    fieldnames = [
        "item_id",
        "item_number",
        "conversation",
        "response_a",
        "response_b",
        "response_c",
        *[f"question_{index}" for index in range(1, len(QUESTIONS) + 1)],
        "options",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "item_id": row["item_id"],
                    "item_number": row["item_number"],
                    "conversation": row["conversation"],
                    "response_a": row["response_a"],
                    "response_b": row["response_b"],
                    "response_c": row["response_c"],
                    **{
                        f"question_{index}": question["title"]
                        for index, question in enumerate(
                            row["questions"], start=1
                        )
                    },
                    "options": " / ".join(row["options"]),
                }
            )


def write_form_markdown(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    """フォーム作成用Markdownを書く。"""
    lines = [
        f"# {title}",
        "",
        "各項目で、会話履歴と3つの匿名応答を読んでください。",
        "各質問について最も当てはまる応答を1つ選んでください。",
        "差がほとんどない場合は「ほぼ同じ」、判断が難しい場合は",
        "「判断できない」を選んでください。",
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
                "### 応答A",
                "",
                row["response_a"],
                "",
                "### 応答B",
                "",
                row["response_b"],
                "",
                "### 応答C",
                "",
                row["response_c"],
                "",
                "### 質問",
                "",
            ]
        )
        for index, question in enumerate(row["questions"], start=1):
            lines.append(f"{index}. {question['title']}")
            lines.append(f"   選択肢: {' / '.join(row['options'])}")
        lines.extend(["", "任意コメント:", "", "---", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_apps_script(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    """Google Apps Scriptでフォームを生成するコードを書く。"""
    payload = json.dumps(rows, ensure_ascii=False)
    questions = json.dumps(
        [question["title"] for question in QUESTIONS],
        ensure_ascii=False,
    )
    options = json.dumps(list(FORM_OPTIONS), ensure_ascii=False)
    script = f"""// Google Apps Scriptへ貼り付け、createEsconvEvaluationFormを実行する。
function createEsconvEvaluationForm() {{
  const form = FormApp.create({json.dumps(title, ensure_ascii=False)});
  form.setDescription(
    'このアンケートでは、相談場面に対する3つの匿名応答を比較します。' +
    'モデル名は表示されません。回答は研究目的で統計的に集計します。'
  );
  const consent = form.addMultipleChoiceItem()
    .setTitle('上記の説明を読み、研究目的で回答を利用することに同意しますか。')
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
  const questions = {questions};
  const options = {options};
  items.forEach((item, index) => {{
    if (index > 0) {{
      form.addPageBreakItem().setTitle(`評価 ${{index + 1}} / ${{items.length}}`);
    }}
    form.addSectionHeaderItem()
      .setTitle('これまでの会話')
      .setHelpText(item.conversation);
    form.addSectionHeaderItem()
      .setTitle('3つの応答')
      .setHelpText(
        '【応答A】\\n' + item.response_a +
        '\\n\\n【応答B】\\n' + item.response_b +
        '\\n\\n【応答C】\\n' + item.response_c
      );
    questions.forEach((question) => {{
      form.addMultipleChoiceItem()
        .setTitle(question)
        .setChoiceValues(options)
        .setRequired(true);
    }});
    form.addParagraphTextItem().setTitle('この評価についてのコメント（任意）');
  }});
  Logger.log('編集URL: ' + form.getEditUrl());
  Logger.log('回答URL: ' + form.getPublishedUrl());
}}
"""
    path.write_text(script, encoding="utf-8")


def write_set(
    *,
    output_dir: Path,
    selected: list[dict[str, Any]],
    selection_type: str,
    title_prefix: str,
    seed: int,
) -> dict[str, Any]:
    """1つの選定セットを3つのフォーム版へ書き出す。"""
    orders = version_orders(len(selected), seed=seed)
    version_manifest: dict[str, Any] = {}
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
                selection_type=selection_type,
            )
            for index, row in enumerate(selected, start=1)
        ]
        title = f"{title_prefix}（フォーム{version}）"
        write_jsonl(version_dir / "form_items_public.jsonl", public_rows)
        write_jsonl(version_dir / "private_model_mapping.jsonl", private_rows)
        write_form_csv(version_dir / "google_form_items.csv", public_rows)
        write_form_markdown(
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
        version_manifest[version] = {
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
    return version_manifest


def main() -> int:
    """Google Form用成果物を生成する。"""
    args = parse_args()
    response_path = args.v2_run / "three_model_responses.jsonl"
    axis_scores = load_axis_scores(
        v2_run=args.v2_run,
        topconf_run=args.topconf_run,
    )
    candidates = build_candidates(
        response_path=response_path,
        axis_scores=axis_scores,
    )
    main_selected = select_model_blind(
        candidates,
        total=args.main_count,
        seed=args.seed,
    )
    exploratory_selected = select_oracle_enriched(
        candidates,
        total=args.exploratory_count,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "questionnaire_spec.json",
        {
            "version": "esconv_google_form_representative.v1",
            "required_question_count_per_item": len(QUESTIONS),
            "representative_axis_count": len(REPRESENTATIVE_AXES),
            "questions": list(QUESTIONS),
            "options": list(FORM_OPTIONS),
            "participant_instruction": (
                "各質問について最も当てはまる応答をA/B/Cから1つ選ぶ。"
                "差が小さい場合は「ほぼ同じ」、評価困難な場合は"
                "「判断できない」を選ぶ。"
            ),
            "references": [
                {
                    "key": "mir_etal_2019",
                    "citation": (
                        "Mir et al. (2019). Evaluating Style Transfer for Text. "
                        "NAACL-HLT 2019."
                    ),
                    "url": "https://aclanthology.org/N19-1049/",
                },
                {
                    "key": "liu_etal_2021",
                    "citation": (
                        "Liu et al. (2021). Towards Emotional Support Dialog "
                        "Systems. ACL-IJCNLP 2021."
                    ),
                    "url": "https://aclanthology.org/2021.acl-long.269/",
                },
                {
                    "key": "zhang_etal_2018",
                    "citation": (
                        "Zhang et al. (2018). Personalizing Dialogue Agents: "
                        "I have a dog, do you have pets too? ACL 2018."
                    ),
                    "url": "https://aclanthology.org/P18-1205/",
                },
            ],
        },
    )

    main_versions = write_set(
        output_dir=args.output_dir / "main_confirmatory",
        selected=main_selected,
        selection_type="model_blind_category_stratified",
        title_prefix="相談支援応答の比較評価",
        seed=args.seed,
    )
    exploratory_versions = write_set(
        output_dir=args.output_dir / "oracle_enriched_exploratory",
        selected=exploratory_selected,
        selection_type="oracle_enriched_posthoc",
        title_prefix="相談支援応答の探索的比較評価",
        seed=args.seed + 1,
    )

    manifest = {
        "version": "esconv_google_form_selection.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "candidate_count": len(candidates),
        "source": {
            "responses": response_path.as_posix(),
            "responses_sha256": sha256_file(response_path),
            "v2_run": args.v2_run.as_posix(),
            "topconf_run": args.topconf_run.as_posix(),
        },
        "main_confirmatory": {
            "count": len(main_selected),
            "selection": (
                "Oracleスコア・モデル応答を使わない、prompt categoryごとの"
                "seed固定層化抽出。主検定にはこちらを使用する。"
            ),
            "category_counts": dict(
                sorted(Counter(row["category"] for row in main_selected).items())
            ),
            "prompt_ids": [row["prompt_id"] for row in main_selected],
            "form_versions": main_versions,
        },
        "oracle_enriched_exploratory": {
            "count": len(exploratory_selected),
            "selection": (
                "代表5軸のBASiS平均と最良control平均の差を使い、"
                "各categoryから上位を抽出した事後的な探索・事例分析用セット。"
            ),
            "research_use_restriction": (
                "Oracle結果を用いた選定なので、無条件の確認的有意差検定や"
                "主結果には使用しない。"
            ),
            "category_counts": dict(
                sorted(
                    Counter(row["category"] for row in exploratory_selected).items()
                )
            ),
            "prompt_ids": [row["prompt_id"] for row in exploratory_selected],
            "minimum_basis_advantage": min(
                row["basis_advantage_over_best_control"]
                for row in exploratory_selected
            ),
            "mean_basis_advantage": mean(
                row["basis_advantage_over_best_control"]
                for row in exploratory_selected
            ),
            "form_versions": exploratory_versions,
        },
        "blinding": {
            "public_files_contain_model_identity": False,
            "private_mapping_must_not_be_shared_with_participants": True,
            "assignment": (
                "参加者をフォームA/B/Cへできるだけ同数に割り付ける。"
            ),
        },
    }
    write_json(args.output_dir / "selection_manifest.json", manifest)
    print(
        "ESConv Google Form用データを書き出しました: "
        f"{args.output_dir} "
        f"(main={len(main_selected)}, exploratory={len(exploratory_selected)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
