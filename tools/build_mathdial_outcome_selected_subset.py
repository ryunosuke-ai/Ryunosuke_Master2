#!/usr/bin/env python3
"""既存MathDial Oracle結果から探索的な成功事例部分集合を再構成する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scripts.run_mathdial_statistics import (
    analyze,
    analyze_strata,
    load_axis_scores,
    load_prompt_strata,
    write_csv,
)


MODELS = ("base", "basis", "random_dpo")
STATUS = "exploratory_outcome_selected_success_case_analysis"
SELECTION_RULE = "basis_category_overall_minus_max_base_random_desc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONLを読めません: {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_sample_ids(
    pedagogical_rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """教育総合のBASiS対最良比較モデルmarginで上位件を固定する。"""
    scores: dict[str, dict[str, float]] = {}
    for row in pedagogical_rows:
        sample_id = str(row["sample_id"])
        model = str(row["model_name"])
        if model in MODELS:
            scores.setdefault(sample_id, {})[model] = float(row["overall_score"])

    complete = {
        sample_id: values
        for sample_id, values in scores.items()
        if all(model in values for model in MODELS)
    }
    if len(complete) < count:
        raise ValueError(
            f"完全な3モデル評価が不足しています: {len(complete)}/{count}"
        )

    ranked = []
    for sample_id, values in complete.items():
        competitor = max(values["base"], values["random_dpo"])
        ranked.append(
            {
                "sample_id": sample_id,
                "selection_margin": values["basis"] - competitor,
                "basis_category_overall": values["basis"],
                "base_category_overall": values["base"],
                "random_dpo_category_overall": values["random_dpo"],
            }
        )
    # 同marginではsample_id降順にして、監査可能な順序を固定する。
    ranked.sort(
        key=lambda row: (row["selection_margin"], row["sample_id"]),
        reverse=True,
    )
    selected = ranked[:count]
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank
    return selected


def filter_rows(
    rows: list[dict[str, Any]], selected_ids: set[str]
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("sample_id")) in selected_ids]


def render_report(
    *,
    manifest: dict[str, Any],
    model_summary: list[dict[str, Any]],
    omnibus: list[dict[str, Any]],
    posthoc: list[dict[str, Any]],
) -> str:
    summary = {
        (row["axis"], row["model_name"]): row for row in model_summary
    }
    omnibus_by_axis = {row["axis"]: row for row in omnibus}
    posthoc_by_key = {
        (row["axis"], row["comparison"]): row for row in posthoc
    }
    lines = [
        "# MathDial outcome-selected top-100 success-case analysis",
        "",
        "> **位置づけ:** この部分集合は既存Oracle教育総合得点を用いて選定した探索的成功事例分析である。選定と検定が同じ評価結果に依存するため、p値を未選定標本の確認的推論として解釈しない。",
        "",
        "## Selection",
        "",
        f"- status: `{manifest['status']}`",
        f"- source records: {manifest['source_count']}",
        f"- selected records: {manifest['selected_count']}",
        f"- rule: `{manifest['selection_rule']}`",
        f"- cutoff margin: {manifest['cutoff_margin']:.6f}",
        "",
        "## Educational axes",
        "",
        "| axis | Base | BASiS | Random | B-Base | p_holm | B-Random | p_holm | Friedman p | W |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for axis in sorted(a for a in omnibus_by_axis if a.startswith("pedagogical_v2.")):
        bb = posthoc_by_key.get((axis, "BASiS_vs_Base"))
        br = posthoc_by_key.get((axis, "BASiS_vs_Random-DPO"))
        omni = omnibus_by_axis[axis]
        lines.append(
            "| {axis} | {base:.3f} | **{basis:.3f}** | {random:.3f} | "
            "{bb_diff} | {bb_p} | {br_diff} | {br_p} | {friedman:.4g} | {w:.3f} |".format(
                axis=axis.removeprefix("pedagogical_v2."),
                base=float(summary[(axis, "base")]["mean"]),
                basis=float(summary[(axis, "basis")]["mean"]),
                random=float(summary[(axis, "random_dpo")]["mean"]),
                bb_diff=f"{float(bb['mean_diff']):+.3f}" if bb else "-",
                bb_p=f"{float(bb['p_holm']):.4g}" if bb else "-",
                br_diff=f"{float(br['mean_diff']):+.3f}" if br else "-",
                br_p=f"{float(br['p_holm']):.4g}" if br else "-",
                friedman=float(omni["p_value"]),
                w=float(omni["kendalls_w"]),
            )
        )
    lines.extend(
        [
            "",
            "## General-quality axes",
            "",
            "| axis | Base | BASiS | Random | B-Base | p_holm | B-Random | p_holm | Friedman p | W |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for axis in sorted(a for a in omnibus_by_axis if a.startswith("general.")):
        bb = posthoc_by_key.get((axis, "BASiS_vs_Base"))
        br = posthoc_by_key.get((axis, "BASiS_vs_Random-DPO"))
        omni = omnibus_by_axis[axis]
        lines.append(
            "| {axis} | {base:.3f} | **{basis:.3f}** | {random:.3f} | "
            "{bb_diff} | {bb_p} | {br_diff} | {br_p} | {friedman:.4g} | {w:.3f} |".format(
                axis=axis.removeprefix("general."),
                base=float(summary[(axis, "base")]["mean"]),
                basis=float(summary[(axis, "basis")]["mean"]),
                random=float(summary[(axis, "random_dpo")]["mean"]),
                bb_diff=f"{float(bb['mean_diff']):+.3f}" if bb else "-",
                bb_p=f"{float(bb['p_holm']):.4g}" if bb else "-",
                br_diff=f"{float(br['mean_diff']):+.3f}" if br else "-",
                br_p=f"{float(br['p_holm']):.4g}" if br else "-",
                friedman=float(omni["p_value"]),
                w=float(omni["kendalls_w"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "この結果が記述するのは、元の150件のうち既存OracleがBASiS優位と評価した100件における再集計である。MathDial全体、未評価データ、または事前選定された対象に対する効果量・有意差ではない。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MathDial既存Oracle結果から探索的上位部分集合を作る"
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("--countは正の整数にしてください。")

    source_eval = args.source_run / "evaluation"
    inputs = {
        "prompts": source_eval / "prompts_ja.jsonl",
        "responses": source_eval / "responses.jsonl",
        "oracle_input": source_eval / "oracle_input.jsonl",
        "pedagogical": source_eval / "oracle/pedagogical_v2/raw.jsonl",
        "general": source_eval / "oracle/general/raw.jsonl",
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}入力がありません: {path}")

    pedagogical_rows = read_jsonl(inputs["pedagogical"])
    selected = select_sample_ids(pedagogical_rows, args.count)
    selected_ids = {row["sample_id"] for row in selected}
    prompt_rows = filter_rows(read_jsonl(inputs["prompts"]), selected_ids)
    prompt_by_id = {str(row["sample_id"]): row for row in prompt_rows}
    for row in selected:
        prompt = prompt_by_id.get(row["sample_id"], {})
        row["selection_teacher_move"] = prompt.get("selection_teacher_move")
        row["selection_stage"] = prompt.get("selection_stage")

    output_eval = args.output_root / "evaluation"
    write_jsonl(args.output_root / "selected_sample_ids.jsonl", selected)
    for name in ("prompts", "responses", "oracle_input", "pedagogical", "general"):
        rows = filter_rows(read_jsonl(inputs[name]), selected_ids)
        if name == "pedagogical":
            output = output_eval / "oracle/pedagogical_v2/raw.jsonl"
            expected = args.count * 3
        elif name == "general":
            output = output_eval / "oracle/general/raw.jsonl"
            expected = args.count * 3
        else:
            output = output_eval / f"{name}.jsonl"
            expected = args.count * 3 if name == "oracle_input" else args.count
        if len(rows) != expected:
            raise ValueError(f"{name}のsubset件数が不正です: {len(rows)}/{expected}")
        write_jsonl(output, rows)

    raw_paths = [
        output_eval / "oracle/pedagogical_v2/raw.jsonl",
        output_eval / "oracle/general/raw.jsonl",
    ]
    axis_scores = load_axis_scores(raw_paths)
    summary, omnibus, posthoc = analyze(
        axis_scores,
        permutations=args.permutations,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    statistics = output_eval / "statistics"
    write_csv(statistics / "model_summary.csv", summary)
    write_csv(statistics / "omnibus_friedman.csv", omnibus)
    write_csv(statistics / "posthoc_pairwise.csv", posthoc)
    strata_summary, strata_pairwise = analyze_strata(
        axis_scores,
        load_prompt_strata(output_eval / "prompts.jsonl"),
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    write_csv(statistics / "stratum_model_summary.csv", strata_summary)
    write_csv(statistics / "stratum_pairwise_summary.csv", strata_pairwise)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "source_run": str(args.source_run),
        "source_count": len({str(row["sample_id"]) for row in pedagogical_rows}),
        "selected_count": args.count,
        "selection_rule": SELECTION_RULE,
        "selection_uses_oracle_outcomes": True,
        "confirmatory_inference_allowed": False,
        "cutoff_margin": selected[-1]["selection_margin"],
        "seed": args.seed,
        "permutations": args.permutations,
        "bootstrap": args.bootstrap,
        "input_hashes": {name: sha256(path) for name, path in inputs.items()},
        "output_hashes": {},
    }
    for path in sorted(args.output_root.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "report.md"}:
            manifest["output_hashes"][str(path.relative_to(args.output_root))] = sha256(path)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "report.md").write_text(
        render_report(
            manifest=manifest,
            model_summary=summary,
            omnibus=omnibus,
            posthoc=posthoc,
        ),
        encoding="utf-8",
    )
    print(f"selected={args.count} cutoff={selected[-1]['selection_margin']:.6f}")
    print(f"report={args.output_root / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
