"""Oracle評価結果を95%信頼区間付きの群化棒グラフへ描画する。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from scripts.analyze_oracle_three_model_significance import (
    friedman_test,
    holm_adjust,
    paired_permutation_p,
)


MODEL_ORDER = ("base", "basis", "random_dpo")
MODEL_LABELS = {
    "base": "Base",
    "basis": "BASiS",
    "random_dpo": "Random-DPO",
}
MODEL_ALIASES = {
    "base": "base",
    "basis": "basis",
    "bayes_dpo": "basis",
    "random": "random_dpo",
    "random_dpo": "random_dpo",
}
MODEL_COLORS = {
    "base": "#4C78A8",
    "basis": "#E45756",
    "random_dpo": "#54A24B",
}
MODEL_HATCHES = {
    "base": "//",
    "basis": "",
    "random_dpo": "..",
}
PAIRWISE = (
    ("BASiS vs Base", "basis", "base"),
    ("BASiS vs Random-DPO", "basis", "random_dpo"),
    ("Base vs Random-DPO", "base", "random_dpo"),
)


@dataclass(frozen=True)
class AxisSpec:
    """1評価軸の表示名、固定平均、raw入力。"""

    key: str
    label: str
    means: dict[str, float]
    raw_path: Path


@dataclass(frozen=True)
class FigureSpec:
    """1実験図の設定。"""

    slug: str
    title: str
    axes: tuple[AxisSpec, ...]


def read_axis_scores(axis: AxisSpec) -> dict[str, dict[str, float]]:
    """raw JSONLからsample単位の3モデルスコアを読む。"""

    by_sample: dict[str, dict[str, float]] = {}
    with axis.raw_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(
                row.get("sample_id") or row.get("prompt_id") or ""
            ).strip()
            model = MODEL_ALIASES.get(str(row.get("model_name") or ""))
            scores = row.get("scores")
            if (
                not sample_id
                or model is None
                or not isinstance(scores, dict)
                or axis.key not in scores
            ):
                continue
            target = by_sample.setdefault(sample_id, {})
            if model in target:
                raise ValueError(
                    f"{axis.raw_path}:{line_number}: "
                    f"{sample_id}/{model}/{axis.key}が重複しています。"
                )
            target[model] = float(scores[axis.key])
    complete = {
        sample_id: values
        for sample_id, values in by_sample.items()
        if all(model in values for model in MODEL_ORDER)
    }
    if not complete:
        raise ValueError(f"{axis.raw_path}: 3モデルが揃ったデータがありません。")
    if len(complete) != len(by_sample):
        raise ValueError(
            f"{axis.raw_path}: 3モデルが揃わないsampleがあります: "
            f"{len(by_sample) - len(complete)}件"
        )
    return complete


def percentile(values: list[float], q: float) -> float:
    """線形補間でpercentileを求める。"""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("空の値からpercentileは計算できません。")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    """平均値のpercentile bootstrap 95% CIを返す。"""

    if samples < 100:
        raise ValueError("bootstrap回数は100以上にしてください。")
    count = len(values)
    estimates = [
        mean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def stars(p_value: float) -> str:
    """有意水準をアスタリスク表記へ変換する。"""

    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def analyze_figure(
    spec: FigureSpec,
    *,
    bootstrap: int,
    permutations: int,
    seed: int,
    mean_tolerance: float = 0.011,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """平均、CI、対応あり検定を計算し、指定平均との一致を検証する。"""

    summary_rows: list[dict[str, Any]] = []
    significance_rows: list[dict[str, Any]] = []
    for axis_index, axis in enumerate(spec.axes):
        by_sample = read_axis_scores(axis)
        sample_ids = sorted(by_sample)
        values_by_model = {
            model: [by_sample[sample_id][model] for sample_id in sample_ids]
            for model in MODEL_ORDER
        }
        for model_index, model in enumerate(MODEL_ORDER):
            raw_mean = mean(values_by_model[model])
            expected = float(axis.means[model])
            if abs(raw_mean - expected) > mean_tolerance:
                raise ValueError(
                    f"{axis.key}/{model}: 指定平均{expected:.3f}と"
                    f"raw平均{raw_mean:.3f}が一致しません。"
                )
            ci_low, ci_high = bootstrap_mean_ci(
                values_by_model[model],
                samples=bootstrap,
                rng=random.Random(seed + axis_index * 101 + model_index),
            )
            summary_rows.append(
                {
                    "axis": axis.key,
                    "axis_label": axis.label,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "n": len(sample_ids),
                    "mean": expected,
                    "raw_mean": raw_mean,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
        friedman_values = {
            "base": values_by_model["base"],
            "bayes_dpo": values_by_model["basis"],
            "random_dpo": values_by_model["random_dpo"],
        }
        chi2, omnibus_p, kendalls_w = friedman_test(friedman_values)
        raw_p_values = []
        pair_records = []
        for pair_index, (label, left, right) in enumerate(PAIRWISE):
            diffs = [
                left_score - right_score
                for left_score, right_score in zip(
                    values_by_model[left],
                    values_by_model[right],
                )
            ]
            raw_p = paired_permutation_p(
                diffs,
                n_permutations=permutations,
                rng=random.Random(
                    seed + axis_index * 1009 + pair_index * 53
                ),
            )
            raw_p_values.append(raw_p)
            pair_records.append(
                {
                    "axis": axis.key,
                    "comparison": label,
                    "left_model": left,
                    "right_model": right,
                    "n": len(sample_ids),
                    "mean_difference": mean(diffs),
                    "friedman_chi2": chi2,
                    "friedman_p": omnibus_p,
                    "kendalls_w": kendalls_w,
                    "p_raw": raw_p,
                }
            )
        adjusted = holm_adjust(raw_p_values)
        for row, p_holm in zip(pair_records, adjusted):
            row["p_holm"] = p_holm
            row["significant"] = omnibus_p < 0.05 and p_holm < 0.05
            row["stars"] = stars(p_holm) if row["significant"] else "ns"
            significance_rows.append(row)
    return summary_rows, significance_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """辞書行をCSVへ保存する。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"{path}: 保存する行がありません。")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def significance_bracket(
    axes: Any,
    left: float,
    right: float,
    height: float,
    label: str,
) -> None:
    """2本の棒の間に有意差括弧を描く。"""

    vertical = 0.08
    axes.plot(
        [left, left, right, right],
        [height, height + vertical, height + vertical, height],
        color="#202020",
        linewidth=1.0,
        clip_on=False,
    )
    axes.text(
        (left + right) / 2,
        height + vertical + 0.015,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def render_figure(
    spec: FigureSpec,
    summary_rows: list[dict[str, Any]],
    significance_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    dpi: int,
    y_min: float,
) -> tuple[Path, Path, Path]:
    """群化棒グラフをPNG、PDF、SVGへ保存する。"""

    matplotlib_cache = output_dir / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "描画にはmatplotlibが必要です。"
            "`python3 -m pip install matplotlib`を実行してください。"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_axis_model = {
        (row["axis"], row["model"]): row for row in summary_rows
    }
    significant = {
        (row["axis"], row["left_model"], row["right_model"]): row
        for row in significance_rows
        if row["significant"]
    }
    axis_count = len(spec.axes)
    x_positions = np.arange(axis_count, dtype=float)
    width = 0.23
    offsets = {"base": -width, "basis": 0.0, "random_dpo": width}
    figure_width = max(11.5, 1.75 * axis_count)
    figure, axes = plt.subplots(figsize=(figure_width, 6.8))
    bar_centers: dict[tuple[str, str], float] = {}
    upper_tops: dict[tuple[str, str], float] = {}

    for model in MODEL_ORDER:
        means = [
            rows_by_axis_model[(axis.key, model)]["mean"]
            for axis in spec.axes
        ]
        lower_errors = [
            rows_by_axis_model[(axis.key, model)]["mean"]
            - rows_by_axis_model[(axis.key, model)]["ci95_low"]
            for axis in spec.axes
        ]
        upper_errors = [
            rows_by_axis_model[(axis.key, model)]["ci95_high"]
            - rows_by_axis_model[(axis.key, model)]["mean"]
            for axis in spec.axes
        ]
        centers = x_positions + offsets[model]
        bars = axes.bar(
            centers,
            means,
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            edgecolor="#303030",
            linewidth=0.7,
            hatch=MODEL_HATCHES[model],
            yerr=np.array([lower_errors, upper_errors]),
            error_kw={
                "ecolor": "#222222",
                "elinewidth": 1.0,
                "capsize": 3,
                "capthick": 1.0,
            },
            zorder=3,
        )
        for axis, bar, value, upper_error in zip(
            spec.axes,
            bars,
            means,
            upper_errors,
        ):
            center = bar.get_x() + bar.get_width() / 2
            bar_centers[(axis.key, model)] = center
            upper_tops[(axis.key, model)] = value + upper_error
            axes.text(
                center,
                value + upper_error + 0.055,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#202020",
                zorder=5,
            )

    for axis in spec.axes:
        pairs = [
            pair
            for pair in PAIRWISE
            if (axis.key, pair[1], pair[2]) in significant
            and "basis" in (pair[1], pair[2])
        ]
        base_height = max(
            upper_tops[(axis.key, model)] for model in MODEL_ORDER
        ) + 0.30
        for layer, (_, left, right) in enumerate(pairs):
            row = significant[(axis.key, left, right)]
            significance_bracket(
                axes,
                bar_centers[(axis.key, left)],
                bar_centers[(axis.key, right)],
                base_height + layer * 0.31,
                str(row["stars"]),
            )

    labels = [
        textwrap.fill(axis.label, width=20, break_long_words=False)
        for axis in spec.axes
    ]
    axes.set_xticks(x_positions, labels)
    axes.tick_params(axis="x", labelsize=9, pad=8)
    axes.set_ylabel("Oracle score (1–10)", fontsize=11)
    axes.set_ylim(y_min, 11.15)
    axes.set_yticks(range(max(0, int(y_min)), 11))
    axes.axhline(10, color="#777777", linewidth=0.8, linestyle=":", zorder=1)
    axes.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8, zorder=0)
    axes.set_axisbelow(True)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.set_title(spec.title, fontsize=14, fontweight="bold", pad=16)
    axes.legend(
        loc="upper left",
        ncols=3,
        frameon=False,
        bbox_to_anchor=(0.0, 1.01),
    )
    axes.text(
        1.0,
        -0.29,
        "Bars: mean; whiskers: bootstrap 95% CI. "
        "Brackets: BASiS vs controls; * p<.05, ** p<.01, *** p<.001 "
        "(paired permutation, Holm-corrected).",
        transform=axes.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#444444",
    )
    figure.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.31)
    png = output_dir / f"{spec.slug}_oracle_scores.png"
    pdf = output_dir / f"{spec.slug}_oracle_scores.pdf"
    svg = output_dir / f"{spec.slug}_oracle_scores.svg"
    figure.savefig(png, dpi=dpi, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    figure.savefig(svg, facecolor="white")
    plt.close(figure)
    return png, pdf, svg


def run_plot_cli(spec: FigureSpec) -> int:
    """データセット固有scriptから利用する共通CLI。"""

    parser = argparse.ArgumentParser(
        description=f"{spec.title}のOracle評価棒グラフを作成します。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/figures/oracle_results"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--y-min",
        type=float,
        default=0.0,
        help="論文用の既定は0。拡大表示する場合のみ明示的に変更する。",
    )
    args = parser.parse_args()
    if not 0.0 <= args.y_min < 10.0:
        raise ValueError("--y-minは0以上10未満にしてください。")
    summary, significance = analyze_figure(
        spec,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed,
    )
    write_csv(
        args.output_dir / f"{spec.slug}_oracle_scores.csv",
        summary,
    )
    write_csv(
        args.output_dir / f"{spec.slug}_significance.csv",
        significance,
    )
    outputs = render_figure(
        spec,
        summary,
        significance,
        output_dir=args.output_dir,
        dpi=args.dpi,
        y_min=args.y_min,
    )
    print("\n".join(str(path) for path in outputs))
    return 0
