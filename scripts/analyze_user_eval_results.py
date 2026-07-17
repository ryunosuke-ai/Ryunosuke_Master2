"""BASiS vs Random人手A/B評価結果を集計する。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ITEMS_PATH = Path("artifacts/user_eval/items/user_eval_items.jsonl")
DEFAULT_RESPONSES_DIR = Path("artifacts/user_eval/responses")
DEFAULT_RESULTS_DIR = Path("artifacts/user_eval/results")
SOURCE_BASIS = "basis"
SOURCE_RANDOM = "random"
EVALUATION_AXES = (
    ("emotion_reception", "Emotion"),
    ("advice_timing", "Advice timing"),
    ("contextual_response", "Context fit"),
    ("warmth", "Warmth"),
    ("conversation_progress", "Progress"),
)
AXIS_KEYS = tuple(key for key, _ in EVALUATION_AXES)
AXIS_JA_LABELS = {
    "emotion_reception": "気持ちの受け止め",
    "advice_timing": "助言のタイミング",
    "contextual_response": "話への合い方",
    "warmth": "温かさ",
    "conversation_progress": "会話の前進",
}
AXIS_WEIGHTS = {
    "emotion_reception": 0.25,
    "advice_timing": 0.20,
    "contextual_response": 0.35,
    "warmth": 0.15,
    "conversation_progress": 0.05,
}


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を読む。"""
    parser = argparse.ArgumentParser(description="人手A/B評価結果をBASiS基準で集計します。")
    parser.add_argument("--items", default=DEFAULT_ITEMS_PATH.as_posix())
    parser.add_argument("--responses-dir", default=DEFAULT_RESPONSES_DIR.as_posix())
    parser.add_argument("--input", dest="inputs", action="append", default=[])
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR.as_posix())
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONLを読み込む。壊れた行は例外にする。"""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} をJSONとして読めません: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} がJSON objectではありません。")
            records.append(payload)
    return records


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    """CSVを書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def discover_response_files(responses_dir: Path, explicit_inputs: list[str]) -> list[Path]:
    """分析対象の回答JSONLを列挙する。"""
    if explicit_inputs:
        return [Path(path) for path in explicit_inputs]
    if not responses_dir.exists():
        return []
    return sorted(path for path in responses_dir.glob("*.jsonl") if path.is_file())


def load_items(path: Path) -> dict[str, dict[str, Any]]:
    """評価itemをitem_idで引ける辞書として読む。"""
    if not path.exists():
        return {}
    items = read_jsonl(path)
    return {str(item["item_id"]): item for item in items}


def rating_to_a_preference(rating: int) -> int:
    """5段階ratingをA基準スコアへ変換する。"""
    mapping = {1: 2, 2: 1, 3: 0, 4: -1, 5: -2}
    if rating not in mapping:
        raise ValueError(f"ratingは1-5である必要があります: {rating}")
    return mapping[rating]


def extract_axis_ratings(record: dict[str, Any]) -> dict[str, int]:
    """回答レコードから軸別ratingを取得する。古い単一rating形式も読む。"""
    axis_ratings = record.get("axis_ratings")
    ratings: dict[str, int] = {}
    if isinstance(axis_ratings, dict):
        for axis_key in AXIS_KEYS:
            if axis_key not in axis_ratings:
                continue
            try:
                rating = int(axis_ratings[axis_key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{axis_key} のratingが不正です: {axis_ratings[axis_key]}") from exc
            if rating < 1 or rating > 5:
                raise ValueError(f"{axis_key} のratingは1-5である必要があります: {rating}")
            ratings[axis_key] = rating
    if len(ratings) == len(AXIS_KEYS):
        return ratings

    if "rating" in record:
        legacy_rating = int(record.get("rating"))
        rating_to_a_preference(legacy_rating)
        return {axis_key: legacy_rating for axis_key in AXIS_KEYS}

    missing = [axis_key for axis_key in AXIS_KEYS if axis_key not in ratings]
    raise ValueError(f"axis_ratingsが不足しています: {', '.join(missing)}")


def axis_score_from_rating(rating: int, basis_position: str) -> int:
    """1軸のraw ratingをBASiS基準スコアに変換する。"""
    a_preference = rating_to_a_preference(rating)
    return a_preference if basis_position == "A" else -a_preference


def weighted_axis_score(axis_scores: dict[str, int]) -> float:
    """Oracle評価のweighted overallに対応する5軸の重み付きスコアを返す。"""
    return sum(float(axis_scores[axis_key]) * AXIS_WEIGHTS[axis_key] for axis_key in AXIS_KEYS)


def normalize_response(record: dict[str, Any], item_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A/B ratingをBASiS基準スコアへ補正する。"""
    item_id = str(record.get("item_id") or "").strip()
    item = item_lookup.get(item_id, {})
    model_a_source = str(record.get("model_a_source") or item.get("model_a_source") or "").strip()
    model_b_source = str(record.get("model_b_source") or item.get("model_b_source") or "").strip()
    if SOURCE_BASIS not in {model_a_source, model_b_source}:
        raise ValueError(f"BASiS sourceを判定できません: item_id={item_id}")
    if SOURCE_RANDOM not in {model_a_source, model_b_source}:
        raise ValueError(f"Random sourceを判定できません: item_id={item_id}")

    basis_position = "A" if model_a_source == SOURCE_BASIS else "B"
    random_position = "A" if model_a_source == SOURCE_RANDOM else "B"
    axis_ratings = extract_axis_ratings(record)
    axis_scores = {
        axis_key: axis_score_from_rating(rating, basis_position)
        for axis_key, rating in axis_ratings.items()
    }
    overall_basis_score = weighted_axis_score(axis_scores)
    if overall_basis_score > 0:
        winner = SOURCE_BASIS
    elif overall_basis_score < 0:
        winner = SOURCE_RANDOM
    else:
        winner = "tie"

    row = {
        "participant_id": str(record.get("participant_id") or ""),
        "session_id": str(record.get("session_id") or ""),
        "item_id": item_id,
        "category": str(record.get("category") or item.get("category") or ""),
        "stratum": str(record.get("stratum") or item.get("stratum") or ""),
        "prompt": str(record.get("prompt") or item.get("prompt") or ""),
        "displayed_order": str(record.get("displayed_order") or item.get("displayed_order") or ""),
        "rating_raw": record.get("rating", ""),
        "basis_position": basis_position,
        "random_position": random_position,
        "basis_score": overall_basis_score,
        "overall_basis_score": overall_basis_score,
        "winner": winner,
        "comment": str(record.get("comment") or ""),
        "timestamp": str(record.get("timestamp") or ""),
        "source_file": str(record.get("_source_file") or ""),
        "oracle_winner": str(item.get("oracle_winner") or record.get("oracle_winner") or ""),
        "oracle_score_gap": item.get("score_gap", record.get("score_gap", "")),
    }
    for axis_key in AXIS_KEYS:
        row[f"{axis_key}_rating_raw"] = axis_ratings[axis_key]
        row[f"{axis_key}_basis_score"] = axis_scores[axis_key]
    return row


def load_and_normalize_responses(paths: list[Path], item_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """複数JSONLを読み込み、BASiS基準に正規化する。"""
    normalized_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        for record in read_jsonl(path):
            record["_source_file"] = path.as_posix()
            row = normalize_response(record, item_lookup)
            key = (row["participant_id"], row["session_id"], row["item_id"])
            if key in normalized_by_key:
                del normalized_by_key[key]
            normalized_by_key[key] = row
    return list(normalized_by_key.values())


def sample_std(values: list[float]) -> float:
    """サンプル標準偏差を返す。"""
    if len(values) <= 1:
        return 0.0
    return stdev(values)


def standard_error(values: list[float]) -> float:
    """標準誤差を返す。"""
    if not values:
        return 0.0
    return sample_std(values) / math.sqrt(len(values))


def exact_sign_test_p_value(basis_wins: int, random_wins: int) -> float | None:
    """tieを除いた二項符号検定の両側p値を返す。"""
    trials = basis_wins + random_wins
    if trials == 0:
        return None
    extreme = max(basis_wins, random_wins)
    tail = sum(math.comb(trials, k) for k in range(extreme, trials + 1)) / (2**trials)
    return min(1.0, 2.0 * tail)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """全体summaryを作る。"""
    scores = [float(row["overall_basis_score"]) for row in rows]
    raw_ratings = [
        int(row[f"{axis_key}_rating_raw"])
        for row in rows
        for axis_key in AXIS_KEYS
    ]
    rating_counts = Counter(raw_ratings)
    winner_counts = Counter(str(row["winner"]) for row in rows)
    total = len(rows)
    std_value = sample_std(scores)
    se_value = standard_error(scores)
    ci_delta = 1.96 * se_value
    summary = {
        "total_responses": total,
        "unique_participants": len({row["participant_id"] for row in rows}),
        "unique_sessions": len({row["session_id"] for row in rows}),
        "unique_items": len({row["item_id"] for row in rows}),
        "basis_win_count": winner_counts[SOURCE_BASIS],
        "random_win_count": winner_counts[SOURCE_RANDOM],
        "tie_count": winner_counts["tie"],
        "basis_win_rate": winner_counts[SOURCE_BASIS] / total if total else 0.0,
        "random_win_rate": winner_counts[SOURCE_RANDOM] / total if total else 0.0,
        "tie_rate": winner_counts["tie"] / total if total else 0.0,
        "mean_rating_raw": mean(raw_ratings) if raw_ratings else 0.0,
        "rating_1_count": rating_counts[1],
        "rating_2_count": rating_counts[2],
        "rating_3_count": rating_counts[3],
        "rating_4_count": rating_counts[4],
        "rating_5_count": rating_counts[5],
        "mean_basis_score": mean(scores) if scores else 0.0,
        "mean_overall_basis_score": mean(scores) if scores else 0.0,
        "std_basis_score": std_value,
        "se_basis_score": se_value,
        "ci95_low": (mean(scores) - ci_delta) if scores else 0.0,
        "ci95_high": (mean(scores) + ci_delta) if scores else 0.0,
        "p_value_sign_test_two_sided": exact_sign_test_p_value(
            winner_counts[SOURCE_BASIS],
            winner_counts[SOURCE_RANDOM],
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    for axis_key, _ in EVALUATION_AXES:
        axis_scores = [float(row[f"{axis_key}_basis_score"]) for row in rows]
        axis_winners = Counter(
            SOURCE_BASIS if score > 0 else SOURCE_RANDOM if score < 0 else "tie"
            for score in axis_scores
        )
        summary[f"{axis_key}_mean_basis_score"] = mean(axis_scores) if axis_scores else 0.0
        summary[f"{axis_key}_basis_win_count"] = axis_winners[SOURCE_BASIS]
        summary[f"{axis_key}_random_win_count"] = axis_winners[SOURCE_RANDOM]
        summary[f"{axis_key}_tie_count"] = axis_winners["tie"]
        summary[f"{axis_key}_basis_win_rate"] = axis_winners[SOURCE_BASIS] / total if total else 0.0
        summary[f"{axis_key}_random_win_rate"] = axis_winners[SOURCE_RANDOM] / total if total else 0.0
        summary[f"{axis_key}_tie_rate"] = axis_winners["tie"] / total if total else 0.0
        summary[f"{axis_key}_weight"] = AXIS_WEIGHTS[axis_key]
    return summary


def grouped_summary(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    """参加者別・item別summaryを作る。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        scores = [float(row["overall_basis_score"]) for row in group_rows]
        winners = Counter(str(row["winner"]) for row in group_rows)
        summary = {
            group_key: key,
            "responses": len(group_rows),
            "basis_win_count": winners[SOURCE_BASIS],
            "random_win_count": winners[SOURCE_RANDOM],
            "tie_count": winners["tie"],
            "basis_win_rate": winners[SOURCE_BASIS] / len(group_rows),
            "random_win_rate": winners[SOURCE_RANDOM] / len(group_rows),
            "tie_rate": winners["tie"] / len(group_rows),
            "mean_basis_score": mean(scores) if scores else 0.0,
            "mean_overall_basis_score": mean(scores) if scores else 0.0,
            "std_basis_score": sample_std(scores),
            "se_basis_score": standard_error(scores),
        }
        for axis_key, _ in EVALUATION_AXES:
            axis_scores = [float(row[f"{axis_key}_basis_score"]) for row in group_rows]
            summary[f"{axis_key}_mean_basis_score"] = mean(axis_scores) if axis_scores else 0.0
        if group_key == "item_id":
            first = group_rows[0]
            summary.update(
                {
                    "category": first.get("category", ""),
                    "stratum": first.get("stratum", ""),
                    "prompt": first.get("prompt", ""),
                    "oracle_winner": first.get("oracle_winner", ""),
                    "oracle_score_gap": first.get("oracle_score_gap", ""),
                }
            )
        summaries.append(summary)
    return summaries


def format_rate(value: float) -> str:
    """割合を表示用文字列にする。"""
    return f"{value * 100:.1f}%"


def write_report(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    response_files: list[Path],
    items_path: Path,
    output_path: Path,
) -> None:
    """Markdownレポートを書き出す。"""
    p_value = summary.get("p_value_sign_test_two_sided")
    if p_value is None:
        p_value_text = "N/A"
    else:
        p_value_text = f"{p_value:.4g}"
    interpretation = (
        "BASiS基準スコアの平均が正であれば、人手評価でもBASiS側が選ばれる傾向を示す。"
        "95%信頼区間が0をまたぐ場合は、追加回答が必要である。"
    )
    if summary["total_responses"] and summary["ci95_low"] > 0:
        interpretation = "95%信頼区間の下限が0を上回り、BASiS優位の傾向が人手評価でも確認された。"
    elif summary["total_responses"] and summary["ci95_high"] < 0:
        interpretation = "95%信頼区間の上限が0を下回り、Random優位の傾向が見られた。"
    axis_lines = [
        (
            f"- {AXIS_JA_LABELS[axis_key]}: 平均 {summary[f'{axis_key}_mean_basis_score']:.3f} "
            f"(重み {summary[f'{axis_key}_weight']:.2f})"
        )
        for axis_key in AXIS_KEYS
    ]

    lines = [
        "# BASiS vs Random ユーザ評価レポート",
        "",
        "## 実験設定",
        "",
        "- 比較対象: BASiS/Bayes-DPO と Random-DPO",
        "- 評価形式: 匿名化されたModel A/Bの5軸・5段階比較",
        "- 総合スコア: 各軸のBASiS基準スコアを、Oracle評価のweighted overallに対応する重みで合成",
        f"- 評価item: `{items_path.as_posix()}`",
        f"- 回答ファイル数: {len(response_files)}",
        "- 個人名はこの集計レポート・CSV・グラフには出力しない。",
        "",
        "## 集計結果",
        "",
        f"- 参加者ID数: {summary['unique_participants']}",
        f"- セッション数: {summary['unique_sessions']}",
        f"- 回答数: {summary['total_responses']}",
        f"- 評価item数: {summary['unique_items']}",
        f"- BASiS勝ち: {summary['basis_win_count']} ({format_rate(summary['basis_win_rate'])})",
        f"- Random勝ち: {summary['random_win_count']} ({format_rate(summary['random_win_rate'])})",
        f"- Tie: {summary['tie_count']} ({format_rate(summary['tie_rate'])})",
        f"- BASiS基準スコア平均: {summary['mean_basis_score']:.3f}",
        f"- 標準偏差: {summary['std_basis_score']:.3f}",
        f"- 標準誤差: {summary['se_basis_score']:.3f}",
        f"- 95%信頼区間: [{summary['ci95_low']:.3f}, {summary['ci95_high']:.3f}]",
        f"- 符号検定 p値(two-sided, ties除外): {p_value_text}",
        "",
        "## 軸別平均",
        "",
        *axis_lines,
        "",
        "## 解釈",
        "",
        interpretation,
        "",
        "## 発表用まとめ文",
        "",
        (
            f"研究室内の追加人手評価では、BASiS勝率は"
            f"{format_rate(summary['basis_win_rate'])}、Random勝率は"
            f"{format_rate(summary['random_win_rate'])}、Tieは"
            f"{format_rate(summary['tie_rate'])}であった。"
            f"BASiS基準スコア平均は{summary['mean_basis_score']:.2f}"
            f"（95% CI [{summary['ci95_low']:.2f}, {summary['ci95_high']:.2f}]）だった。"
        ),
        "",
        "## 入力回答ファイル",
        "",
    ]
    lines.extend(f"- `{path.as_posix()}`" for path in response_files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_bar_svg(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    y_label: str,
    path: Path,
    as_rate: bool = False,
) -> None:
    """シンプルな棒グラフSVGを生成する。"""
    width = 960
    height = 560
    margin_left = 92
    margin_bottom = 92
    margin_top = 72
    margin_right = 40
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    max_value = max(values) if values else 1.0
    if as_rate:
        max_value = max(1.0, max_value)
    elif max_value <= 0:
        max_value = 1.0
    bar_gap = 36
    bar_w = (chart_w - bar_gap * (len(labels) + 1)) / max(1, len(labels))
    colors = ["#2563eb", "#dc2626", "#737373", "#16a34a", "#9333ea"]

    def y_pos(value: float) -> float:
        return margin_top + chart_h - (value / max_value) * chart_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-size="28" font-family="Arial" font-weight="700">{title}</text>',
        f'<text x="28" y="{height / 2}" transform="rotate(-90 28 {height / 2})" text-anchor="middle" font-size="18" font-family="Arial">{y_label}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_h}" x2="{margin_left + chart_w}" y2="{margin_top + chart_h}" stroke="#222" stroke-width="2"/>',
    ]
    for tick in range(0, 6):
        value = max_value * tick / 5
        y = y_pos(value)
        label = f"{value * 100:.0f}%" if as_rate else f"{value:.0f}"
        parts.append(f'<line x1="{margin_left - 6}" y1="{y:.1f}" x2="{margin_left}" y2="{y:.1f}" stroke="#222"/>')
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="14" font-family="Arial">{label}</text>'
        )
        if tick:
            parts.append(
                f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + chart_w}" y2="{y:.1f}" stroke="#e5e7eb"/>'
            )
    for index, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + bar_gap + index * (bar_w + bar_gap)
        y = y_pos(value)
        h = margin_top + chart_h - y
        color = colors[index % len(colors)]
        value_label = f"{value * 100:.1f}%" if as_rate else f"{value:.0f}"
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="3"/>',
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-size="18" font-family="Arial" font-weight="700">{value_label}</text>',
                f'<text x="{x + bar_w / 2:.1f}" y="{margin_top + chart_h + 34}" text-anchor="middle" font-size="16" font-family="Arial">{label}</text>',
            ]
        )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def make_bar_png(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    y_label: str,
    path: Path,
    as_rate: bool = False,
) -> None:
    """PillowでPNG棒グラフを生成する。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - Pillowなし環境用
        raise RuntimeError("PNG生成にはPillowが必要です。") from exc

    width = 960
    height = 560
    margin_left = 92
    margin_bottom = 92
    margin_top = 72
    margin_right = 40
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    max_value = max(values) if values else 1.0
    if as_rate:
        max_value = max(1.0, max_value)
    elif max_value <= 0:
        max_value = 1.0
    bar_gap = 36
    bar_w = (chart_w - bar_gap * (len(labels) + 1)) / max(1, len(labels))
    colors = ["#2563eb", "#dc2626", "#737373", "#16a34a", "#9333ea"]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_title = ImageFont.load_default()
    font_text = ImageFont.load_default()

    def y_pos(value: float) -> float:
        return margin_top + chart_h - (value / max_value) * chart_h

    draw.text((width / 2, 28), title, fill="#111111", anchor="mm", font=font_title)
    draw.text((24, height / 2), y_label, fill="#111111", anchor="mm", font=font_text)
    draw.line((margin_left, margin_top, margin_left, margin_top + chart_h), fill="#222222", width=2)
    draw.line(
        (margin_left, margin_top + chart_h, margin_left + chart_w, margin_top + chart_h),
        fill="#222222",
        width=2,
    )
    for tick in range(0, 6):
        value = max_value * tick / 5
        y = y_pos(value)
        label = f"{value * 100:.0f}%" if as_rate else f"{value:.0f}"
        draw.line((margin_left - 6, y, margin_left, y), fill="#222222")
        draw.text((margin_left - 12, y), label, fill="#111111", anchor="rm", font=font_text)
        if tick:
            draw.line((margin_left, y, margin_left + chart_w, y), fill="#e5e7eb")
    for index, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + bar_gap + index * (bar_w + bar_gap)
        y = y_pos(value)
        h = margin_top + chart_h - y
        color = colors[index % len(colors)]
        value_label = f"{value * 100:.1f}%" if as_rate else f"{value:.0f}"
        draw.rectangle((x, y, x + bar_w, y + h), fill=color)
        draw.text((x + bar_w / 2, y - 12), value_label, fill="#111111", anchor="mm", font=font_text)
        draw.text((x + bar_w / 2, margin_top + chart_h + 30), label, fill="#111111", anchor="mm", font=font_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def basis_score_bucket(score: float) -> str:
    """重み付き総合スコアを発表用の範囲ラベルへ丸める。"""
    if score <= -1.5:
        return "<=-1.5"
    if score < -0.5:
        return "-1.5 to -0.5"
    if score <= 0.5:
        return "-0.5 to +0.5"
    if score < 1.5:
        return "+0.5 to +1.5"
    return ">=+1.5"


def make_diverging_bar_svg(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    path: Path,
) -> None:
    """負値と正値を含む横棒グラフSVGを生成する。"""
    width = 1080
    height = 600
    margin_left = 170
    margin_right = 64
    margin_top = 82
    margin_bottom = 88
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    row_gap = chart_h / max(1, len(labels))
    min_value = -2.0
    max_value = 2.0

    def x_pos(value: float) -> float:
        return margin_left + ((value - min_value) / (max_value - min_value)) * chart_w

    zero_x = x_pos(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-size="28" font-family="Arial" font-weight="700">{title}</text>',
    ]
    for tick in [-2, -1, 0, 1, 2]:
        x = x_pos(float(tick))
        stroke = "#111827" if tick == 0 else "#e5e7eb"
        width_attr = "2" if tick == 0 else "1"
        parts.append(
            f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + chart_h}" stroke="{stroke}" stroke-width="{width_attr}"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{margin_top + chart_h + 30}" text-anchor="middle" font-size="15" font-family="Arial">{tick:+d}</text>'
        )
    for index, (label, value) in enumerate(zip(labels, values)):
        y_center = margin_top + row_gap * index + row_gap / 2
        bar_h = min(42, row_gap * 0.55)
        value_x = x_pos(max(min(value, max_value), min_value))
        x = min(zero_x, value_x)
        bar_w = abs(value_x - zero_x)
        color = "#2563eb" if value > 0 else "#dc2626" if value < 0 else "#737373"
        text_x = value_x + 8 if value >= 0 else value_x - 8
        anchor = "start" if value >= 0 else "end"
        parts.extend(
            [
                f'<text x="{margin_left - 14}" y="{y_center + 5:.1f}" text-anchor="end" font-size="17" font-family="Arial">{label}</text>',
                f'<rect x="{x:.1f}" y="{y_center - bar_h / 2:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>',
                f'<text x="{text_x:.1f}" y="{y_center + 5:.1f}" text-anchor="{anchor}" font-size="17" font-family="Arial" font-weight="700">{value:+.2f}</text>',
            ]
        )
    parts.append(
        f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-size="15" font-family="Arial" fill="#475569">Positive values favor BASiS; negative values favor Random.</text>'
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def make_diverging_bar_png(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    path: Path,
) -> None:
    """負値と正値を含む横棒グラフPNGを生成する。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - Pillowなし環境用
        raise RuntimeError("PNG生成にはPillowが必要です。") from exc

    width = 1080
    height = 600
    margin_left = 170
    margin_right = 64
    margin_top = 82
    margin_bottom = 88
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    row_gap = chart_h / max(1, len(labels))
    min_value = -2.0
    max_value = 2.0
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_title = ImageFont.load_default()
    font_text = ImageFont.load_default()

    def x_pos(value: float) -> float:
        return margin_left + ((value - min_value) / (max_value - min_value)) * chart_w

    zero_x = x_pos(0.0)
    draw.text((width / 2, 28), title, fill="#111111", anchor="mm", font=font_title)
    for tick in [-2, -1, 0, 1, 2]:
        x = x_pos(float(tick))
        color = "#111827" if tick == 0 else "#e5e7eb"
        line_width = 2 if tick == 0 else 1
        draw.line((x, margin_top, x, margin_top + chart_h), fill=color, width=line_width)
        draw.text((x, margin_top + chart_h + 28), f"{tick:+d}", fill="#111111", anchor="mm", font=font_text)
    for index, (label, value) in enumerate(zip(labels, values)):
        y_center = margin_top + row_gap * index + row_gap / 2
        bar_h = min(42, row_gap * 0.55)
        value_x = x_pos(max(min(value, max_value), min_value))
        x = min(zero_x, value_x)
        bar_w = abs(value_x - zero_x)
        color = "#2563eb" if value > 0 else "#dc2626" if value < 0 else "#737373"
        text_x = value_x + 32 if value >= 0 else value_x - 32
        draw.text((margin_left - 12, y_center), label, fill="#111111", anchor="rm", font=font_text)
        draw.rectangle((x, y_center - bar_h / 2, x + bar_w, y_center + bar_h / 2), fill=color)
        draw.text((text_x, y_center), f"{value:+.2f}", fill="#111111", anchor="mm", font=font_text)
    draw.text(
        (width / 2, height - 24),
        "Positive values favor BASiS; negative values favor Random.",
        fill="#475569",
        anchor="mm",
        font=font_text,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_figures(rows: list[dict[str, Any]], summary: dict[str, Any], figures_dir: Path) -> None:
    """発表用グラフをPNG/SVGで保存する。"""
    winner_rates = [
        float(summary["basis_win_rate"]),
        float(summary["random_win_rate"]),
        float(summary["tie_rate"]),
    ]
    raw_ratings = [
        int(row[f"{axis_key}_rating_raw"])
        for row in rows
        for axis_key in AXIS_KEYS
    ]
    rating_counts = Counter(raw_ratings)
    score_bucket_labels = ["<=-1.5", "-1.5 to -0.5", "-0.5 to +0.5", "+0.5 to +1.5", ">=+1.5"]
    basis_score_counts = Counter(basis_score_bucket(float(row["overall_basis_score"])) for row in rows)
    figure_specs = [
        (
            ["BASiS win", "Random win", "Tie"],
            winner_rates,
            "Win rate",
            "Rate",
            "win_rate_bar",
            True,
        ),
        (
            ["1", "2", "3", "4", "5"],
            [rating_counts[value] for value in range(1, 6)],
            "Raw A/B rating distribution",
            "Count",
            "rating_distribution",
            False,
        ),
        (
            score_bucket_labels,
            [basis_score_counts[label] for label in score_bucket_labels],
            "Weighted BASiS score distribution",
            "Count",
            "basis_score_distribution",
            False,
        ),
    ]
    for labels, values, title, y_label, stem, as_rate in figure_specs:
        make_bar_svg(
            labels,
            values,
            title=title,
            y_label=y_label,
            path=figures_dir / f"{stem}.svg",
            as_rate=as_rate,
        )
        make_bar_png(
            labels,
            values,
            title=title,
            y_label=y_label,
            path=figures_dir / f"{stem}.png",
            as_rate=as_rate,
        )
    axis_labels = [label for _, label in EVALUATION_AXES]
    axis_values = [float(summary[f"{axis_key}_mean_basis_score"]) for axis_key in AXIS_KEYS]
    make_diverging_bar_svg(
        axis_labels,
        axis_values,
        title="Mean BASiS score by axis",
        path=figures_dir / "axis_mean_scores.svg",
    )
    make_diverging_bar_png(
        axis_labels,
        axis_values,
        title="Mean BASiS score by axis",
        path=figures_dir / "axis_mean_scores.png",
    )


def write_analysis_outputs(
    *,
    rows: list[dict[str, Any]],
    item_lookup: dict[str, dict[str, Any]],
    response_files: list[Path],
    items_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """CSV、レポート、図をまとめて出力する。"""
    if not rows:
        raise ValueError("分析対象の回答がありません。")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    participant_rows = grouped_summary(rows, "participant_id")
    item_rows = grouped_summary(rows, "item_id")

    normalized_fields = [
        "participant_id",
        "session_id",
        "item_id",
        "category",
        "stratum",
        "prompt",
        "displayed_order",
        "rating_raw",
        "basis_position",
        "random_position",
        "basis_score",
        "overall_basis_score",
        "winner",
        "comment",
        "timestamp",
        "source_file",
        "oracle_winner",
        "oracle_score_gap",
    ]
    for axis_key in AXIS_KEYS:
        normalized_fields.extend([f"{axis_key}_rating_raw", f"{axis_key}_basis_score"])
    summary_fields = [
        "total_responses",
        "unique_participants",
        "unique_sessions",
        "unique_items",
        "basis_win_count",
        "random_win_count",
        "tie_count",
        "basis_win_rate",
        "random_win_rate",
        "tie_rate",
        "mean_rating_raw",
        "rating_1_count",
        "rating_2_count",
        "rating_3_count",
        "rating_4_count",
        "rating_5_count",
        "mean_basis_score",
        "mean_overall_basis_score",
        "std_basis_score",
        "se_basis_score",
        "ci95_low",
        "ci95_high",
        "p_value_sign_test_two_sided",
        "generated_at",
    ]
    for axis_key in AXIS_KEYS:
        summary_fields.extend(
            [
                f"{axis_key}_weight",
                f"{axis_key}_mean_basis_score",
                f"{axis_key}_basis_win_count",
                f"{axis_key}_random_win_count",
                f"{axis_key}_tie_count",
                f"{axis_key}_basis_win_rate",
                f"{axis_key}_random_win_rate",
                f"{axis_key}_tie_rate",
            ]
        )
    participant_fields = [
        "participant_id",
        "responses",
        "basis_win_count",
        "random_win_count",
        "tie_count",
        "basis_win_rate",
        "random_win_rate",
        "tie_rate",
        "mean_basis_score",
        "mean_overall_basis_score",
        "std_basis_score",
        "se_basis_score",
    ]
    for axis_key in AXIS_KEYS:
        participant_fields.append(f"{axis_key}_mean_basis_score")
    item_fields = [
        "item_id",
        "category",
        "stratum",
        "responses",
        "basis_win_count",
        "random_win_count",
        "tie_count",
        "basis_win_rate",
        "random_win_rate",
        "tie_rate",
        "mean_basis_score",
        "mean_overall_basis_score",
        "std_basis_score",
        "se_basis_score",
        "oracle_winner",
        "oracle_score_gap",
        "prompt",
    ]
    for axis_key in AXIS_KEYS:
        item_fields.insert(-3, f"{axis_key}_mean_basis_score")

    write_csv(rows, output_dir / "normalized_responses.csv", normalized_fields)
    write_csv([summary], output_dir / "summary.csv", summary_fields)
    write_csv(participant_rows, output_dir / "participant_summary.csv", participant_fields)
    write_csv(item_rows, output_dir / "item_summary.csv", item_fields)
    write_report(
        rows=rows,
        summary=summary,
        response_files=response_files,
        items_path=items_path,
        output_path=output_dir / "report.md",
    )
    write_figures(rows, summary, output_dir / "figures")
    return {
        "summary": summary,
        "participant_summary": participant_rows,
        "item_summary": item_rows,
        "item_lookup_size": len(item_lookup),
    }


def main() -> None:
    """CLI entrypoint。"""
    args = parse_args()
    items_path = Path(args.items)
    responses_dir = Path(args.responses_dir)
    output_dir = Path(args.output_dir)
    response_files = discover_response_files(responses_dir, args.inputs)
    if not response_files:
        raise FileNotFoundError(f"回答JSONLが見つかりません: {responses_dir}")
    item_lookup = load_items(items_path)
    rows = load_and_normalize_responses(response_files, item_lookup)
    result = write_analysis_outputs(
        rows=rows,
        item_lookup=item_lookup,
        response_files=response_files,
        items_path=items_path,
        output_dir=output_dir,
    )
    summary = result["summary"]
    print(f"分析対象回答: {summary['total_responses']}")
    print(f"BASiS勝率: {summary['basis_win_rate']:.2%}")
    print(f"Random勝率: {summary['random_win_rate']:.2%}")
    print(f"Tie率: {summary['tie_rate']:.2%}")
    print(f"結果を書き出しました: {output_dir}")


if __name__ == "__main__":
    main()
