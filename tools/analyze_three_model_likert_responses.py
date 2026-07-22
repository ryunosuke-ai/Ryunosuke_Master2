"""3モデルLikert人手評価をprivate keyで復号し、参加者単位で検定する。"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from core.three_model_likert_survey import (
    RESPONSE_POSITIONS,
    connect_database,
    export_responses_csv,
    load_definition,
)
from scripts.analyze_oracle_three_model_significance import (
    effect_size,
    friedman_test,
    holm_adjust,
    paired_permutation_p,
)


MODELS = ("base", "basis", "random_dpo")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(probability * (len(ordered) - 1))))]


def bootstrap_ci(values: list[float], draws: int, rng: random.Random) -> tuple[float, float]:
    sampled = sorted(mean(values[rng.randrange(len(values))] for _ in values) for _ in range(draws))
    return percentile(sampled, 0.025), percentile(sampled, 0.975)


def load_private(path: Path) -> dict[str, dict[str, str]]:
    result = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["item_id"])] = {str(key): str(value) for key, value in row["position_to_model"].items()}
    return result


def load_ratings(database: Path, private: dict[str, dict[str, str]]) -> tuple[dict[str, Any], Counter[str]]:
    values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    choices: Counter[str] = Counter()
    with connect_database(database) as connection:
        rows = connection.execute("SELECT participant_id,item_id,ratings_json,final_choice FROM responses ORDER BY participant_id,item_id").fetchall()
    for row in rows:
        item_id = str(row["item_id"])
        if item_id not in private:
            raise ValueError(f"private answer keyにitemがありません: {item_id}")
        mapping = private[item_id]
        ratings = json.loads(row["ratings_json"])
        participant = str(row["participant_id"])
        for axis, positions in ratings.items():
            for position in RESPONSE_POSITIONS:
                values[axis][participant][mapping[position]].append(float(positions[position]))
        choice = str(row["final_choice"])
        if choice.startswith("応答") and choice[-1] in mapping:
            choices[mapping[choice[-1]]] += 1
        else:
            choices[choice] += 1
    return values, choices


def analyze(values: dict[str, Any], *, permutations: int, bootstrap: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    summaries, omnibus, posthoc = [], [], []
    comparisons = (("BASiS_vs_Base", "basis", "base"), ("BASiS_vs_Random-DPO", "basis", "random_dpo"), ("Base_vs_Random-DPO", "base", "random_dpo"))
    for axis, participants in sorted(values.items()):
        complete = {participant: {model: mean(scores[model]) for model in MODELS} for participant, scores in participants.items() if all(scores.get(model) for model in MODELS)}
        ids = sorted(complete)
        if len(ids) < 2:
            continue
        model_values = {model: [complete[participant][model] for participant in ids] for model in MODELS}
        for model in MODELS:
            low, high = bootstrap_ci(model_values[model], bootstrap, rng)
            summaries.append({"axis": axis, "model": model, "participants": len(ids), "mean": mean(model_values[model]), "std": stdev(model_values[model]), "participant_bootstrap_ci95_low": low, "participant_bootstrap_ci95_high": high})
        translated = {"base": model_values["base"], "bayes_dpo": model_values["basis"], "random_dpo": model_values["random_dpo"]}
        chi2, p_value, kendalls_w = friedman_test(translated)
        omnibus.append({"axis": axis, "participants": len(ids), "friedman_chi2": chi2, "p_value": p_value, "kendalls_w": kendalls_w, "significant": p_value < 0.05})
        if p_value >= 0.05:
            continue
        raw = []
        axis_rows = []
        for name, left, right in comparisons:
            differences = [complete[participant][left] - complete[participant][right] for participant in ids]
            p_raw = paired_permutation_p(differences, n_permutations=permutations, rng=rng)
            low, high = bootstrap_ci(differences, bootstrap, rng)
            raw.append(p_raw)
            axis_rows.append({"axis": axis, "comparison": name, "participants": len(ids), "mean_difference": mean(differences), "cohens_dz": effect_size(differences), "p_raw": p_raw, "participant_bootstrap_ci95_low": low, "participant_bootstrap_ci95_high": high})
        for row, adjusted in zip(axis_rows, holm_adjust(raw)):
            row["p_holm"] = adjusted
            row["significant"] = p_value < 0.05 and adjusted < 0.05
            posthoc.append(row)
    return summaries, omnibus, posthoc


def main() -> int:
    parser = argparse.ArgumentParser(description="3モデル人手評価の統計")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--private-answer-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    definition = load_definition(args.definition)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_responses_csv(args.database, definition, args.output_dir / "responses_long_private.csv")
    values, choices = load_ratings(args.database, load_private(args.private_answer_key))
    summary, omnibus, posthoc = analyze(values, permutations=args.permutations, bootstrap=args.bootstrap, seed=args.seed)
    write_csv(args.output_dir / "axis_model_summary.csv", summary)
    write_csv(args.output_dir / "friedman.csv", omnibus)
    write_csv(args.output_dir / "holm_posthoc.csv", posthoc)
    choice_total = sum(choices.values())
    write_csv(args.output_dir / "final_choice_counts.csv", [{"choice": key, "count": value, "win_rate": value / choice_total if choice_total else 0.0} for key, value in sorted(choices.items())])
    (args.output_dir / "metadata.json").write_text(json.dumps({"dataset": definition["dataset"], "analysis_unit": "participant", "permutations": args.permutations, "bootstrap": args.bootstrap, "seed": args.seed}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
