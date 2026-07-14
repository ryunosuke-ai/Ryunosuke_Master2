"""scoring途中結果からBASiS選別可能件数を高速に測る。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from core.transition_bayes_model import load_transition_bayes_model
from tools.extract_high_posterior_dialogues import (
    derive_selection_labels_from_model,
    select_high_posterior_records,
)


def measure_pool(
    scored: list[dict],
    *,
    model_path: str | Path,
    method: str,
    margin: float,
    per_dialogue_limit: int = 3,
) -> dict:
    model = load_transition_bayes_model(model_path)
    labels = derive_selection_labels_from_model(
        model, method=method, minimum_margin=margin
    )
    selected = select_high_posterior_records(
        scored,
        min_posterior=0.0,
        max_records=None,
        target_records=None,
        sort_by_posterior=False,
        sort_by_selection=True,
        per_dialogue_limit=per_dialogue_limit,
        prefer_states=labels["prefer_states"],
        prefer_observations=labels["prefer_observations"],
        low_priority_states=labels["low_priority_states"],
        exclude_states=labels["exclude_states"],
        exclude_observations=labels["exclude_observations"],
        require_preferred=True,
    )
    return {
        "scored_records": len(scored),
        "scored_conversations": len({row.get("conversation_id") for row in scored}),
        "eligible_records": len(selected),
        "eligible_conversations": len(
            {row.get("conversation_id") for row in selected}
        ),
        "eligible_observation_distribution": dict(
            sorted(Counter(str(row.get("observation")) for row in selected).items())
        ),
        "fallback_count": sum(bool(row.get("llm_error")) for row in scored),
        "label_derivation": labels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BASiS選別可能件数の測定")
    parser.add_argument("--input", required=True)
    parser.add_argument("--bayes-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--method",
        choices=("mean_difference", "state_specific_margin"),
        default="state_specific_margin",
    )
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--required", type=int, default=0)
    parser.add_argument("--history", help="batchごとの進捗を追記するJSONL")
    args = parser.parse_args()
    scored = [
        json.loads(line)
        for line in Path(args.input).open(encoding="utf-8")
        if line.strip()
    ]
    report = measure_pool(
        scored,
        model_path=args.bayes_model,
        method=args.method,
        margin=args.margin,
    )
    report["required_records"] = args.required
    report["sufficient"] = report["eligible_records"] >= args.required
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.history:
        history = Path(args.history)
        history.parent.mkdir(parents=True, exist_ok=True)
        previous: dict | None = None
        if history.exists():
            with history.open(encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        previous = json.loads(line)
        progress_key = (
            report["scored_records"],
            report["eligible_records"],
            report["fallback_count"],
        )
        previous_key = (
            previous.get("scored_records"),
            previous.get("eligible_records"),
            previous.get("fallback_count"),
        ) if previous else None
        if progress_key != previous_key:
            with history.open("a", encoding="utf-8") as file:
                file.write(json.dumps(report, ensure_ascii=False) + "\n")
    print(
        "[selection pool] "
        f"eligible={report['eligible_records']}/{args.required} "
        f"scored={report['scored_records']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
