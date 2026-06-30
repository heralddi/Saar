from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

REQUIRED_COLUMNS = {"pair_id", "category", "sentence", "label", "prediction"}
BINARY_VALUES = {"0", "1"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("input file has no rows")

    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")

    for row in rows:
        if row["label"] not in BINARY_VALUES or row["prediction"] not in BINARY_VALUES:
            raise ValueError("label and prediction must be 0 or 1")

    return rows


def is_correct(row: dict[str, str]) -> bool:
    return row["label"] == row["prediction"]


def group_by_pair(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    pairs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    return dict(pairs)


def check_pair(pair_id: str, pair_rows: list[dict[str, str]]) -> None:
    if len(pair_rows) != 2:
        raise ValueError(f"pair {pair_id} must have exactly two rows")

    labels = {row["label"] for row in pair_rows}
    if labels != BINARY_VALUES:
        raise ValueError(f"pair {pair_id} must have one 0 label and one 1 label")

    categories = {row["category"] for row in pair_rows}
    if len(categories) != 1:
        raise ValueError(f"pair {pair_id} has mixed categories")


def pair_stats(pair_id: str, pair_rows: list[dict[str, str]]) -> dict[str, str | float | bool]:
    check_pair(pair_id, pair_rows)
    item_accuracy = mean(1 if is_correct(row) else 0 for row in pair_rows)
    pair_consistent = item_accuracy == 1.0
    predicted_accept_rate = mean(int(row["prediction"]) for row in pair_rows)
    gold_accept_rate = mean(int(row["label"]) for row in pair_rows)

    return {
        "pair_id": pair_id,
        "category": pair_rows[0]["category"],
        "item_accuracy": item_accuracy,
        "pair_consistent": pair_consistent,
        "predicted_accept_rate": predicted_accept_rate,
        "gold_accept_rate": gold_accept_rate,
    }


def summarize(rows: list[dict[str, str]], bootstrap_samples: int = 1000, seed: int = 7) -> list[dict[str, str | int | float]]:
    pairs = [pair_stats(pair_id, pair_rows) for pair_id, pair_rows in group_by_pair(rows).items()]
    by_category: dict[str, list[dict[str, str | float | bool]]] = defaultdict(list)
    for pair in pairs:
        by_category[str(pair["category"])].append(pair)

    summary = []
    for category, category_pairs in sorted(by_category.items()):
        item_accuracy = mean(float(pair["item_accuracy"]) for pair in category_pairs)
        pair_consistency = mean(1 if pair["pair_consistent"] else 0 for pair in category_pairs)
        consistency_gap = item_accuracy - pair_consistency
        ci_low, ci_high = bootstrap_gap_ci(category_pairs, bootstrap_samples, seed)
        summary.append(
            {
                "category": category,
                "pairs": len(category_pairs),
                "items": len(category_pairs) * 2,
                "item_accuracy": round(item_accuracy, 3),
                "pair_consistency": round(pair_consistency, 3),
                "consistency_gap": round(consistency_gap, 3),
                "gap_ci_low": round(ci_low, 3),
                "gap_ci_high": round(ci_high, 3),
                "predicted_accept_rate": round(mean(float(pair["predicted_accept_rate"]) for pair in category_pairs), 3),
                "gold_accept_rate": round(mean(float(pair["gold_accept_rate"]) for pair in category_pairs), 3),
            }
        )

    return summary


def bootstrap_gap_ci(pairs: list[dict[str, str | float | bool]], samples: int, seed: int) -> tuple[float, float]:
    if samples <= 0:
        return (0.0, 0.0)

    rng = random.Random(seed)
    gaps = []
    for _ in range(samples):
        draw = [rng.choice(pairs) for _ in pairs]
        item_accuracy = mean(float(pair["item_accuracy"]) for pair in draw)
        pair_consistency = mean(1 if pair["pair_consistent"] else 0 for pair in draw)
        gaps.append(item_accuracy - pair_consistency)

    gaps.sort()
    low = gaps[int(0.025 * (samples - 1))]
    high = gaps[int(0.975 * (samples - 1))]
    return low, high


def write_summary(rows: list[dict[str, str | int | float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "pairs",
        "items",
        "item_accuracy",
        "pair_consistency",
        "consistency_gap",
        "gap_ci_low",
        "gap_ci_high",
        "predicted_accept_rate",
        "gold_accept_rate",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = read_rows(args.input)
    summary = summarize(rows, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    write_summary(summary, args.output)


if __name__ == "__main__":
    main()
