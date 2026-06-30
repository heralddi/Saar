from pathlib import Path

import pytest

from src.score_minimal_pairs import pair_stats, read_rows, summarize


def test_pair_stats_requires_both_sentences_correct_for_consistency():
    pair = [
        {"pair_id": "1", "category": "agreement", "label": "1", "prediction": "1"},
        {"pair_id": "1", "category": "agreement", "label": "0", "prediction": "0"},
    ]

    stats = pair_stats("1", pair)

    assert stats["item_accuracy"] == 1.0
    assert stats["pair_consistent"]


def test_pair_stats_rejects_bad_pair_shape():
    pair = [
        {"pair_id": "1", "category": "agreement", "label": "1", "prediction": "1"},
    ]

    with pytest.raises(ValueError):
        pair_stats("1", pair)


def test_summary_reports_consistency_gap_and_ci():
    rows = read_rows(Path("examples/minimal_pair_predictions.csv"))
    summary = summarize(rows, bootstrap_samples=200, seed=3)
    by_category = {row["category"]: row for row in summary}

    agreement = by_category["subject_verb_agreement"]
    assert agreement["item_accuracy"] == 0.667
    assert agreement["pair_consistency"] == 0.333
    assert agreement["consistency_gap"] == 0.333
    assert agreement["gap_ci_low"] <= agreement["consistency_gap"] <= agreement["gap_ci_high"]


def test_summary_reports_prediction_bias():
    rows = read_rows(Path("examples/minimal_pair_predictions.csv"))
    summary = summarize(rows, bootstrap_samples=200, seed=3)
    by_category = {row["category"]: row for row in summary}

    quantifier = by_category["quantifier_scope"]
    assert quantifier["predicted_accept_rate"] == 0.75
    assert quantifier["gold_accept_rate"] == 0.5
