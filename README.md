# Saar

Small tools for evaluating language understanding.

## Minimal-pair contrast evaluation

This script scores model predictions on linguistic minimal pairs. It compares single-sentence accuracy with pair consistency and reports the gap with a small bootstrap confidence interval. The gap is useful when a model gets many labels right but does not reliably prefer the acceptable sentence within a controlled pair.

Run the example:

```bash
python src/score_minimal_pairs.py examples/minimal_pair_predictions.csv --output outputs/minimal_pair_summary.csv
```

The input needs one row per sentence with these columns: `pair_id`, `category`, `sentence`, `label`, and `prediction`. Each pair must have exactly one acceptable sentence and one unacceptable sentence.
