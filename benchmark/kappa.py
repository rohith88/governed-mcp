"""
T-06: Cohen's kappa inter-rater agreement calculator.

Computes Cohen's kappa for benchmark task annotations between two raters.
Target: kappa > 0.92 before freezing the benchmark.

Rater input format (JSON):
  [
    {"task_id": "DB_001", "correct_tool": "dev_database_query", "difficulty": "easy"},
    ...
  ]

Usage:
    python benchmark/kappa.py --rater1 annotations/rater1.json --rater2 annotations/rater2.json
    python benchmark/kappa.py --rater1 r1.json --rater2 r2.json --field correct_tool
    python benchmark/kappa.py --rater1 r1.json --rater2 r2.json --field difficulty
"""

import argparse
import json
from pathlib import Path
from collections import Counter


def cohen_kappa(rater1_labels: list, rater2_labels: list) -> float:
    """
    Compute Cohen's kappa for two lists of categorical labels.
    Assumes paired annotations (same task order in both lists).
    """
    assert len(rater1_labels) == len(rater2_labels), \
        "Rater label lists must be the same length."

    n = len(rater1_labels)
    categories = sorted(set(rater1_labels) | set(rater2_labels))

    # Observed agreement
    observed_agree = sum(a == b for a, b in zip(rater1_labels, rater2_labels)) / n

    # Expected agreement under chance
    r1_counts = Counter(rater1_labels)
    r2_counts = Counter(rater2_labels)
    expected_agree = sum(
        (r1_counts.get(cat, 0) / n) * (r2_counts.get(cat, 0) / n)
        for cat in categories
    )

    if expected_agree == 1.0:
        return 1.0  # Degenerate case: all labels same

    kappa = (observed_agree - expected_agree) / (1 - expected_agree)
    return kappa


def per_category_agreement(rater1_labels: list, rater2_labels: list) -> dict[str, dict]:
    """Compute per-category precision/recall/F1 for disagrement analysis."""
    categories = sorted(set(rater1_labels) | set(rater2_labels))
    results = {}
    for cat in categories:
        tp = sum(a == cat and b == cat for a, b in zip(rater1_labels, rater2_labels))
        fp = sum(a != cat and b == cat for a, b in zip(rater1_labels, rater2_labels))
        fn = sum(a == cat and b != cat for a, b in zip(rater1_labels, rater2_labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[cat] = {"precision": precision, "recall": recall, "f1": f1, "count_r1": rater1_labels.count(cat)}
    return results


def load_annotations(path: str, field: str) -> tuple[list[str], list[str]]:
    """Load task_ids and field values from an annotation file."""
    with open(path) as f:
        data = json.load(f)
    task_ids = [item["task_id"] for item in data]
    labels = [str(item.get(field, "MISSING")) for item in data]
    return task_ids, labels


def main():
    parser = argparse.ArgumentParser(description="Compute Cohen's kappa for benchmark annotations")
    parser.add_argument("--rater1", required=True, help="Path to rater 1 annotations JSON")
    parser.add_argument("--rater2", required=True, help="Path to rater 2 annotations JSON")
    parser.add_argument("--field", default="correct_tool",
                        help="Field to compare (default: correct_tool; also try: difficulty)")
    parser.add_argument("--threshold", type=float, default=0.92,
                        help="Minimum kappa threshold to pass (default: 0.92)")
    args = parser.parse_args()

    if not Path(args.rater1).exists():
        print(f"ERROR: Rater 1 file not found: {args.rater1}")
        return

    if not Path(args.rater2).exists():
        print(f"ERROR: Rater 2 file not found: {args.rater2}")
        return

    ids1, labels1 = load_annotations(args.rater1, args.field)
    ids2, labels2 = load_annotations(args.rater2, args.field)

    # Align by task_id
    id_to_label1 = dict(zip(ids1, labels1))
    id_to_label2 = dict(zip(ids2, labels2))
    common_ids = sorted(set(ids1) & set(ids2))

    if not common_ids:
        print("ERROR: No common task_ids found between the two rater files.")
        return

    aligned1 = [id_to_label1[tid] for tid in common_ids]
    aligned2 = [id_to_label2[tid] for tid in common_ids]

    kappa = cohen_kappa(aligned1, aligned2)
    observed_agree = sum(a == b for a, b in zip(aligned1, aligned2)) / len(aligned1)
    disagreements = [(tid, a, b) for tid, a, b in zip(common_ids, aligned1, aligned2) if a != b]

    print(f"\n── Cohen's Kappa Report ──────────────────────────────────────")
    print(f"  Field:            {args.field}")
    print(f"  Tasks compared:   {len(common_ids)}")
    print(f"  Only in rater1:   {len(ids1) - len(common_ids)}")
    print(f"  Only in rater2:   {len(ids2) - len(common_ids)}")
    print(f"  Observed agree:   {observed_agree:.4f} ({sum(a==b for a,b in zip(aligned1,aligned2))}/{len(aligned1)})")
    print(f"  Cohen's kappa:    {kappa:.4f}")
    print(f"  Threshold:        {args.threshold}")
    print(f"  Status:           {'PASS ✓' if kappa >= args.threshold else 'FAIL ✗ — benchmark not frozen'}")

    if disagreements:
        print(f"\n  Disagreements ({len(disagreements)} tasks):")
        for tid, a, b in disagreements[:20]:
            print(f"    {tid}: rater1={a!r}  rater2={b!r}")
        if len(disagreements) > 20:
            print(f"    ... and {len(disagreements) - 20} more")

    print(f"\n  Per-category agreement:")
    per_cat = per_category_agreement(aligned1, aligned2)
    for cat, stats in sorted(per_cat.items(), key=lambda x: -x[1]["count_r1"])[:15]:
        print(f"    {cat:<40} F1={stats['f1']:.3f}  P={stats['precision']:.3f}  R={stats['recall']:.3f}  n={stats['count_r1']}")

    print()


if __name__ == "__main__":
    main()
