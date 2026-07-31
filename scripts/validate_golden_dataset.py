"""
Validates the golden dataset for structural consistency: correct counts
per category, no duplicate questions, no missing/empty required fields,
and no leftover known-bad patterns from issues found and fixed while
building this dataset (e.g. the confidence-interval probability
misinterpretation, or a drafting refusal accidentally saved as if it
were real ground truth).

Usage:
    python scripts/validate_golden_dataset.py
"""

import json
from pathlib import Path

GOLDEN_PATH = Path("data/golden_qa_pairs.json")
EXPECTED_CATEGORIES = ["literature_only", "yield_only", "multi_tool", "out_of_scope"]
EXPECTED_PER_CATEGORY = 10

# Known-bad patterns found and fixed during dataset construction — kept
# here as a permanent regression check, not just a one-off diagnostic.
BAD_PATTERNS = [
    "DRAFTING FAILED",
    "probability of falling",
    "statistically likely to fall",
]


def main():
    with open(GOLDEN_PATH) as f:
        data = json.load(f)

    print(f"Total entries: {len(data)}\n")

    # Category counts
    by_category = {}
    for d in data:
        by_category[d["category"]] = by_category.get(d["category"], 0) + 1

    print("Category breakdown:")
    all_categories_ok = True
    for cat in EXPECTED_CATEGORIES:
        count = by_category.get(cat, 0)
        ok = count == EXPECTED_PER_CATEGORY
        all_categories_ok = all_categories_ok and ok
        status = "OK" if ok else "MISMATCH"
        print(f"  {cat}: {count}/{EXPECTED_PER_CATEGORY} [{status}]")
    print()

    # Duplicate questions
    questions = [d["question"] for d in data]
    duplicates = sorted(set(q for q in questions if questions.count(q) > 1))
    print(f"Duplicate questions: {len(duplicates)}")
    for q in duplicates:
        print(f"  - {q}")
    print()

    # Missing/empty required fields
    issues = []
    for i, d in enumerate(data):
        for field in ["question", "category", "ground_truth"]:
            if not d.get(field, "").strip():
                issues.append(f"Entry {i}: missing/empty '{field}'")
    print(f"Missing/empty fields: {len(issues)}")
    for issue in issues:
        print(f"  - {issue}")
    print()

    # Leftover known-bad patterns
    flagged = []
    for d in data:
        for pattern in BAD_PATTERNS:
            if pattern in d["ground_truth"]:
                flagged.append((d["question"], pattern))
    print(f"Leftover known-bad patterns: {len(flagged)}")
    for q, p in flagged:
        print(f'  - "{p}" in: {q}')
    print()

    all_ok = (
        all_categories_ok
        and len(duplicates) == 0
        and len(issues) == 0
        and len(flagged) == 0
        and len(data) == len(EXPECTED_CATEGORIES) * EXPECTED_PER_CATEGORY
    )
    print("=" * 50)
    print("RESULT: ALL CHECKS PASSED" if all_ok else "RESULT: ISSUES FOUND — see above")
    print("=" * 50)


if __name__ == "__main__":
    main()
