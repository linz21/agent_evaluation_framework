"""
Comprehensive validation for benchmark result files — checks for every
real issue found and fixed throughout this project's development:

1. Structural failures (agent run errors, empty answers, missing latency)
2. Runaway citation-bracket artifacts (a real bug found in Project 2's
   own generator — greedy decoding with no repetition control)
3. Leftover "Final Answer:" text (a real regression in the Crop Advisory
   ReAct Agent's own repeated-Final-Answer-block fix)
4. Judge call gaps (accuracy_correct or hallucinated left as None,
   e.g. from a judge API error, deprecated model, or credit exhaustion)

Run this on every result file before trusting it for analysis.

Usage:
    python scripts/validate_results.py                      # both versions
    python scripts/validate_results.py --version qwen3-4b    # just one
"""

import argparse
import json
import re
from pathlib import Path

EXPECTED_CATEGORIES = ["literature_only", "yield_only", "multi_tool", "out_of_scope"]
EXPECTED_PER_CATEGORY = 10
VERSIONS = ["qwen3-4b", "claude-sonnet-4.5"]


def validate_one_file(version_id: str) -> bool:
    path = Path(f"data/results/{version_id}.json")
    if not path.exists():
        print(f"=== {version_id} ===")
        print(f"  FILE NOT FOUND: {path}")
        return False

    with open(path) as f:
        data = json.load(f)

    print(f"=== {version_id} ===")
    print(f"{len(data)} total entries")

    by_category = {}
    for d in data:
        by_category[d["category"]] = by_category.get(d["category"], 0) + 1
    categories_ok = True
    for cat in EXPECTED_CATEGORIES:
        count = by_category.get(cat, 0)
        ok = count == EXPECTED_PER_CATEGORY
        categories_ok = categories_ok and ok
        print(f"  {cat}: {count}/{EXPECTED_PER_CATEGORY} [{'OK' if ok else 'MISMATCH'}]")

    bad = [d for d in data if d.get("agent_run_failed")
           or not d.get("agent_answer", "").strip()
           or d.get("latency_seconds") is None]
    print(f"{len(bad)} structurally bad entries")
    for d in bad:
        print(f"  - {d['question']}")

    artifacts = [d["question"] for d in data if re.findall(
        r"(?:\[\d+\]\s*){8,}", d.get("retrieved_context", "") + d.get("agent_answer", ""))]
    print(f"{len(artifacts)} entries with runaway bracket artifacts")
    for q in artifacts:
        print(f"  - {q}")

    repeats = [d["question"] for d in data if d.get("agent_answer", "").count("Final Answer:") > 0]
    print(f"{len(repeats)} entries with leftover 'Final Answer:' text")
    for q in repeats:
        print(f"  - {q}")

    reminder_leaks = [d["question"] for d in data if "REMINDER:" in d.get("agent_answer", "")]
    print(f"{len(reminder_leaks)} entries with leaked prompt REMINDER text")
    for q in reminder_leaks:
        print(f"  - {q}")

    none_acc = [d["question"] for d in data if d["accuracy_correct"] is None]
    none_hall = [d["question"] for d in data if d["hallucinated"] is None]
    print(f"{len(none_acc)} accuracy judge gaps, {len(none_hall)} hallucination judge gaps")
    for q in set(none_acc + none_hall):
        print(f"  - {q}")

    all_clean = (
        categories_ok and len(data) == len(EXPECTED_CATEGORIES) * EXPECTED_PER_CATEGORY
        and len(bad) == 0 and len(artifacts) == 0 and len(repeats) == 0
        and len(reminder_leaks) == 0
        and len(none_acc) == 0 and len(none_hall) == 0
    )
    print(f"RESULT: {'CLEAN' if all_clean else 'ISSUES FOUND'}")
    print()
    return all_clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None, choices=VERSIONS)
    args = parser.parse_args()

    versions = [args.version] if args.version else VERSIONS
    results = [validate_one_file(v) for v in versions]

    print("=" * 50)
    print("ALL FILES CLEAN" if all(results) else "SOME FILES HAVE ISSUES — see above")
    print("=" * 50)


if __name__ == "__main__":
    main()
