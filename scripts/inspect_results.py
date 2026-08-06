"""
Diagnostic: prints full detail for literature_only results, to inspect
whether the correct=False/hallucinated=True pattern seen in early
testing reflects genuine model behavior or a bug in the harness itself
(e.g. a malformed answer being passed to the judge).

Usage:
    python scripts/inspect_results.py --category literature_only
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None)
    parser.add_argument("--file", default="data/results/claude-sonnet-4.5.json")
    args = parser.parse_args()

    with open(args.file) as f:
        results = json.load(f)

    if args.category:
        results = [r for r in results if r["category"] == args.category]

    for r in results:
        print("=" * 70)
        print(f"QUESTION: {r['question']}")
        print(f"CATEGORY: {r['category']}")
        print(f"TOOLS CALLED: {r['tools_called']}  (expected: {r['expected_tools']})")
        print(f"ITERATIONS: {r['iterations']}")
        print()
        print(f"AGENT ANSWER:\n{r['agent_answer']}")
        print()
        print(f"ACCURACY: correct={r['accuracy_correct']}")
        print(f"  reasoning: {r['accuracy_reasoning']}")
        print()
        print(f"HALLUCINATION: hallucinated={r['hallucinated']}")
        print(f"  reasoning: {r['hallucination_reasoning']}")
        print(f"  unsupported_claims: {r['unsupported_claims']}")
        print()
        print(f"RETRIEVED CONTEXT (tool observations from this run):\n{r.get('retrieved_context', '(not saved in this result)')}")
        print()
        print(f"MEMORY CONTEXT (real prior interactions the model saw):\n{r.get('memory_context', '(not saved in this result)') or '(none for this question)'}")
        print()


if __name__ == "__main__":
    main()
