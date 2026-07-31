"""
Validates the LLM-judge metrics (judge_task_accuracy, judge_hallucination)
against 3 clear synthetic test cases with known expected verdicts.

Uses claude-haiku-4-5 (cheap) rather than the configured judge model
(Opus) — the JSON-parsing mechanics being validated here are identical
regardless of which model judges, so this validates the code correctly
without spending real budget on the more expensive judge model before
the actual benchmark run.

Usage:
    python scripts/validate_metrics.py
"""

import os
import sys

sys.path.insert(0, ".")

from src.eval.metrics import judge_task_accuracy, judge_hallucination


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")

    print("=" * 60)
    print("Test 1: obviously CORRECT answer (expect correct=True)")
    print("=" * 60)
    r1 = judge_task_accuracy(
        question="What corn yield should I expect in Illinois in 2024?",
        ground_truth="169.82 bushels per acre, with a 95% confidence interval of 147.05 to 193.28.",
        agent_answer="The predicted yield is 169.82 bu/acre, 95% CI 147.05-193.28.",
        api_key=api_key, judge_model="claude-haiku-4-5",
    )
    print(r1)
    print()

    print("=" * 60)
    print("Test 2: obviously WRONG answer (expect correct=False)")
    print("=" * 60)
    r2 = judge_task_accuracy(
        question="What corn yield should I expect in Illinois in 2024?",
        ground_truth="169.82 bushels per acre.",
        agent_answer="The predicted yield is 250 bushels per acre.",
        api_key=api_key, judge_model="claude-haiku-4-5",
    )
    print(r2)
    print()

    print("=" * 60)
    print("Test 3: obviously fabricated answer (expect hallucinated=True)")
    print("=" * 60)
    r3 = judge_hallucination(
        question="What corn yield should I expect in Illinois in 2024?",
        ground_truth="169.82 bushels per acre, based on the yield prediction model.",
        agent_answer="169.82 bushels per acre, according to a 2023 USDA field study of 500 farms across the Midwest.",
        api_key=api_key, judge_model="claude-haiku-4-5",
    )
    print(r3)
    print()

    print("=" * 60)
    all_pass = r1["correct"] is True and r2["correct"] is False and r3["hallucinated"] is True
    print("RESULT:", "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED — review above")
    print("=" * 60)


if __name__ == "__main__":
    main()
