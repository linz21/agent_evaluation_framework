"""
Retries ONLY the hallucination judge call for entries where it previously
failed (hallucinated=None), reusing the already-saved agent_answer and
context rather than re-running the full agent — the agent's own answer
succeeded fine; only the judge call itself failed (e.g. an empty
response from an overly-cautious safety classifier trigger on certain
technical vocabulary).

Usage:
    python scripts/retry_failed_judgments.py --version claude-sonnet-4.5
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import judge_hallucination, judge_task_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--judge-model", default=None,
                        help="Override the configured judge model (e.g. for a stuck "
                             "entry that reproducibly fails with the default judge)")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set.")

    results_path = Path(f"data/results/{args.version}.json")
    with open(results_path) as f:
        data = json.load(f)

    with open("configs/config.yaml") as f:
        import yaml
        cfg = yaml.safe_load(f)
    judge_model = args.judge_model or cfg["metrics"]["judge_model"]
    print(f"Using judge model: {judge_model}")

    retried = 0
    for entry in data:
        if entry.get("hallucinated") is None:
            print(f"Retrying hallucination judge for: {entry['question'][:60]}")
            try:
                result = judge_hallucination(
                    entry["question"], entry["agent_answer"], api_key, judge_model,
                    tool_context=entry.get("retrieved_context"),
                    memory_context=entry.get("memory_context"),
                )
                entry["hallucinated"] = result["hallucinated"]
                entry["hallucination_reasoning"] = result["reasoning"]
                entry["unsupported_claims"] = result["unsupported_claims"]
                print(f"  -> hallucinated={result['hallucinated']}")
                retried += 1
            except Exception as e:
                print(f"  STILL FAILED: {e}")

        if entry.get("accuracy_correct") is None:
            print(f"Retrying accuracy judge for: {entry['question'][:60]}")
            try:
                # Need the golden dataset's ground_truth for this one
                with open("data/golden_qa_pairs.json") as f:
                    golden = json.load(f)
                gt = next(g["ground_truth"] for g in golden if g["question"] == entry["question"])
                result = judge_task_accuracy(
                    entry["question"], gt, entry["agent_answer"], api_key, judge_model
                )
                entry["accuracy_correct"] = result["correct"]
                entry["accuracy_reasoning"] = result["reasoning"]
                print(f"  -> correct={result['correct']}")
                retried += 1
            except Exception as e:
                print(f"  STILL FAILED: {e}")

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n{retried} judge calls retried and saved.")


if __name__ == "__main__":
    main()
