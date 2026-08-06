"""
Runs the golden dataset through each configured agent version, computing
all 4 metrics (task accuracy, tool-use correctness/efficiency,
hallucination, latency) for every (question, version) pair.

Saves incrementally after EVERY question — safe to interrupt (Ctrl+C)
and resume later; already-completed (question, version) pairs are
automatically skipped on the next run.

Usage:
    python scripts/run_benchmark.py                          # all versions, all questions
    python scripts/run_benchmark.py --version qwen3-4b        # just one version
    python scripts/run_benchmark.py --limit 3                 # test with just 3 questions first
    python scripts/run_benchmark.py --version qwen3-4b --limit 3   # combine both, for cheap testing
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.eval.agent_runner import run_question_on_version
from src.eval.metrics import (
    extract_tools_called,
    extract_observations,
    extract_memory_context,
    compute_tool_selection_correctness,
    compute_tool_efficiency,
    judge_task_accuracy,
    judge_hallucination,
    time_agent_run,
)


def load_existing_results(results_path: Path) -> list[dict]:
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return []


def save_results(results_path: Path, results: list[dict]):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def run_one_version(version: dict, golden_data: list[dict], project3_path: Path,
                    results_dir: Path, judge_model: str, api_key: str, limit: int = None):
    version_id = version["id"]
    results_path = results_dir / f"{version_id}.json"
    existing = load_existing_results(results_path)
    done_questions = {r["question"] for r in existing}

    questions_to_run = golden_data[:limit] if limit else golden_data

    print(f"\n{'='*70}\nRunning version: {version_id}\n{'='*70}")
    print(f"{len(existing)} already completed, {len(questions_to_run)} total in this run\n")

    for entry in questions_to_run:
        question = entry["question"]
        if question in done_questions:
            print(f"[skip, already done] {question[:60]}")
            continue

        print(f"[running] {question[:60]}")
        session_id = str(uuid.uuid4())

        try:
            # Wraps run_question_on_version to also capture latency —
            # time_agent_run expects an object with .run(question,
            # verbose), so wrap run_question_on_version in a tiny adapter
            class _Adapter:
                def run(self, q, verbose=False):
                    return run_question_on_version(q, session_id, version, project3_path)

            run_result = time_agent_run(_Adapter(), question)
        except Exception as e:
            print(f"  ERROR running agent: {e}")
            run_result = {"answer": "", "transcript": "", "iterations": 0, "latency_seconds": None}

        tools_called = extract_tools_called(run_result.get("transcript", ""))
        tool_selection = compute_tool_selection_correctness(entry["category"], tools_called)
        tool_efficiency = compute_tool_efficiency(tools_called)
        retrieved_context = extract_observations(run_result.get("transcript", ""))
        memory_context = extract_memory_context(run_result.get("transcript", ""))
        # Combined: everything the model actually had access to when
        # generating its answer — tool outputs from THIS run, plus any
        # real memory context (short/long-term) also present in the
        # prompt. A claim grounded in either is faithful, not hallucinated.
        # NOTE: previously pre-combined into one "full_context" string,
        # but the judge prompt still labeled the whole blob "retrieved
        # source content" — a framing that specifically implies tool
        # retrieval, causing the judge to not recognize the memory
        # portion as legitimate grounding (confirmed via a real test:
        # the judge's own reasoning only referenced the tool-failure
        # error message, ignoring memory content that was technically
        # present in the combined string). Now passed as separate,
        # clearly-labeled parameters instead — see judge_hallucination.

        agent_run_failed = not run_result.get("answer", "").strip()

        if agent_run_failed:
            # Don't call either judge on an empty answer — a real test
            # showed this produces a confusing judge response ("I need to
            # see the agent's answer...") instead of a clear, honest
            # signal that the AGENT RUN ITSELF failed upstream. Recording
            # this explicitly makes agent-run failures visibly distinct
            # from genuine judged incorrectness/hallucination in the
            # results, rather than looking like a None-valued judge quirk.
            print("  SKIPPED judging: agent produced no answer (see agent error above)")
            accuracy = {"correct": None, "reasoning": "Agent run failed — no answer to judge."}
            hallucination = {"hallucinated": None, "reasoning": "Agent run failed — no answer to judge.", "unsupported_claims": []}
        else:
            try:
                accuracy = judge_task_accuracy(
                    question, entry["ground_truth"], run_result["answer"], api_key, judge_model
                )
            except Exception as e:
                print(f"  ERROR judging accuracy: {e}")
                accuracy = {"correct": None, "reasoning": f"Judge error: {e}"}

            try:
                hallucination = judge_hallucination(
                    question, run_result["answer"], api_key, judge_model,
                    tool_context=retrieved_context, memory_context=memory_context,
                )
            except Exception as e:
                print(f"  ERROR judging hallucination: {e}")
                hallucination = {"hallucinated": None, "reasoning": f"Judge error: {e}", "unsupported_claims": []}

        result_row = {
            "question": question,
            "category": entry["category"],
            "version_id": version_id,
            "agent_answer": run_result["answer"],
            "iterations": run_result.get("iterations"),
            "latency_seconds": run_result.get("latency_seconds"),
            "tools_called": tools_called,
            "tool_selection_correct": tool_selection["correct"],
            "expected_tools": tool_selection["expected_tools"],
            "tool_efficiency": tool_efficiency,
            "accuracy_correct": accuracy["correct"],
            "accuracy_reasoning": accuracy["reasoning"],
            "hallucinated": hallucination["hallucinated"],
            "hallucination_reasoning": hallucination["reasoning"],
            "unsupported_claims": hallucination["unsupported_claims"],
            "retrieved_context": retrieved_context,
            "memory_context": memory_context,
            "agent_run_failed": agent_run_failed,
        }

        existing.append(result_row)
        save_results(results_path, existing)
        print(f"  -> correct={accuracy['correct']}, hallucinated={hallucination['hallucinated']}, "
             f"tools_ok={tool_selection['correct']}, latency={run_result.get('latency_seconds')}")

    print(f"\nVersion {version_id} complete: {len(existing)} results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None, help="Run just one agent version by id")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N golden questions (for cheap testing)")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    project3_path = Path(cfg["target_project"]["path"]).resolve()
    results_dir = Path(cfg["leaderboard"]["results_dir"])
    judge_model = cfg["metrics"]["judge_model"]

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set.")

    with open(cfg["golden_dataset"]["path"]) as f:
        golden_data = json.load(f)

    versions = cfg["agent_versions"]
    if args.version:
        versions = [v for v in versions if v["id"] == args.version]
        if not versions:
            raise ValueError(f"No agent version found with id={args.version!r}")

    for version in versions:
        run_one_version(version, golden_data, project3_path, results_dir, judge_model, api_key, args.limit)


if __name__ == "__main__":
    main()
