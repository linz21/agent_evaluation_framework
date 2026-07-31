"""
Interactive golden dataset builder — walks through all ~100 candidate
questions (from src/eval/question_bank.py), drafts a real-data-grounded
answer for each, and lets you review/edit/skip before saving. Saves
incrementally after every accepted question, so it's safe to stop and
resume across multiple sessions.

Usage:
    python scripts/build_golden_dataset.py                 # all categories
    python scripts/build_golden_dataset.py --category literature_only
    python scripts/build_golden_dataset.py --category yield_only
    python scripts/build_golden_dataset.py --category multi_tool
    python scripts/build_golden_dataset.py --category out_of_scope
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.eval.question_bank import (
    LITERATURE_QUESTIONS, YIELD_QUESTIONS, MULTI_TOOL_QUESTIONS, OUT_OF_SCOPE_QUESTIONS,
)
from src.eval.golden_dataset_builder import (
    get_real_literature_context, get_real_yield_prediction, draft_answer_with_llm,
)

GOLDEN_PATH = Path("data/golden_qa_pairs.json")


def safe_draft(question: str, context: str, api_key: str, model: str) -> str:
    """
    Wraps draft_answer_with_llm(), catching the RefusaL-related RuntimeError
    it now raises (see that function's docstring) and surfacing it clearly
    to the reviewer instead of letting the script crash or silently
    returning truncated text. Returns a clearly-marked placeholder the
    reviewer will immediately recognize as needing a manual answer via
    'edit', rather than something that could be mistaken for a real draft.
    """
    try:
        return draft_answer_with_llm(question, context, api_key, model)
    except RuntimeError as e:
        return f"[DRAFTING FAILED — {e}]\n\nUse 'edit' to write this answer manually."


def load_existing() -> list[dict]:
    if GOLDEN_PATH.exists():
        with open(GOLDEN_PATH) as f:
            return json.load(f)
    return []


def save(pairs: list[dict]):
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "w") as f:
        json.dump(pairs, f, indent=2)


def review_and_save(question: str, category: str, draft: str, real_context: str,
                    existing: list[dict], reference_pmids: list = None):
    print(f"\n{'='*70}")
    print(f"CATEGORY: {category}")
    print(f"QUESTION: {question}")
    print(f"{'='*70}")
    print(f"\n--- Real data used for drafting ---\n{real_context}")
    print(f"\n--- Drafted answer ---\n{draft}")
    print(f"\n{'-'*70}")
    print("Press Enter to ACCEPT as-is, type 'edit' to revise, or 'skip' to skip:")
    action = input("> ").strip().lower()

    if action == "skip":
        print("Skipped.")
        return

    final_answer = draft
    if action == "edit":
        print("Type your revised answer (Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "" and (not lines or lines[-1] == ""):
                break
            lines.append(line)
        final_answer = " ".join(l for l in lines if l).strip() or draft

    existing.append({
        "question": question,
        "category": category,
        "ground_truth": final_answer,
        "reference_pmids": reference_pmids or [],
    })
    save(existing)
    print(f"Saved. ({len(existing)} total)")


def build_literature(cfg, project2_path, api_key, model, existing, done_questions):
    for question in LITERATURE_QUESTIONS:
        if question in done_questions:
            continue
        context, chunks = get_real_literature_context(question, project2_path)
        draft = safe_draft(question, context, api_key, model)
        pmids = [c["pmid"] for c in chunks]
        review_and_save(question, "literature_only", draft, context, existing, pmids)


def build_yield(cfg, yield_api_url, api_key, model, existing, done_questions):
    for question, state, year in YIELD_QUESTIONS:
        if question in done_questions:
            continue
        context = get_real_yield_prediction(state, year, yield_api_url)
        draft = safe_draft(question, context, api_key, model)
        review_and_save(question, "yield_only", draft, context, existing)


def build_multi_tool(cfg, project2_path, yield_api_url, api_key, model, existing, done_questions):
    """
    FIXED from an earlier version that only called literature retrieval
    using the full compound question as the search query — this pulled in
    "yield prediction methodology" papers (ML models, SHAP values) instead
    of actual agronomic practice papers, since "yield"/"predict" keywords
    dominated the semantic match, AND never called Project 1's real yield
    API at all for this category. A real test showed this produce a
    drafted answer honestly (correctly) reporting no yield data was
    available — technically accurate given what was fetched, but
    misleading about what the golden answer SHOULD contain, since real
    yield data was available all along, just never requested.

    Now calls BOTH real data sources — Project 1's yield API (using the
    question's explicit state/year) and Project 2's literature search
    (using a separate, narrowly-targeted query, not the full compound
    question) — and combines both into the context shown to the drafting
    LLM, matching what a correctly-functioning multi-tool agent should
    actually be able to produce.
    """
    for question, state, year, literature_query in MULTI_TOOL_QUESTIONS:
        if question in done_questions:
            continue

        yield_result = get_real_yield_prediction(state, year, yield_api_url)
        lit_context, chunks = get_real_literature_context(literature_query, project2_path)

        combined_context = (
            f"--- Yield prediction (for {state}, {year}) ---\n{yield_result}\n\n"
            f"--- Literature search results (query: \"{literature_query}\") ---\n{lit_context}"
        )

        draft = safe_draft(question, combined_context, api_key, model)
        pmids = [c["pmid"] for c in chunks]
        review_and_save(question, "multi_tool", draft, combined_context, existing, pmids)


def build_out_of_scope(existing, done_questions):
    for question in OUT_OF_SCOPE_QUESTIONS:
        if question in done_questions:
            continue
        expected = (
            "The agent should decline to answer, note that this is outside its "
            "scope (corn yield / agronomic research), and should NOT attempt a "
            "guess or call either tool."
        )
        review_and_save(question, "out_of_scope", expected,
                        "N/A — no real data needed for this category", existing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=[
        "literature_only", "yield_only", "multi_tool", "out_of_scope"
    ], default=None)
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    project2_path = Path(cfg["target_project"]["project2_path"]).resolve()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — needed to draft candidate answers.")
    model = cfg["golden_dataset"]["draft_llm_model"]

    yield_api_url = "http://54.214.151.133:8000/predict"

    existing = load_existing()
    done_questions = {p["question"] for p in existing}
    print(f"{len(existing)} question(s) already in the golden dataset.\n")

    categories_to_run = [args.category] if args.category else [
        "literature_only", "yield_only", "multi_tool", "out_of_scope"
    ]

    if "literature_only" in categories_to_run:
        build_literature(cfg, project2_path, api_key, model, existing, done_questions)
    if "yield_only" in categories_to_run:
        build_yield(cfg, yield_api_url, api_key, model, existing, done_questions)
    if "multi_tool" in categories_to_run:
        build_multi_tool(cfg, project2_path, yield_api_url, api_key, model, existing, done_questions)
    if "out_of_scope" in categories_to_run:
        build_out_of_scope(existing, done_questions)

    print(f"\n{'='*70}\nDONE. {len(existing)} question(s) in {GOLDEN_PATH}\n{'='*70}")


if __name__ == "__main__":
    main()
