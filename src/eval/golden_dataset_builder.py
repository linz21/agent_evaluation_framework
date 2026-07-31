"""
Semi-automated golden dataset drafting — for each candidate question,
pulls REAL data (retrieved passages from Project 2's corpus, or a real
prediction from Project 1's API), has an LLM draft a candidate answer
grounded in that real data, and returns the draft for human review.

NOTHING is auto-accepted. Every draft must be reviewed/edited by a human
before being added to the golden dataset — the LLM's job here is only to
save typing, not to make the final call on correctness (same principle
established in Project 2: an ungrounded external LLM answer was
explicitly rejected as ground truth; this differs by grounding every
draft in real retrieved data AND requiring human review, addressing both
concerns that led to that earlier rejection).

Reuses the cross-project import pattern established in Project 3's
literature_search_tool.py (namespace collision + working-directory
coupling both handled the same way) to call Project 2's real retriever,
since this project needs literature-grounded drafts for a much larger
question set than Project 3's own use of that same corpus.
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

import anthropic
import requests
import yaml

log = logging.getLogger(__name__)

_project2_retriever = None
_project2_generate_answer = None


def _load_project2_modules(project2_path: Path):
    """
    Loads Project 2's retriever + generator modules by explicit file path,
    handling the same two cross-project issues found building Project 3:
    (1) both projects use the top-level package name `src`, requiring a
    temporary sys.modules['src'] swap; (2) Project 2's own config uses
    paths relative to ITS OWN directory, requiring a temporary os.chdir().
    See Project 3's literature_search_tool.py for the original diagnosis
    of both issues.
    """
    global _project2_retriever, _project2_generate_answer
    if _project2_retriever is not None:
        return _project2_retriever, _project2_generate_answer

    this_project_src = sys.modules.get("src")
    project2_src_init = project2_path / "src" / "__init__.py"
    project2_src_spec = importlib.util.spec_from_file_location(
        "src", project2_src_init, submodule_search_locations=[str(project2_path / "src")]
    )
    project2_src_module = importlib.util.module_from_spec(project2_src_spec)
    sys.modules["src"] = project2_src_module

    original_cwd = os.getcwd()
    try:
        os.chdir(project2_path)

        spec = importlib.util.spec_from_file_location(
            "project2_retriever", project2_path / "src" / "retrieval" / "retriever.py"
        )
        retriever_module = importlib.util.module_from_spec(spec)
        sys.modules["project2_retriever"] = retriever_module
        spec.loader.exec_module(retriever_module)

        gen_spec = importlib.util.spec_from_file_location(
            "project2_generator", project2_path / "src" / "generation" / "generator.py"
        )
        generator_module = importlib.util.module_from_spec(gen_spec)
        sys.modules["project2_generator"] = generator_module
        gen_spec.loader.exec_module(generator_module)

        with open(project2_path / "configs" / "config.yaml") as f:
            project2_cfg = yaml.safe_load(f)

        _project2_retriever = retriever_module.HybridRetriever(project2_cfg)
        _project2_retriever._get_embedder()
        _project2_generate_answer = generator_module.generate_answer

    finally:
        os.chdir(original_cwd)
        if this_project_src is not None:
            sys.modules["src"] = this_project_src
        else:
            sys.modules.pop("src", None)

    return _project2_retriever, _project2_generate_answer


def get_real_literature_context(question: str, project2_path: Path, top_k: int = 5) -> tuple[str, list]:
    """
    Retrieves real passages from Project 2's corpus for a question.
    Returns (formatted_context_text, list_of_chunk_dicts) — the context
    text is what gets shown to both the drafting LLM and the human
    reviewer, so the reviewer can verify the draft against the exact same
    real source material the LLM saw.
    """
    retriever, _ = _load_project2_modules(project2_path)
    chunks = retriever.search(question, top_k=top_k)

    if not chunks:
        return "No relevant passages found in the corpus.", []

    context_lines = []
    for i, c in enumerate(chunks, 1):
        context_lines.append(f"[{i}] {c['title']} ({c.get('year', 'n.d.')}) — PMID {c['pmid']}")
        # NOTE: previously truncated to [:500] chars, which cut off most
        # PubMed abstracts (typically 1000-2000+ chars) right before their
        # results/conclusion section — a real test showed Claude correctly
        # (but unhelpfully) refusing to draft an answer since it genuinely
        # couldn't see the actual findings, only intro/methods text. Full
        # abstract text is well within any LLM's context window, so no
        # truncation is actually necessary here.
        context_lines.append(f"    {c['text']}")
    return "\n".join(context_lines), chunks


def get_real_yield_prediction(state: str, year: int, yield_api_url: str) -> str:
    """
    Calls Project 1's real live API. Returns a plain description of the
    real result (not yet drafted into a full answer) — used both to draft
    the golden answer AND as ground truth the reviewer can check the
    drafted answer's numbers against directly.
    """
    try:
        response = requests.post(
            yield_api_url,
            json={"year": year, "state": state, "planted_acres": 50000},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return (
            f"Real API result: {data['predicted_yield_bu_per_acre']} bu/acre "
            f"(95% CI: {data['ci_lower']}-{data['ci_upper']} bu/acre)"
        )
    except Exception as e:
        return f"API call failed: {e} — this question cannot be drafted until the API is reachable."


def draft_answer_with_llm(question: str, real_context: str, api_key: str,
                          model: str = "claude-sonnet-4-5") -> str:
    """
    Has an LLM draft a candidate answer grounded ONLY in the real_context
    provided — explicitly instructed not to add outside knowledge, since
    the whole point is a draft the reviewer can verify against real data,
    not a plausible-sounding answer from general training knowledge.
    """
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Based ONLY on the following real data, write a factual, concise
answer (3-5 sentences) to the question below. Do not add any information,
claims, or specifics not present in the provided data. If the data doesn't
fully answer the question, say so explicitly rather than filling gaps
with outside knowledge.

IMPORTANT — if the data includes a confidence interval, do NOT describe it
using probability language like "the actual value has a 95% probability of
falling within this range" or "is statistically likely to fall within this
range" — this is a common but technically incorrect interpretation of a
frequentist confidence interval. Instead, simply state the interval and
describe it as reflecting the level of uncertainty in the estimate (e.g.
"reflecting the level of uncertainty in this estimate"), without making a
probability claim about where the true value specifically falls.

IMPORTANT — if the data includes BOTH a specific numeric prediction (e.g.
a yield forecast for one state/year) AND general research findings (e.g.
agronomic practices studied elsewhere), synthesize them together normally.
Real agronomic research is virtually never conducted in the exact
state/year of a specific forecast — do NOT treat generally-applicable
findings as insufficient or irrelevant just because they weren't
conducted in that exact location. It is appropriate and expected to
present the specific forecast alongside genuinely relevant general
findings, clearly distinguishing the two, rather than declining to
synthesize because the findings aren't location-matched.

Question: {question}

Real data:
{real_context}

Answer:"""

    response = client.messages.create(
        model=model, max_tokens=1200, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    # CRITICAL: check stop_reason before trusting the returned text as a
    # complete draft. A real, repeated bug: this exact question ("What
    # precision agriculture technologies improve crop management?") was
    # accidentally accepted into the golden dataset TWICE with truncated
    # text, because the function silently returned partial text with no
    # indication anything was wrong. Root cause, confirmed via direct
    # diagnostic: stop_reason="refusal" (NOT a max_tokens issue — output
    # was only ~150 tokens, far under any of the limits tried). Likely a
    # false-positive safety classifier trigger on benign agricultural
    # vocabulary ("nanoparticle delivery", "molecular delivery",
    # "engineered release") that superficially overlaps with language
    # that could describe unrelated, more sensitive topics in a different
    # context. Rather than silently return the truncated refusal text,
    # this now raises so the caller can handle it explicitly (skip,
    # rephrase, or flag for manual answer-writing) instead of it being
    # mistaken for a normal, complete draft.
    if response.stop_reason == "refusal":
        raise RuntimeError(
            f"Claude refused to draft an answer for this question "
            f"(stop_reason='refusal', likely a false-positive safety "
            f"trigger on specific vocabulary in the source material). "
            f"Partial text was: {response.content[0].text if response.content else '(none)'}"
        )

    return response.content[0].text if response.content else ""
