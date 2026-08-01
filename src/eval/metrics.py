"""
Metrics for benchmarking agent versions against the golden dataset:

1. Task accuracy (LLM-as-judge) — does the agent's answer correctly
   address the question, compared to the human-reviewed ground truth?
2. Tool-use efficiency — did the agent call the RIGHT tool(s) for this
   question's category, and how many tool calls did it take?
3. Hallucination rate (LLM-as-judge) — does the answer contain specific
   claims NOT supported by the ground truth, even if the overall answer
   seems plausible? A DIFFERENT question from accuracy: an answer can be
   incomplete/vague without fabricating anything, or can be superficially
   plausible while adding invented specifics.
4. Latency — real wall-clock time per agent.run() call.

The judge model is intentionally a DIFFERENT model than any agent version
being evaluated (see configs/config.yaml's comment on this) — Opus is
used here, since claude-sonnet-4.5 is one of the evaluated versions.

Judge prompts request a structured JSON verdict for reliable parsing —
tested against clear synthetic cases (obviously correct, obviously wrong,
obviously fabricated) before being trusted, same "validate before use"
principle as src/stats/.
"""

import json
import logging
import re
import time

import anthropic

log = logging.getLogger(__name__)

EXPECTED_TOOLS_BY_CATEGORY = {
    "literature_only": {"search_literature"},
    "yield_only": {"predict_corn_yield"},
    "multi_tool": {"predict_corn_yield", "search_literature"},
    "out_of_scope": set(),
}


def extract_observations(transcript: str) -> str:
    """
    Extracts all "Observation:" blocks from a transcript — the ACTUAL
    tool results returned during this specific run (real retrieved
    literature passages, real yield API responses). Used to give the
    hallucination judge genuine grounding beyond just the golden
    dataset's ground_truth field, which is a necessarily brief,
    deliberately conservative summary that can legitimately omit real
    details present in the full retrieved sources — see
    judge_hallucination's docstring for why this matters.
    """
    observations = re.findall(r"Observation:\s*(.*?)(?=\nThought:|\Z)", transcript, re.DOTALL)
    return "\n\n".join(obs.strip() for obs in observations)


def extract_tools_called(transcript: str) -> list[str]:
    """
    Parses the sequence of tool names actually called during an agent
    run, from the raw transcript text Project 3's ReactAgent.run()
    returns (which includes literal "Action: tool_name" lines).

    IMPORTANT: matches only KNOWN real tool names explicitly, rather than
    any word after "Action:". Found via a real test: Project 3's own
    prompt template includes an instructional line describing the
    expected format — "Action: the tool name, one of [predict_corn_yield,
    search_literature]" — which is part of the full transcript on EVERY
    run (the transcript field includes the whole prompt, not just the
    model's generated content). A naive r"Action:\\s*(\\w+)" regex matched
    this instructional text too, capturing the literal word "the" as a
    phantom tool call on every single result, which broke
    tool_selection_correctness for every question, not just this one.
    """
    return re.findall(r"Action:\s*(predict_corn_yield|search_literature)", transcript)


def compute_tool_selection_correctness(category: str, tools_called: list[str]) -> dict:
    """
    Compares the ACTUAL set of tools called against the EXPECTED set for
    this question's category. Uses set comparison (not sequence/order),
    since order doesn't matter for correctness here, only which tools
    were used at all — e.g. calling search_literature twice for a
    literature_only question is still "correct" tool selection, just
    less efficient (see compute_tool_efficiency for that separately).
    """
    expected = EXPECTED_TOOLS_BY_CATEGORY.get(category, set())
    actual = set(tools_called)
    return {
        "expected_tools": sorted(expected),
        "actual_tools": sorted(actual),
        "correct": actual == expected,
    }


def compute_tool_efficiency(tools_called: list[str]) -> int:
    """
    Total number of tool calls made during one run — a simple proxy for
    efficiency. Fewer calls for a straightforward question is generally
    better, though this is a rough proxy, not a full cost/quality
    tradeoff measure — reported alongside tool_selection_correctness so
    a low call count on a WRONG tool selection isn't mistaken for good
    performance.
    """
    return len(tools_called)


def _call_judge(prompt: str, api_key: str, judge_model: str) -> dict:
    """
    Shared helper for both judge-based metrics — calls the judge model
    and parses a JSON verdict from its response. Raises clearly if
    parsing fails, rather than silently returning a default/guessed
    result that could corrupt downstream statistics.
    """
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=judge_model, max_tokens=500, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text if response.content else ""

    # Judge is instructed to respond with ONLY a JSON object, but strip
    # markdown code fences defensively in case the model wraps it anyway
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Judge response was not valid JSON: {text!r}") from e


def judge_task_accuracy(question: str, ground_truth: str, agent_answer: str,
                        api_key: str, judge_model: str = "claude-opus-4-1") -> dict:
    """
    Binary correct/incorrect verdict — does the agent's answer correctly
    address the question, compared to the human-reviewed ground truth?
    An answer can be "correct" even if worded very differently from the
    ground truth, as long as the substantive content matches; it can also
    be "correct" while being MORE conservative than ground truth (e.g.
    correctly saying "the data doesn't fully cover this" when that's
    accurate), since faithfulness to what's actually known matters more
    than matching a specific level of confidence.
    """
    prompt = f"""You are evaluating whether an AI agent's answer correctly
addresses a question, compared to a human-verified ground truth answer.

Question: {question}

Ground truth answer: {ground_truth}

Agent's answer: {agent_answer}

Judge whether the agent's answer is substantively CORRECT — it does not
need to match the ground truth's exact wording, but its factual claims
and conclusions should be consistent with the ground truth. An answer
that is appropriately MORE cautious than the ground truth (e.g. correctly
declining to guess where data is insufficient) should be judged correct.
An answer that makes claims contradicting the ground truth, or fabricates
specifics not supported by it, should be judged incorrect.

Respond with ONLY a JSON object, no other text:
{{"correct": true or false, "reasoning": "one sentence explaining why"}}"""

    result = _call_judge(prompt, api_key, judge_model)
    return {
        "correct": bool(result.get("correct")),
        "reasoning": result.get("reasoning", ""),
    }


def judge_hallucination(question: str, agent_answer: str,
                        api_key: str, judge_model: str = "claude-opus-4-1",
                        retrieved_context: str = None) -> dict:
    """
    Detects whether the agent's answer is FAITHFUL to what it actually
    retrieved — a DIFFERENT check from accuracy (judge_task_accuracy),
    which compares against ground truth instead.

    DESIGN ALIGNED WITH RAGAS: confirmed via RAGAS/DeepEval documentation
    that hallucination/faithfulness should be checked against the
    RETRIEVED CONTEXT the model actually had access to, NOT ground truth
    — these are two established, separate metrics (RAGAS: "Faithfulness
    measures... against the retrieved context" vs. "Answer similarity...
    against the ground truth answer"; DeepEval: "Use faithfulness for
    RAG... don't use [ground-truth-based checking] on a live RAG
    system"). The model never saw ground_truth when generating its
    answer, so judging faithfulness against it is the wrong reference —
    ground_truth is for accuracy/correctness, retrieved_context is for
    faithfulness/hallucination.

    ITERATION HISTORY (kept for context, since each was a real, tested
    fix):
    Round 1: original prompt told the judge citations not in ground_truth
    ARE hallucination — wrong, since real citations are code-guaranteed
    and validated. Fixed by excluding citations from consideration.
    Round 2: judged against ground_truth (with retrieved_context as
    secondary support) — caused false positives on true claims omitted
    from ground_truth's necessarily brief summary (a real UAV-imaging
    test case). Round 3 (this version): ground_truth removed from the
    faithfulness check entirely, checking ONLY against retrieved_context,
    matching established RAG evaluation practice.

    Handles the no-context case (e.g. out-of-scope questions where no
    tool was called) explicitly: if there's no retrieved_context, an
    answer with no substantive claims (a decline) is trivially faithful;
    but an answer that makes substantive claims WITHOUT any retrieved
    grounding is flagged as a real, different problem — confidently
    answering from ungrounded parametric knowledge, exactly the
    fabrication pattern found repeatedly during Project 3's development.
    """
    if not retrieved_context or not retrieved_context.strip():
        prompt = f"""You are checking whether an AI agent's answer makes
any SUBSTANTIVE factual or scientific claim WITHOUT having any retrieved
source material to ground it in (no tool was called, or no content was
returned).

Question: {question}

Agent's answer: {agent_answer}

Does the agent's answer contain any substantive factual or scientific
claim (a mechanism, statistic, or specific finding) presented as fact,
despite having NO retrieved source material to support it? Declining to
answer, or appropriately saying information isn't available, is NOT a
problem. Confidently stating specific facts with no grounding IS a
problem — this is the exact fabrication-on-failure pattern this check is
designed to catch.

Respond with ONLY a JSON object, no other text:
{{"hallucinated": true or false, "reasoning": "one sentence explaining why",
  "unsupported_claims": ["list", "of", "specific", "ungrounded", "claims", "if", "any"]}}"""
    else:
        prompt = f"""You are checking an AI agent's answer for FAITHFULNESS
to the actual retrieved source content — does the answer stick to what
was genuinely retrieved, or does it add specific claims not supported by
that source material?

Question: {question}

Actual retrieved source content (the ONLY source of truth for this
check): {retrieved_context}

Agent's answer: {agent_answer}

IMPORTANT: IGNORE citations, source titles, publication years, and
meta-statements like "based on the search_literature tool" or "sources
consulted" — these are handled and verified by a SEPARATE, already-tested
system, not part of this check.

Does the agent's answer contain any SUBSTANTIVE factual or scientific
claim — a mechanism, a statistic, a specific finding — that is NOT
supported by the retrieved source content above? Being appropriately
vague, or declining to answer, is NOT a problem. Inventing a
specific-sounding detail not present in the retrieved content IS a
problem, even if it sounds plausible or is independently true — the
question is whether the answer is faithful to what was actually
retrieved, not whether it's true in some absolute sense.

Respond with ONLY a JSON object, no other text:
{{"hallucinated": true or false, "reasoning": "one sentence explaining why",
  "unsupported_claims": ["list", "of", "specific", "unsupported", "claims", "if", "any"]}}"""

    result = _call_judge(prompt, api_key, judge_model)
    return {
        "hallucinated": bool(result.get("hallucinated")),
        "reasoning": result.get("reasoning", ""),
        "unsupported_claims": result.get("unsupported_claims", []),
    }


def time_agent_run(agent, question: str) -> dict:
    """
    Wraps a single agent.run() call with wall-clock timing. Returns the
    normal run() result dict plus a 'latency_seconds' key.
    """
    start = time.time()
    result = agent.run(question, verbose=False)
    elapsed = time.time() - start
    result["latency_seconds"] = elapsed
    return result
