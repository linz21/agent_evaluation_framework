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


def extract_memory_context(transcript: str) -> str:
    """
    Extracts the memory context block from the transcript — either
    short-term "Recent conversation in this session:" (Redis) or
    long-term "Potentially relevant past interactions:" (vector store),
    or both. This is real content the model legitimately had access to
    when generating its answer, just like tool Observations — found via
    a real test that answers correctly recalling a TRUE prior result
    from memory (e.g. a real yield number from an earlier session,
    without re-calling predict_corn_yield) were being flagged as
    hallucination, since extract_observations() alone only captures
    THIS run's tool outputs, missing memory content the model also saw.
    """
    match = re.search(
        r"((?:Recent conversation in this session:|Potentially relevant past interactions).*?)\n\nBegin!",
        transcript, re.DOTALL
    )
    return match.group(1).strip() if match else ""


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
        # NOTE: temperature=0 removed — a real error surfaced when
        # switching to claude-opus-4-8 (after claude-opus-4-1 was
        # deprecated): "temperature is deprecated for this model." Newer
        # models don't need explicit temperature=0 for near-deterministic
        # output the way older ones did.
        model=judge_model, max_tokens=500,
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
                        tool_context: str = None, memory_context: str = None) -> dict:
    """
    Detects whether the agent's answer is FAITHFUL to what it actually
    had access to — a DIFFERENT check from accuracy (judge_task_accuracy),
    which compares against ground truth instead.

    DESIGN ALIGNED WITH RAGAS: confirmed via RAGAS/DeepEval documentation
    that hallucination/faithfulness should be checked against the
    RETRIEVED CONTEXT the model actually had access to, NOT ground truth.

    ITERATION HISTORY:
    Round 1: citations incorrectly flagged as hallucination — fixed,
    excluded from consideration.
    Round 2: judged against ground_truth — caused false positives on
    true claims simply omitted from ground_truth's brief summary — fixed,
    switched to checking against retrieved tool context only.
    Round 3 (THIS version): a real test showed the model correctly,
    honestly recalling a TRUE prior result from memory (e.g. a real
    yield number from an earlier session) still got flagged as
    hallucination — tool_context and memory_context were being combined
    into ONE string before being passed in, but the prompt still labeled
    the whole blob "retrieved source content," a framing that specifically
    implies tool retrieval. The judge's own reasoning confirmed it wasn't
    recognizing the memory portion as legitimate grounding even though it
    was technically present in the string. Fixed by taking both as
    SEPARATE parameters and giving each its own clearly labeled section
    in the prompt, explicitly stating a claim grounded in EITHER is
    faithful — not one mislabeled combined blob.

    Handles the no-context case (neither tool nor memory content) the
    same as before: an answer with no substantive claims is trivially
    faithful; substantive claims with zero grounding are flagged as the
    real fabrication-on-failure pattern this check exists to catch.
    """
    has_context = (tool_context and tool_context.strip()) or (memory_context and memory_context.strip())

    if not has_context:
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
        sections = []
        if tool_context and tool_context.strip():
            sections.append(f"TOOL RESULTS FROM THIS RUN (real, current data):\n{tool_context}")
        if memory_context and memory_context.strip():
            sections.append(
                f"MEMORY OF PAST INTERACTIONS (also real, genuine grounding — a claim "
                f"the agent correctly recalls from a prior real exchange, even from a "
                f"different session, is faithful, NOT hallucination):\n{memory_context}"
            )
        context_block = "\n\n".join(sections)

        prompt = f"""You are checking an AI agent's answer for FAITHFULNESS
to what it actually had access to — does the answer stick to real, given
information, or does it add specific claims not supported by ANY of it?

Question: {question}

{context_block}

Agent's answer: {agent_answer}

IMPORTANT: IGNORE citations, source titles, publication years, and
meta-statements like "based on the search_literature tool" or "sources
consulted" — these are handled and verified by a SEPARATE, already-tested
system, not part of this check.

Does the agent's answer contain any SUBSTANTIVE factual or scientific
claim — a mechanism, a statistic, a specific finding — that is NOT
supported by EITHER section above (tool results or memory)? Being
appropriately vague, or declining to answer, is NOT a problem. A claim
correctly recalled from memory of a genuine past interaction is NOT
hallucination, even if it's not in this run's tool results. Inventing a
specific-sounding detail not present in EITHER section IS a problem,
even if it sounds plausible or is independently true.

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
