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


def extract_tools_called(transcript: str) -> list[str]:
    """
    Parses the sequence of tool names actually called during an agent
    run, from the raw transcript text Project 3's ReactAgent.run()
    returns (which includes literal "Action: tool_name" lines).
    """
    return re.findall(r"Action:\s*(\w+)", transcript)


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


def judge_hallucination(question: str, ground_truth: str, agent_answer: str,
                        api_key: str, judge_model: str = "claude-opus-4-1") -> dict:
    """
    Detects whether the agent's answer contains specific factual claims
    NOT supported by the ground truth — a DIFFERENT check from accuracy.
    An answer can be accurate (correctly conservative) without
    hallucinating; conversely an answer could technically be "on-topic"
    while still inventing unsupported specifics (fake statistics, fake
    citations, fake claims of certainty). This specifically targets the
    fabrication failure mode found repeatedly during Project 3's
    development (fabricated claims about prior conversation turns,
    fabricated answers on tool failure).
    """
    prompt = f"""You are checking an AI agent's answer for HALLUCINATION —
specific factual claims, statistics, or citations that are NOT supported
by the ground truth answer, even if the overall answer seems plausible.

Question: {question}

Ground truth answer (the only source of verified facts): {ground_truth}

Agent's answer: {agent_answer}

Does the agent's answer contain any specific claim, number, citation, or
fact that is NOT supported by the ground truth? Being appropriately vague
or declining to answer is NOT hallucination. Adding a specific-sounding
detail, statistic, or citation not present in the ground truth IS
hallucination, even if it sounds plausible.

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
