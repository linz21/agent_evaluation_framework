"""
Test suite for the agent evaluation framework.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestExtractToolsCalled:
    """
    Regression tests for extract_tools_called(), specifically guarding
    against a real bug found via testing: Project 3's own prompt template
    includes an instructional line describing the expected format
    ("Action: the tool name, one of [predict_corn_yield,
    search_literature]"), which is part of every transcript regardless
    of what the agent actually did. A naive regex matched this
    instructional text too, capturing "the" as a phantom tool call on
    every single benchmark result.
    """

    def test_instructional_text_alone_produces_no_matches(self):
        from src.eval.metrics import extract_tools_called
        instructional = (
            "Use the following format exactly:\n"
            "Question: the input question you must answer\n"
            "Thought: think about whether you need a tool, and if so, which one\n"
            "Action: the tool name, one of [predict_corn_yield, search_literature]\n"
            "Action Input: a JSON object with the tool's arguments"
        )
        assert extract_tools_called(instructional) == []

    def test_real_tool_call_extracted_correctly_alongside_instructional_text(self):
        from src.eval.metrics import extract_tools_called
        transcript = (
            "Action: the tool name, one of [predict_corn_yield, search_literature]\n"
            "Question: What corn yield should I expect in Illinois?\n"
            "Action: predict_corn_yield\n"
            "Action Input: {}"
        )
        assert extract_tools_called(transcript) == ["predict_corn_yield"]

    def test_multi_tool_question_extracts_both_in_order(self):
        from src.eval.metrics import extract_tools_called
        transcript = (
            "Action: predict_corn_yield\n"
            "Action Input: {}\n"
            "Observation: 169.82 bu/acre\n"
            "Action: search_literature\n"
            "Action Input: {}"
        )
        assert extract_tools_called(transcript) == ["predict_corn_yield", "search_literature"]

    def test_out_of_scope_question_with_no_real_tool_calls_is_empty(self):
        from src.eval.metrics import extract_tools_called
        transcript = (
            "Action: the tool name, one of [predict_corn_yield, search_literature]\n"
            "Thought: This question is outside my scope.\n"
            "Final Answer: I can't help with that."
        )
        assert extract_tools_called(transcript) == []


class TestExtractMemoryContext:
    def test_long_term_memory_extracted(self):
        from src.eval.metrics import extract_memory_context
        transcript = (
            "IMPORTANT: do NOT include a numbered reference list...\n\n"
            "Potentially relevant past interactions (may be from a different session):\n"
            "- Q: What corn yield should I expect in Nebraska in 2024?\n"
            "  A: The predicted corn yield for Nebraska in 2024 is 169.56 bushels per acre.\n\n"
            "Begin!\nQuestion: ...\nThought:"
        )
        result = extract_memory_context(transcript)
        assert "169.56 bushels per acre" in result

    def test_short_term_memory_extracted(self):
        from src.eval.metrics import extract_memory_context
        transcript = (
            "Recent conversation in this session:\n"
            "user: What corn yield should I expect in Illinois?\n"
            "assistant: 169.82 bu/acre\n\n"
            "Begin!\nQuestion: ...\nThought:"
        )
        result = extract_memory_context(transcript)
        assert "169.82 bu/acre" in result

    def test_no_memory_context_returns_empty(self):
        from src.eval.metrics import extract_memory_context
        transcript = "Some transcript with no memory section.\n\nBegin!\nQuestion: ...\nThought:"
        assert extract_memory_context(transcript) == ""


class TestExtractObservations:
    def test_single_observation_extracted(self):
        from src.eval.metrics import extract_observations
        transcript = (
            "Action: search_literature\n"
            "Action Input: {}\n"
            "Observation: Real retrieved content here.\n"
            "Thought: I now know the final answer."
        )
        result = extract_observations(transcript)
        assert result == "Real retrieved content here."

    def test_multiple_observations_joined(self):
        from src.eval.metrics import extract_observations
        transcript = (
            "Observation: First tool result.\n"
            "Thought: Need another tool.\n"
            "Observation: Second tool result.\n"
            "Thought: Done."
        )
        result = extract_observations(transcript)
        assert "First tool result." in result
        assert "Second tool result." in result

    def test_no_observations_returns_empty_string(self):
        from src.eval.metrics import extract_observations
        transcript = "Thought: This is out of scope.\nFinal Answer: I can't help with that."
        assert extract_observations(transcript) == ""


class TestToolSelectionCorrectness:
    def test_out_of_scope_correct_when_no_tools_called(self):
        from src.eval.metrics import compute_tool_selection_correctness
        result = compute_tool_selection_correctness("out_of_scope", [])
        assert result["correct"] is True

    def test_yield_only_correct_when_yield_tool_called(self):
        from src.eval.metrics import compute_tool_selection_correctness
        result = compute_tool_selection_correctness("yield_only", ["predict_corn_yield"])
        assert result["correct"] is True

    def test_multi_tool_requires_both_tools(self):
        from src.eval.metrics import compute_tool_selection_correctness
        result = compute_tool_selection_correctness("multi_tool", ["predict_corn_yield"])
        assert result["correct"] is False
