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
