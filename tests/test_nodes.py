"""
Tests for nodes/planner.py, nodes/data_fetcher.py, nodes/synthesizer.py,
and nodes/formatter.py.

Every test mocks get_llm() at the point each node imports it, so these
tests never make a real API call and never require an OpenAI API key.
"""

from unittest.mock import MagicMock, patch

from nodes.data_fetcher import data_fetcher
from nodes.formatter import formatter
from nodes.planner import planner
from nodes.synthesizer import synthesizer


def _mock_llm(response_text: str) -> MagicMock:
    """Builds a fake LLM whose .invoke() returns an object with .content."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=response_text)
    return llm


def test_planner_reads_user_query_and_returns_only_plan():
    fake_llm = _mock_llm("RESTATED GOAL: ...\nINFORMATION NEEDS:\n1. ...")

    with patch("nodes.planner.get_llm", return_value=fake_llm):
        result = planner({"user_query": "What is LangGraph?"})

    # Node must return ONLY the field it owns.
    assert result == {"plan": "RESTATED GOAL: ...\nINFORMATION NEEDS:\n1. ..."}

    # The user query must have been passed through as the human message.
    called_messages = fake_llm.invoke.call_args[0][0]
    human_messages = [m for m in called_messages if m.__class__.__name__ == "HumanMessage"]
    assert human_messages[0].content == "What is LangGraph?"


def test_data_fetcher_reads_plan_and_returns_only_retrieved_data():
    fake_llm = _mock_llm("GOAL CONTEXT: ...\nFINDINGS:\n1. ... -> ...")

    with patch("nodes.data_fetcher.get_llm", return_value=fake_llm):
        result = data_fetcher({"plan": "RESTATED GOAL: ...\nINFORMATION NEEDS:\n1. ..."})

    assert result == {"retrieved_data": "GOAL CONTEXT: ...\nFINDINGS:\n1. ... -> ..."}

    called_messages = fake_llm.invoke.call_args[0][0]
    human_messages = [m for m in called_messages if m.__class__.__name__ == "HumanMessage"]
    assert human_messages[0].content == "RESTATED GOAL: ...\nINFORMATION NEEDS:\n1. ..."


def test_synthesizer_reads_retrieved_data_and_returns_only_synthesized_answer():
    fake_llm = _mock_llm("DRAFT ANSWER:\n...\nKNOWN GAPS: None")

    with patch("nodes.synthesizer.get_llm", return_value=fake_llm):
        result = synthesizer({"retrieved_data": "GOAL CONTEXT: ...\nFINDINGS:\n1. ... -> ..."})

    assert result == {"synthesized_answer": "DRAFT ANSWER:\n...\nKNOWN GAPS: None"}

    called_messages = fake_llm.invoke.call_args[0][0]
    human_messages = [m for m in called_messages if m.__class__.__name__ == "HumanMessage"]
    assert human_messages[0].content == "GOAL CONTEXT: ...\nFINDINGS:\n1. ... -> ..."


def test_formatter_reads_synthesized_answer_and_returns_only_formatted_answer():
    fake_llm = _mock_llm("Here is your final answer.")

    with patch("nodes.formatter.get_llm", return_value=fake_llm):
        result = formatter({"synthesized_answer": "DRAFT ANSWER:\n...\nKNOWN GAPS: None"})

    assert result == {"formatted_answer": "Here is your final answer."}

    called_messages = fake_llm.invoke.call_args[0][0]
    human_messages = [m for m in called_messages if m.__class__.__name__ == "HumanMessage"]
    assert human_messages[0].content == "DRAFT ANSWER:\n...\nKNOWN GAPS: None"


def test_each_node_sends_exactly_one_system_and_one_human_message():
    fake_llm = _mock_llm("output")

    with patch("nodes.planner.get_llm", return_value=fake_llm):
        planner({"user_query": "test"})

    called_messages = fake_llm.invoke.call_args[0][0]
    assert len(called_messages) == 2
    assert called_messages[0].__class__.__name__ == "SystemMessage"
    assert called_messages[1].__class__.__name__ == "HumanMessage"
