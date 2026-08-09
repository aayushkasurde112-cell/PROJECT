"""
Formatter node.

Reads `synthesized_answer` from state, asks the LLM to format it into
the final deliverable, and writes the result into `formatted_answer`.
This is the last node before END.
"""

from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from prompts.formatter_prompt import FORMATTER_SYSTEM_PROMPT
from utils.llm import get_llm


def formatter(state: AgentState) -> AgentState:
    llm = get_llm()
    synthesized_answer = state["synthesized_answer"]

    response = llm.invoke(
        [
            SystemMessage(content=FORMATTER_SYSTEM_PROMPT),
            HumanMessage(content=synthesized_answer),
        ]
    )

    # Return only the field this node owns; see planner.py for why.
    return {"formatted_answer": response.content}
