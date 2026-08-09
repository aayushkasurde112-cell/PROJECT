"""
Planner node.

Reads `user_query` from state, asks the LLM to produce a plan, and
writes the result into `plan`.
"""

from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from prompts.planner_prompt import PLANNER_SYSTEM_PROMPT
from utils.llm import get_llm


def planner(state: AgentState) -> AgentState:
    llm = get_llm()
    user_query = state["user_query"]

    response = llm.invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=user_query),
        ]
    )

    # Return only the field this node owns. LangGraph merges this
    # partial dict into the running state, so other fields are left
    # untouched rather than being overwritten wholesale.
    return {"plan": response.content}
