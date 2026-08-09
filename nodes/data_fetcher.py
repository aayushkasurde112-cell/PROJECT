"""
Data Fetcher node.

Reads `plan` from state, asks the LLM to determine/simulate what data
is needed, and writes the result into `retrieved_data`.

Note: this skeleton does not wire up any real retrieval tools (no RAG,
no external APIs) per project scope — the LLM call here is a
placeholder for wherever actual data fetching logic eventually lives.
"""

from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from prompts.data_fetcher_prompt import DATA_FETCHER_SYSTEM_PROMPT
from utils.llm import get_llm


def data_fetcher(state: AgentState) -> AgentState:
    llm = get_llm()
    plan = state["plan"]

    response = llm.invoke(
        [
            SystemMessage(content=DATA_FETCHER_SYSTEM_PROMPT),
            HumanMessage(content=plan),
        ]
    )

    # Return only the field this node owns; see planner.py for why.
    return {"retrieved_data": response.content}
