"""
Synthesizer node.

Reads `retrieved_data` from state, asks the LLM to synthesize an
answer, and writes the result into `synthesized_answer`.
"""

from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from prompts.synthesizer_prompt import SYNTHESIZER_SYSTEM_PROMPT
from utils.llm import get_llm


def synthesizer(state: AgentState) -> AgentState:
    llm = get_llm()
    retrieved_data = state["retrieved_data"]

    response = llm.invoke(
        [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=retrieved_data),
        ]
    )

    # Return only the field this node owns; see planner.py for why.
    return {"synthesized_answer": response.content}
