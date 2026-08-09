"""
Shared state for the multi-agent LangGraph workflow.

Every node in the graph receives this state, reads the fields it needs,
and returns an updated copy (or the same dict with new keys set).
LangGraph merges the returned dict back into the running state.

Design notes:
- We use `total=False` so fields can be genuinely optional in early
  execution (e.g. `retrieved_data` doesn't exist until data_fetcher runs).
- `loop_score` is included now so the state shape is stable before the
  loop detector is implemented. It defaults conceptually to None/0.0
  until Aayush's cosine-similarity function populates it.
- Keep this file the single source of truth for what flows through the
  graph. Add new fields here first, then wire them into the relevant node.
"""

from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    # Set once, at the start of the run.
    user_query: str

    # Written by planner()
    plan: str

    # Written by data_fetcher()
    retrieved_data: str

    # Written by synthesizer()
    synthesized_answer: str

    # Written by formatter() — the final output of the graph.
    formatted_answer: str

    # Reserved for Aayush's cosine-similarity loop detector.
    # Populated once utils/loop_detector.py is implemented and wired
    # into a node (or a conditional edge) that compares successive
    # outputs.
    loop_score: Optional[float]
