"""
Graph definition for the multi-agent workflow.

Current (implemented) flow:

    START -> planner -> data_fetcher -> synthesizer -> formatter -> END

This is a strictly linear graph. Each node reads one field of state
and writes the next, per state.py.

--------------------------------------------------------------------
FUTURE WORK (not implemented, not wired in — for reference only):

Once utils/loop_detector.py is implemented, the intended shape is:

                        ┌── loop detected → Debugger
                        │
    Current Agent → Loop Detector
                        │
                        └── no loop → Continue

That would use `graph.add_conditional_edges(...)` after whichever node
Aayush decides should trigger a loop check, routing to either a new
"debugger" node or continuing to the next stage based on `loop_score`
in state. We are intentionally NOT adding that node or conditional
edge yet — it depends on logic that hasn't been implemented, and
wiring it in now would mean guessing at routing thresholds that
belong to the loop detector's design.
--------------------------------------------------------------------
"""

from langgraph.graph import StateGraph, START, END

from state import AgentState
from nodes.planner import planner
from nodes.data_fetcher import data_fetcher
from nodes.synthesizer import synthesizer
from nodes.formatter import formatter


def build_graph():
    """Builds and compiles the linear multi-agent StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("data_fetcher", data_fetcher)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("formatter", formatter)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "data_fetcher")
    graph.add_edge("data_fetcher", "synthesizer")
    graph.add_edge("synthesizer", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()
