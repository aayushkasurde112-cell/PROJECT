# ResilioAgent — Multi-Agent LangGraph Project

A stateful, multi-agent workflow built with [LangGraph](https://github.com/langchain-ai/langgraph).
Four agents run in sequence, each reading from and writing to a shared state object,
followed by a standalone cosine-similarity loop-detection utility.

## What LangGraph is doing here

LangGraph models the workflow as a graph of nodes (agents) connected by edges.
State is passed into each node; the node returns the fields it's responsible for,
and LangGraph merges that partial result into the running state before handing it
to the next node. This project's graph is currently linear:

```
START -> planner -> data_fetcher -> synthesizer -> formatter -> END
```

## What the State contains

Defined in `state.py` as a `TypedDict` (`AgentState`):

| Field                | Set by         | Description                              |
|-----------------------|----------------|-------------------------------------------|
| `user_query`          | caller (main.py) | The original input question              |
| `plan`                 | `planner`      | The plan produced for answering the query |
| `retrieved_data`       | `data_fetcher` | Findings gathered based on the plan       |
| `synthesized_answer`   | `synthesizer`  | Draft answer synthesized from findings    |
| `formatted_answer`     | `formatter`    | Final, user-facing formatted output       |
| `loop_score`           | *(caller of `calculate_loop_score`)* | Cosine-similarity score comparing two outputs; see below |

## What each node does

- **`nodes/planner.py`** — takes `user_query`, calls the LLM with `PLANNER_SYSTEM_PROMPT`, returns `{"plan": ...}`.
- **`nodes/data_fetcher.py`** — takes `plan`, calls the LLM with `DATA_FETCHER_SYSTEM_PROMPT`, returns `{"retrieved_data": ...}`.
- **`nodes/synthesizer.py`** — takes `retrieved_data`, calls the LLM with `SYNTHESIZER_SYSTEM_PROMPT`, returns `{"synthesized_answer": ...}`.
- **`nodes/formatter.py`** — takes `synthesized_answer`, calls the LLM with `FORMATTER_SYSTEM_PROMPT`, returns `{"formatted_answer": ...}`.

Every node follows the same pattern: read the one state field it needs → build
`[SystemMessage, HumanMessage]` → `llm.invoke(...)` → return **only** the field it
owns (LangGraph merges this into the running state — nodes never return or
overwrite the whole state dict).

**Note on information flow between nodes:** each node's *human* message is exactly
the previous node's output text — nothing else is carried forward automatically.
Because of this, the system prompts are written so each agent's output is
self-contained (e.g. the Planner restates the goal in its own output, since the
Data Fetcher never sees the original `user_query` directly). See `prompts/` for
the full specification of each agent's expected input/output contract.

## Where the system prompts live

`prompts/` — one file per agent, each exporting a single constant:

- `prompts/planner_prompt.py` → `PLANNER_SYSTEM_PROMPT`
- `prompts/data_fetcher_prompt.py` → `DATA_FETCHER_SYSTEM_PROMPT`
- `prompts/synthesizer_prompt.py` → `SYNTHESIZER_SYSTEM_PROMPT`
- `prompts/formatter_prompt.py` → `FORMATTER_SYSTEM_PROMPT`

Each prompt defines the agent's role, its exact input, its required output format,
constraints (including explicit anti-hallucination rules), how it should behave
when information is missing, and how its output is consumed by the next agent.

## How the nodes are connected

See `graph.py`. The graph is currently a straight line (no branching):

```
START -> planner -> data_fetcher -> synthesizer -> formatter -> END
```

## How `loop_score` is calculated

`utils/loop_detector.py` exposes:

```python
from utils.loop_detector import calculate_loop_score, LOOP_SIMILARITY_THRESHOLD

score = calculate_loop_score(previous_output, current_output)
```

- Both arguments are plain text (e.g. two successive `synthesized_answer` or
  `formatted_answer` values from different iterations of a future
  retry/loop-back path).
- Each text is embedded with a `sentence-transformers` model (default:
  `all-MiniLM-L6-v2`, configurable via the `LOOP_DETECTOR_MODEL` env var).
- The function returns the cosine similarity between the two embeddings as a
  float `loop_score`, generally in `[0.0, 1.0]` (1.0 = essentially identical
  meaning, 0.0 = unrelated).
- Empty strings, whitespace-only strings, or non-string input return `0.0`
  safely (no exception) — two "nothing" outputs are never treated as a loop.
- The embedding model itself is loaded lazily on first real use, so importing
  this module or calling it with empty/invalid input never requires a model
  download or network access.

This module is **standalone**: it does not read or write LangGraph state, know
about a Debugger node, or get called anywhere in `graph.py` yet. It's a pure
function ready to be wired in by whoever owns the conditional edge (see below).

## Why 0.95 is the current loop threshold

`LOOP_SIMILARITY_THRESHOLD = 0.95` is exported from `utils/loop_detector.py` per
the project requirement: two consecutive outputs with cosine similarity above
0.95 are considered near-duplicates, indicating the pipeline is likely stuck
repeating itself rather than making progress. This is intentionally a strict
threshold (close to 1.0) so that legitimately similar-but-different outputs
(e.g. answers that happen to share a lot of phrasing) aren't misclassified as a
loop. The threshold is a module constant specifically so downstream code (the
conditional edge) imports it rather than re-hardcoding `0.95` elsewhere.

## Connecting `loop_score` to a Debugger conditional edge (for Pari/Gangotri)

Not implemented here by design — this is intentionally left for the owning
team members. The intended integration, using what already exists:

```python
from langgraph.graph import StateGraph, START, END
from utils.loop_detector import calculate_loop_score, LOOP_SIMILARITY_THRESHOLD

def check_for_loop(state: AgentState) -> AgentState:
    # Compare whichever two outputs are relevant to the retry path
    # (e.g. current vs. previous synthesized_answer across iterations)
    score = calculate_loop_score(state["previous_output"], state["current_output"])
    return {"loop_score": score}

def route_after_loop_check(state: AgentState) -> str:
    if state["loop_score"] > LOOP_SIMILARITY_THRESHOLD:
        return "debugger"
    return "continue"

graph.add_node("loop_check", check_for_loop)
graph.add_node("debugger", debugger_node)  # not yet implemented
graph.add_conditional_edges(
    "loop_check",
    route_after_loop_check,
    {"debugger": "debugger", "continue": "some_next_node"},
)
```

The exact node(s) that produce `previous_output`/`current_output` for comparison
depend on where in the (eventual) retry loop this check should sit — that's part
of the Debugger integration design, not something this module assumes.

## Installing dependencies

```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then fill in your real OPENAI_API_KEY
```

The first time `calculate_loop_score()` actually runs (not just on empty input),
`sentence-transformers` will download the configured embedding model
(`all-MiniLM-L6-v2` by default) from Hugging Face, which requires network access
to `huggingface.co`. After the first download it's cached locally.

## Configuring `.env`

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-real-key-here
OLLAMA_MODEL=llama3
```

`.env` is listed in `.gitignore` and is never committed. `.env.example` mirrors
this structure with placeholder values only — never a real key. `main.py` calls
`load_dotenv()` before importing `graph` (and therefore before any node can call
`get_llm()`), so environment variables are guaranteed to be loaded first.

Note: `utils/llm.py` already wires OpenAI as the primary provider with an
automatic Ollama fallback on API/network errors — that fallback logic belongs to
Ved's circuit-breaker work and was left untouched here.

## Running the project

```bash
python main.py
```

This runs the compiled graph against a hardcoded test query in `main.py` and
prints the full final state, including every intermediate field, to the console.

## Running tests

```bash
pytest tests/ -v
```

Covers:
- `tests/test_loop_detector.py` — cosine-similarity math (pure, offline), empty/
  invalid input handling (offline), mocked-embedding tests validating "nearly
  identical -> high score" and "clearly different -> low score" without network
  access, and one real-model integration test that's automatically skipped (not
  failed) if the embedding model can't be downloaded.
- `tests/test_nodes.py` — all four nodes, with `get_llm()` mocked so no real API
  call or API key is required. Verifies each node reads the correct input field,
  sends exactly one `SystemMessage` + one `HumanMessage`, and returns only the
  state field it owns.

No test in this suite requires a live OpenAI API key.

## Project structure

```
project/
├── main.py                    # Entry point — builds and runs the graph
├── state.py                   # Shared AgentState TypedDict
├── graph.py                   # StateGraph definition (nodes + edges)
├── nodes/                     # One function per agent
│   ├── planner.py
│   ├── data_fetcher.py
│   ├── synthesizer.py
│   └── formatter.py
├── prompts/                   # System prompts for each agent
│   ├── planner_prompt.py
│   ├── data_fetcher_prompt.py
│   ├── synthesizer_prompt.py
│   └── formatter_prompt.py
├── utils/
│   ├── llm.py                 # Central LLM initialization (OpenAI primary, Ollama fallback)
│   └── loop_detector.py       # Cosine-similarity loop_score calculation (standalone)
├── tests/
│   ├── test_loop_detector.py
│   └── test_nodes.py
├── .env.example
├── .gitignore
└── requirements.txt
```
