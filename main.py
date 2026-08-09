"""
Entry point — runs the compiled graph with a simple test query.

Usage:
    python main.py
"""

from dotenv import load_dotenv

load_dotenv()

from graph import build_graph


def main():
    app = build_graph()

    initial_state = {
        "user_query": "What are the key benefits of using LangGraph for multi-agent systems?",
    }

    result = app.invoke(initial_state)

    print("\n--- Final State ---")
    for key, value in result.items():
        print(f"\n[{key}]")
        print(value)


if __name__ == "__main__":
    main()
