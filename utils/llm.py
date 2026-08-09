import os
from functools import lru_cache


@lru_cache(maxsize=None)
def get_llm():
    """
    Returns a configured LangChain chat LLM instance that nodes call
    `.invoke()` on. The provider is selected via the LLM_PROVIDER env var.

    Supported providers (with required env vars):
    - groq       (default) → GROQ_API_KEY, LLM_MODEL default llama-3.3-70b-versatile
    - google/gemini        → GOOGLE_API_KEY, LLM_MODEL default gemini-2.0-flash
    - openai               → OPENAI_API_KEY, LLM_MODEL default gpt-4o-mini

    Cached with lru_cache so repeated calls within a process reuse the
    same client instead of re-initializing per node call.
    """
    provider = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()

    if provider in {"groq",}:
        # pyrefly: ignore [missing-import]
        from langchain_groq import ChatGroq

        model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(model=model, temperature=0)

    if provider in {"google", "gemini", "googlegenerativeai"}:
        # pyrefly: ignore [missing-import]
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=0)

    if provider in {"openai",}:
        # pyrefly: ignore [missing-import]
        from langchain_openai import ChatOpenAI

        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)

    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}'. "
        "Supported providers: groq, google/gemini, openai."
    )
