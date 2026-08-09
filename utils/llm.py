import os
from functools import lru_cache
 
 
@lru_cache(maxsize=None)
def get_llm():
    """
    Returns a configured LangChain ChatGoogleGenerativeAI instance that
    nodes call `.invoke()` on.
 
    Cached with lru_cache so repeated calls within a process reuse the
    same client instead of re-initializing per node call.
    """
    # pyrefly: ignore [missing-import]
    from langchain_google_genai import ChatGoogleGenerativeAI
 
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(model=model, temperature=0)