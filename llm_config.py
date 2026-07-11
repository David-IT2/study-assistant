"""
Central place that decides which LLM and embedding model to use, based on
the LLM_PROVIDER setting in your .env file.

LLM_PROVIDER=openai  -> uses OpenAI's API (costs money, needs OPENAI_API_KEY)
LLM_PROVIDER=ollama  -> uses a free local model via Ollama (no API key, no cost)

Everything else in the project calls get_llm() / get_embeddings() instead of
importing ChatOpenAI/OllamaLLM directly, so this is the only file you need
to touch to switch providers.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # ensures .env is loaded no matter which module imports this first

PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

# Defaults for local models — override these in .env if you want different ones
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def get_llm(temperature: float = 0):
    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=temperature)

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


def get_embeddings():
    if PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


def using_ollama() -> bool:
    return PROVIDER == "ollama"
