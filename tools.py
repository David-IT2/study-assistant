"""
Defines the tools the agent can choose to use:
- calculator: basic math
- web_search: free DuckDuckGo search, no API key needed
- search_my_documents: RAG retrieval over whatever is in chroma_db/ (built by ingest.py)
"""
import os

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma

from llm_config import get_embeddings

DB_DIR = "chroma_db"


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '12 * (4 + 3)'. Returns the numeric result."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error evaluating expression: {e}"


web_search = DuckDuckGoSearchRun(
    name="web_search",
    description="Search the live web for current information not found in the user's documents.",
)


def get_document_tool():
    """
    Returns a retriever tool over the local vector database, or None if the
    database hasn't been built yet (i.e. ingest.py hasn't been run).
    """
    if not os.path.isdir(DB_DIR):
        return None

    embeddings = get_embeddings()
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return create_retriever_tool(
        retriever,
        name="search_my_documents",
        description=(
            "Search the user's uploaded documents for relevant information. "
            "Use this for any question that might be answered by their notes or files."
        ),
    )


def get_all_tools():
    tools = [calculator, web_search]
    doc_tool = get_document_tool()
    if doc_tool:
        tools.append(doc_tool)
    return tools
