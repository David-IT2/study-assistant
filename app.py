"""
Research & Study Assistant — a local web app combining:
- LangChain (chains, prompts)
- RAG (retrieval over your own documents)
- Conversational memory
- A tool-using agent (calculator, web search, document search)
- Gradio (the web UI you're looking at)

Run with:  python app.py
Then open the local URL Gradio prints (usually http://127.0.0.1:7860)
"""
import os
import shutil

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from ingest import build_index, DOCS_DIR
from agent import build_agent_executor
from llm_config import using_ollama

load_dotenv()

if using_ollama():
    print("Using local Ollama model — no API key needed.")
else:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or set LLM_PROVIDER=ollama in .env to run fully locally instead."
        )

# Global agent, rebuilt whenever the knowledge base changes
agent_executor = build_agent_executor(verbose=True)


def chat(message, history):
    """Called on every chat message. 'history' is Gradio's list of role/content dicts."""
    chat_history = []
    for turn in history:
        if turn["role"] == "user":
            chat_history.append(HumanMessage(content=turn["content"]))
        else:
            chat_history.append(AIMessage(content=turn["content"]))

    result = agent_executor.invoke({"input": message, "chat_history": chat_history})
    return result["output"]


def add_documents(files):
    """Copies uploaded files into docs/, rebuilds the vector index, and refreshes the agent."""
    global agent_executor

    if not files:
        return "No files selected."

    os.makedirs(DOCS_DIR, exist_ok=True)
    added = []
    for f in files:
        dest = os.path.join(DOCS_DIR, os.path.basename(f.name))
        shutil.copy(f.name, dest)
        added.append(os.path.basename(dest))

    build_index()
    agent_executor = build_agent_executor(verbose=True)  # rebuild so the new docs are searchable

    return f"Added: {', '.join(added)}. Knowledge base rebuilt — you can now ask about these."


mode_label = "🖥️ Running locally with Ollama (free, private)" if using_ollama() else "☁️ Running with OpenAI API"

with gr.Blocks(title="Research & Study Assistant") as demo:
    gr.Markdown(
        f"# 📚 Research & Study Assistant\n"
        f"*{mode_label}*\n\n"
        "Ask questions, do quick math, search the web, or upload your own "
        "documents below and ask about them. The assistant decides which tool to use."
    )

    with gr.Accordion("📁 Add documents to the knowledge base (.pdf or .txt)", open=False):
        file_upload = gr.File(file_count="multiple", label="Choose files")
        add_button = gr.Button("Add to knowledge base")
        upload_status = gr.Textbox(label="Status", interactive=False)
        add_button.click(fn=add_documents, inputs=file_upload, outputs=upload_status)

    gr.ChatInterface(
        fn=chat,
        type="messages",
        examples=[
            "What documents do you currently have access to?",
            "What is 245 * 8?",
            "Summarize the key points in my uploaded documents.",
        ],
    )

if __name__ == "__main__":
    demo.launch()
