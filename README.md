# Research & Study Assistant

A local web app that combines LangChain, RAG, conversational memory, and a
tool-using agent (calculator + web search + your own documents), with a
Gradio chat interface.

## What it does

- Chat with an AI assistant in your browser
- Upload your own `.pdf` or `.txt` files and ask questions about them (RAG)
- The assistant automatically decides whether to search your documents,
  do a calculation, or search the live web to answer a question
- Remembers context within a conversation



## . (Optional) Add documents before you start

Drop `.pdf` or `.txt` files into the `docs/` folder, then run:

```bash
python ingest.py
```

This builds a local vector database (`chroma_db/`) so the assistant can
search these files. You can skip this step and upload files directly in
the web app instead.

## . Run the app

```bash
python app.py
```

Open the URL printed in the terminal (usually `http://127.0.0.1:7860`).

## 4. Using the app

- Type in the chat box to ask anything
- Expand "📁 Add documents to the knowledge base" to upload files directly
  from the browser — the index rebuilds automatically
- Try a math question, a general knowledge question, and a question about
  your uploaded documents, to see the agent pick different tools for each

## Project structure

```
study-assistant/
├── app.py           # Gradio web app (run this)
├── agent.py         # Builds the tool-calling agent
├── tools.py         # Calculator, web search, document retriever tools
├── ingest.py        # Builds the vector database from docs/
├── llm_config.py    # Switches between OpenAI and local Ollama models
├── docs/            # Put your .pdf / .txt files here
├── chroma_db/        # Auto-created — the vector database (safe to delete/rebuild)
├── requirements.txt
├── .env.example      # Copy to .env — set LLM_PROVIDER here (openai or ollama)
└── README.md
```

## How it works (short version)

1. `ingest.py` splits your documents into chunks, converts them to
   embeddings, and stores them in a local Chroma vector database.
2. `tools.py` wraps that database as a searchable "tool," alongside a
   calculator and a free web search tool.
3. `agent.py` gives an LLM access to all three tools and lets it decide,
   per question, which (if any) to use.
4. `app.py` wraps that agent in a Gradio chat interface, and adds a file
   upload panel that re-runs ingestion whenever you add new documents.
