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

## 1. Setup (one-time)

**Requirements:** Python 3.10 or newer.

```bash
# 1. Unzip this project and open a terminal in the folder
cd study-assistant

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file
cp .env.example .env             # Windows: copy .env.example .env
```

Now choose **one** of the two options below.

### Option A — OpenAI (easiest, costs a small amount per request)

Open `.env` and:
- Leave `LLM_PROVIDER=openai`
- Paste your key from https://platform.openai.com/api-keys into `OPENAI_API_KEY`

### Option B — Ollama (100% free, runs fully on your computer, no key, no internet needed after setup)

1. Install Ollama from **https://ollama.com** (available for Mac, Windows, Linux).
2. Pull a chat model and an embedding model (one-time download, a few GB):
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
3. Make sure Ollama is running (the installer usually starts it automatically;
   otherwise run `ollama serve` in a separate terminal).
4. Open `.env` and set:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3.2
   OLLAMA_EMBED_MODEL=nomic-embed-text
   ```
   You can leave `OPENAI_API_KEY` blank — it's ignored in this mode.

**Notes on Ollama mode:**
- Everything (chat + document embeddings) runs on your machine — nothing is sent to OpenAI.
- Web search still works normally (DuckDuckGo doesn't need a key either way).
- Response quality and speed depend on your computer's hardware. `llama3.2` (3B) runs
  fine on most modern laptops; if you have a beefier machine and want better answers,
  try `ollama pull llama3.1` (8B) or `ollama pull qwen2.5:14b` and update `OLLAMA_MODEL`.
- If you switch providers later, delete `chroma_db/` and re-run `python ingest.py` —
  embeddings from different providers aren't compatible with each other.

## 2. (Optional) Add documents before you start

Drop `.pdf` or `.txt` files into the `docs/` folder, then run:

```bash
python ingest.py
```

This builds a local vector database (`chroma_db/`) so the assistant can
search these files. You can skip this step and upload files directly in
the web app instead (see below).

## 3. Run the app

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

## Troubleshooting

- **"OPENAI_API_KEY is not set"** — make sure you created `.env` (not just
  `.env.example`) and it contains your real key.
- **Agent isn't using a tool you expect** — this is normal occasionally;
  rephrase the question to be more explicit (e.g. "search my documents for...").
- **Want to start the knowledge base over** — delete the `chroma_db/` folder
  and re-run `python ingest.py`, or re-upload files in the app.
- **Rate limit errors** — you may be on a free/trial OpenAI tier with low
  limits; check your usage at platform.openai.com.

## How it works (short version)

1. `ingest.py` splits your documents into chunks, converts them to
   embeddings, and stores them in a local Chroma vector database.
2. `tools.py` wraps that database as a searchable "tool," alongside a
   calculator and a free web search tool.
3. `agent.py` gives an LLM access to all three tools and lets it decide,
   per question, which (if any) to use.
4. `app.py` wraps that agent in a Gradio chat interface, and adds a file
   upload panel that re-runs ingestion whenever you add new documents.
