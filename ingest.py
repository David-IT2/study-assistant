"""
Builds (or rebuilds) the local vector database from files in the docs/ folder.

Run this whenever you add or change files in docs/:
    python ingest.py

Supports .pdf, .docx, and .txt files.
"""
import os
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from llm_config import get_embeddings

load_dotenv()

DOCS_DIR = "docs"
DB_DIR = "chroma_db"


def load_documents():
    documents = []
    if not os.path.isdir(DOCS_DIR):
        return documents

    for filename in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, filename)
        lower = filename.lower()
        try:
            if lower.endswith(".pdf"):
                documents.extend(PyPDFLoader(path).load())
            elif lower.endswith(".docx"):
                documents.extend(Docx2txtLoader(path).load())
            elif lower.endswith(".txt"):
                documents.extend(TextLoader(path, encoding="utf-8").load())
        except Exception as e:
            print(f"Skipping '{filename}' - could not read it ({type(e).__name__})")
    return documents


def build_index():
    documents = load_documents()
    if not documents:
        print(
            f"No .pdf, .docx, or .txt files found in '{DOCS_DIR}/'. "
            "Add some files there first, then re-run this script."
        )
        return

    print(f"Loaded {len(documents)} document page(s)/file(s) from {DOCS_DIR}/")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Wipe any existing database so re-running this script doesn't duplicate chunks
    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR)

    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    print(f"Vector database built and saved to {DB_DIR}/")
    print("You can now run: python app.py")


if __name__ == "__main__":
    build_index()
