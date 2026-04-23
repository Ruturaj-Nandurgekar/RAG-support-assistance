# ingest.py
"""
Ingests a PDF into ChromaDB with HuggingFace embeddings.
Run once (or after uploading a new PDF) to rebuild the index.
"""

import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_FOLDER = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PDF_PATH = "knowledge_base.pdf"


def main(pdf_path: str = PDF_PATH):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"[ingest] ❌ PDF not found: {pdf_path}")
        return False

    # ── Load ──────────────────────────────────────────────────────────────────
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    print(f"[ingest] Loaded {len(pages)} page(s) from {pdf_path}")

    # ── Split ─────────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    print(f"[ingest] Split into {len(chunks)} chunk(s).")

    # ── Wipe old index so we don't accumulate duplicate docs ──────────────────
    if Path(CHROMA_FOLDER).exists():
        shutil.rmtree(CHROMA_FOLDER)
        print(f"[ingest] Cleared old ChromaDB at '{CHROMA_FOLDER}'.")

    # ── Embed & store ─────────────────────────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_FOLDER,
    )
    # Note: Chroma >= 0.4 auto-persists; explicit .persist() is deprecated.
    print(f"[ingest] ✅ ChromaDB built at '{CHROMA_FOLDER}' ({len(chunks)} chunks).")
    return True


if __name__ == "__main__":
    main()