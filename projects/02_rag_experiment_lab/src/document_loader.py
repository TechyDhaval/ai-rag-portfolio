"""
src/document_loader.py
Loads PDF files and splits them into overlapping text chunks with
configurable chunk_size and chunk_overlap — both settable at runtime.

Key difference from Project 01:
  chunk_size and chunk_overlap are passed as arguments rather than read
  from config at import time. This lets the Streamlit UI re-chunk documents
  with different settings without restarting the app, which is essential
  for comparing how chunk size affects retrieval quality.

Learning note — why re-chunking matters:
  Changing chunk_size doesn't just affect the number of chunks — it changes
  what information each chunk contains, which changes what the embedding
  model sees, which changes which chunks get retrieved. There is no universal
  "best" chunk size; it depends on document structure, embedding model token
  limit, and the types of questions users ask. The Experiment Lab lets you
  measure this empirically.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.config import config

_SUMMARY_PAGES = 2


def _create_summary_chunk(path: Path, pages: List[Document]) -> Document:
    """Build a single Document summarising the first pages of a PDF."""
    overview_pages = pages[:_SUMMARY_PAGES]
    combined_text = "\n\n".join(p.page_content for p in overview_pages)
    return Document(
        page_content=f"[Document Overview]\n{combined_text}",
        metadata={
            "source": str(path),
            "filename": path.name,
            "page": 0,
            "page_display": 1,
        },
    )


def load_and_split_pdf(
    pdf_path: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Load a single PDF and split it into chunks.

    Args:
        pdf_path:      Path to the PDF file.
        chunk_size:    Characters per chunk. Defaults to config.CHUNK_SIZE.
        chunk_overlap: Overlap between consecutive chunks. Defaults to config.CHUNK_OVERLAP.

    Returns:
        List of Document objects with source-attribution metadata.

    Learning note — chunk_size units:
        RecursiveCharacterTextSplitter measures chunk_size in characters, not
        tokens. 1000 characters ≈ 200 tokens for English text. If your
        embedding model has a 256-token limit (e.g. all-MiniLM-L6-v2), keep
        chunk_size ≤ 1200 chars. Models with 512-token limits (e.g. bge-small)
        can handle chunk_size up to ~2500 chars safely.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path.resolve()}")

    print(f"[Loader] Loading PDF: {path.name}")
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    print(f"[Loader] Loaded {len(pages)} page(s)")

    for page in pages:
        page.metadata["filename"] = path.name
        page.metadata["page_display"] = page.metadata.get("page", 0) + 1

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"[Loader] Split into {len(chunks)} chunk(s) "
          f"(size={chunk_size}, overlap={chunk_overlap})")

    summary = _create_summary_chunk(path, pages)
    return [summary] + chunks


def load_and_split_pdfs(
    pdf_paths: List[str],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Load multiple PDFs and return all chunks as a single list.

    Args:
        pdf_paths:     List of paths to PDF files.
        chunk_size:    Characters per chunk (passed to each PDF).
        chunk_overlap: Overlap between chunks (passed to each PDF).

    Returns:
        Combined list of Document chunks from all PDFs.
    """
    all_chunks: List[Document] = []
    for pdf_path in pdf_paths:
        chunks = load_and_split_pdf(pdf_path, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    print(f"[Loader] Total chunks across {len(pdf_paths)} PDF(s): {len(all_chunks)}")
    return all_chunks
