"""
src/document_loader.py
Loads one or more PDF files and splits them into overlapping text chunks
ready for embedding. Uses LangChain's PyPDFLoader and
RecursiveCharacterTextSplitter.

Each chunk carries:
  - metadata["filename"]     — base filename of the source PDF
  - metadata["page_display"] — 1-indexed page number
  - metadata["page"]         — 0-indexed page number (set by PyPDFLoader)

A "Document Overview" summary chunk is also prepended for each PDF so that
broad questions like "what is this document about?" reliably find context.

Learning note — why this loader choice?
  PyPDFLoader is simple, pure-Python, and works for most machine-readable PDFs.
  For scanned PDFs (images of text), tables, or multi-column layouts, consider:
    - UnstructuredPDFLoader  — handles complex layouts, tables, images
    - AzureAIDocumentIntelligenceLoader — cloud OCR, highest quality
  See CONCEPTS.md §3 and ARCHITECTURE.md for a fuller comparison.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.config import config

# Number of pages to include in the summary chunk per PDF
_SUMMARY_PAGES = 2


def _create_summary_chunk(path: Path, pages: List[Document]) -> Document:
    """
    Build a single Document that summarises the beginning of a PDF.

    Concatenates the first _SUMMARY_PAGES pages so that broad questions
    ("what is this document about?") hit a chunk with high-level context.

    Learning note — why not use a keyword prefix for retrieval?
      The "[Document Overview]" text is not a retrieval keyword — semantic
      search does not match on strings. The chunk is retrieved because the
      abstract and introduction (pages 1-2) have high embedding similarity to
      broad questions like "what is this thesis about?". Concatenating both
      pages into a single chunk ensures the LLM receives the full overview
      rather than one fragment from a split abstract.
    """
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


def load_and_split_pdf(pdf_path: str) -> List[Document]:
    """
    Load a single PDF and return a list of chunked Document objects.

    The list starts with a summary chunk covering the first pages, followed
    by all regular overlap-split chunks.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of LangChain Document objects with source-attribution metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path.resolve()}")

    print(f"[Loader] Loading PDF: {path.name}")
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    print(f"[Loader] Loaded {len(pages)} page(s)")

    # Enrich metadata so every downstream chunk carries its source identity.
    # filename and page_display are used by _DOCUMENT_PROMPT in rag_chain.py
    # to generate citations like "[Source: thesis.pdf, Page 3]".
    for page in pages:
        page.metadata["filename"] = path.name
        page.metadata["page_display"] = page.metadata.get("page", 0) + 1

    # [Learning] RecursiveCharacterTextSplitter tries separators in priority
    # order: paragraph break (\n\n) → line break (\n) → space → char.
    # This keeps paragraphs intact wherever possible, which preserves the
    # semantic coherence of each chunk better than fixed-character splitting.
    #
    # chunk_size=1000 chars ≈ 200 tokens — chosen to match all-MiniLM-L6-v2's
    # 256-token input limit. Text beyond 256 tokens is silently truncated by
    # the embedding model, so keeping chunks at ~200 tokens provides headroom.
    #
    # chunk_overlap=200 chars ensures sentences that straddle a chunk boundary
    # appear in both adjacent chunks, preventing context from being cut off.
    # A good rule of thumb: overlap ≈ 10–20% of chunk_size.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"[Loader] Split into {len(chunks)} chunk(s) "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    # Prepend the summary chunk so broad questions always have context
    summary = _create_summary_chunk(path, pages)
    return [summary] + chunks


def load_and_split_pdfs(pdf_paths: List[str]) -> List[Document]:
    """
    Load multiple PDFs and return all chunks as a single list.

    Each PDF is loaded and split independently; the resulting chunks are
    merged. Metadata on each chunk includes its source filename and page.

    Args:
        pdf_paths: List of absolute or relative paths to PDF files.

    Returns:
        Combined list of Document chunks from all provided PDFs.

    Raises:
        FileNotFoundError: If any PDF does not exist.
    """
    all_chunks: List[Document] = []
    for pdf_path in pdf_paths:
        chunks = load_and_split_pdf(pdf_path)
        all_chunks.extend(chunks)

    print(f"[Loader] Total chunks across {len(pdf_paths)} PDF(s): {len(all_chunks)}")
    return all_chunks
