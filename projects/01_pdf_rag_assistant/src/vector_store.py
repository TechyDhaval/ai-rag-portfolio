"""
src/vector_store.py
Builds and persists a FAISS vector store from document chunks.

Supports two embedding providers (controlled by EMBEDDING_PROVIDER env var):
  huggingface — sentence-transformers library, runs 100% locally (default)
  ollama      — models served by Ollama daemon (run `ollama serve` first)

Learning note — why local embeddings?
  OpenAI embeddings are higher quality but send your text to external servers
  and cost money per token. Local models keep data private, have no rate limits,
  and are free. For most PDF RAG use cases the quality gap is acceptable.
  See CONCEPTS.md §2 and §10 for a full comparison of free models.

Learning note — model mismatch detection:
  FAISS stores raw float vectors. If you build an index with model A (384 dims)
  and try to query it with model B (768 dims), FAISS raises a cryptic dimension
  error at query time — not at load time. This file saves a model_info.json
  alongside the index and warns you at load time if the configured model does
  not match the one used to build the index, so the error is caught early.
"""

import json
import warnings
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from src.config import config


# ── Embedding factory ─────────────────────────────────────────────────────────

def _get_embeddings(provider: str = None, model: str = None):
    """
    Return an embedding model based on EMBEDDING_PROVIDER config.

    huggingface (default):
        Uses sentence-transformers. Model is downloaded once and cached in
        ~/.cache/huggingface/. No server required.

    ollama:
        Uses Ollama's local REST API (http://localhost:11434 by default).
        Ollama must be installed and running: `ollama serve`
        Pull the model first: `ollama pull nomic-embed-text`

    [Learning] Both providers produce floating-point embeddings that are stored
    identically in FAISS. The difference is purely in how the embedding is
    computed — there is no code change needed in vector_store or rag_chain when
    you switch providers.

    Args:
        provider: Override EMBEDDING_PROVIDER from config (e.g. from UI selection).
        model:    Override EMBEDDING_MODEL from config (e.g. from UI selection).
    """
    provider = (provider or config.EMBEDDING_PROVIDER).lower()
    model = model or config.EMBEDDING_MODEL

    if provider == "ollama":
        # [Learning] OllamaEmbeddings calls Ollama's /api/embeddings endpoint.
        # Ollama quantises models to 4-bit by default, so memory footprint is tiny
        # (nomic-embed-text ≈ 270 MB RAM). Trade-off: needs the Ollama daemon
        # running — not suitable for serverless or CI environments.
        from langchain_community.embeddings import OllamaEmbeddings
        print(f"[VectorStore] Embedding provider: ollama | model: {model}")
        print("[VectorStore] Ensure Ollama is running: `ollama serve`")
        return OllamaEmbeddings(model=model)

    # Default: huggingface
    from langchain_huggingface import HuggingFaceEmbeddings
    print(f"[VectorStore] Embedding provider: huggingface | model: {model}")
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={
            "device": "cpu",
            # [Learning] trust_remote_code=True is required by models that ship
            # custom Python code in their HuggingFace repository (e.g. nomic-embed).
            # It is safe here because we only load models the user explicitly selected.
            "trust_remote_code": True,
        },
        # [Learning] normalize_embeddings=True scales every vector to unit length.
        # This converts cosine similarity into a simple dot product (A · B), which
        # is faster and makes similarity scores consistent in the [0, 1] range.
        # It is required for MMR retrieval and score-threshold filtering to behave
        # correctly. Always enable this unless your retriever explicitly expects
        # unnormalised vectors.
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Model info helpers (mismatch detection) ───────────────────────────────────

def _model_info_path() -> Path:
    return Path(config.FAISS_INDEX_PATH) / "model_info.json"


def _save_model_info(provider: str, model: str) -> None:
    """Persist the active provider + model name alongside the FAISS index."""
    info = {
        "embedding_provider": provider,
        "embedding_model": model,
    }
    _model_info_path().write_text(json.dumps(info, indent=2))


def _check_model_info(provider: str, model: str) -> None:
    """
    Warn if the given embedding model differs from the one that built the index.

    [Learning] FAISS vectors are dimensionality-dependent. Loading a 768-dim
    index and querying it with 384-dim vectors causes a hard crash inside the
    native FAISS C++ library — the Python exception is confusing ('inner_product'
    index error or silent wrong results). Checking model_info.json at load time
    turns that crash into a clear, actionable warning message.
    """
    info_file = _model_info_path()
    if not info_file.exists():
        return  # Index built before mismatch detection was added — skip silently

    saved = json.loads(info_file.read_text())

    if (
        saved.get("embedding_provider") != provider
        or saved.get("embedding_model") != model
    ):
        warnings.warn(
            "\n"
            "  *** Embedding model mismatch detected! ***\n"
            f"  Index was built with : provider={saved.get('embedding_provider')!r}, "
            f"model={saved.get('embedding_model')!r}\n"
            f"  Currently selected   : provider={provider!r}, "
            f"model={model!r}\n"
            "  This will cause dimension errors or silently wrong results.\n"
            "  Fix: re-ingest your documents with the selected model.",
            stacklevel=2,
        )


# ── Public API ────────────────────────────────────────────────────────────────

def build_vector_store(
    chunks: List[Document],
    provider: str = None,
    model: str = None,
) -> FAISS:
    """
    Create a FAISS vector store from document chunks and save it to disk.

    Also saves model_info.json alongside the index so that load_vector_store()
    can detect if the embedding model was changed between sessions.

    Args:
        chunks:   List of Document objects (output of document_loader).
        provider: Embedding provider to use ('huggingface' or 'ollama').
                  Defaults to EMBEDDING_PROVIDER from config / .env.
        model:    Embedding model name. Defaults to EMBEDDING_MODEL from config.

    Returns:
        An in-memory FAISS vector store ready for similarity search.

    Learning note — flat index vs approximate index:
        FAISS defaults to a flat (brute-force) index that performs exact
        nearest-neighbour search. For typical PDF workloads (hundreds to a few
        thousand chunks) this is fast enough (under 10ms per query on CPU).
        For datasets with >100k vectors, switch to an IVF or HNSW index for
        approximate (but much faster) nearest-neighbour search.
    """
    provider = (provider or config.EMBEDDING_PROVIDER).lower()
    model = model or config.EMBEDDING_MODEL

    embeddings = _get_embeddings(provider, model)
    print(f"[VectorStore] Building FAISS index from {len(chunks)} chunk(s)...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    save_path = config.FAISS_INDEX_PATH
    # [Learning] Persisting the index skips the embedding step on subsequent runs.
    # Embedding is the slowest part of ingestion (CPU-bound, seconds to minutes).
    # The saved index contains both the vectors and the original document text.
    # Re-ingest whenever source documents change — FAISS indexes cannot be updated
    # incrementally; you must rebuild the entire index.
    vector_store.save_local(save_path)
    _save_model_info(provider, model)
    print(f"[VectorStore] Index saved to: {save_path}/")

    return vector_store


def load_vector_store(provider: str = None, model: str = None) -> FAISS:
    """
    Load an existing FAISS index from disk.

    Checks model_info.json and warns if the selected embedding model differs
    from the one used to build the index.

    Args:
        provider: Embedding provider ('huggingface' or 'ollama').
                  Defaults to EMBEDDING_PROVIDER from config / .env.
        model:    Embedding model name. Defaults to EMBEDDING_MODEL from config.

    Returns:
        A FAISS vector store loaded from the saved index path.

    Raises:
        FileNotFoundError: If no index has been built yet.

    Learning note — allow_dangerous_deserialization:
        FAISS indexes are stored as pickle files. Python's pickle can execute
        arbitrary code when deserialised, which is a security risk if you load
        an index from an untrusted source. Only load indexes you built yourself.
        The flag exists to make this risk explicit rather than silent.
    """
    index_path = Path(config.FAISS_INDEX_PATH)
    if not index_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{index_path}'. "
            "Run the ingestion step first by providing a PDF."
        )

    provider = (provider or config.EMBEDDING_PROVIDER).lower()
    model = model or config.EMBEDDING_MODEL

    _check_model_info(provider, model)

    embeddings = _get_embeddings(provider, model)
    print(f"[VectorStore] Loading existing FAISS index from: {index_path}/")
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,   # see docstring above
    )
