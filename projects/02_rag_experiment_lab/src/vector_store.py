"""
src/vector_store.py
Builds and persists FAISS vector stores with configurable embedding models.

Key difference from Project 01:
  Supports building indexes into named sub-directories under the base path,
  keyed by embedding model name. This allows keeping indexes for different
  embedding models side-by-side so you can switch models without re-ingesting
  every time.

Learning note — why separate indexes per model?
  FAISS stores raw float vectors. Vectors from model A (384 dims) cannot be
  queried with model B (768 dims). Even two 384-dim models produce vectors in
  different semantic spaces, making cross-model queries meaningless. Each
  model needs its own index.
"""

import json
import warnings
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from src.config import config
from src.embeddings import get_embeddings


def _index_dir(base_path: str, model: str) -> Path:
    """Return the sub-directory for a specific model's FAISS index."""
    safe_name = model.replace("/", "_").replace("\\", "_")
    return Path(base_path) / safe_name


def _save_model_info(index_path: Path, provider: str, model: str) -> None:
    """Persist embedding metadata alongside the FAISS index."""
    info = {"embedding_provider": provider, "embedding_model": model}
    (index_path / "model_info.json").write_text(json.dumps(info, indent=2))


def _read_model_info(index_path: Path) -> dict:
    """Read model_info.json from an index directory, or return {}."""
    info_file = index_path / "model_info.json"
    if info_file.exists():
        try:
            return json.loads(info_file.read_text())
        except Exception:
            pass
    return {}


def build_vector_store(
    chunks: List[Document],
    provider: str = None,
    model: str = None,
    base_path: str = None,
) -> FAISS:
    """
    Create a FAISS vector store from chunks and save it to disk.

    The index is saved into a sub-directory named after the model so that
    indexes for different embedding models coexist on disk.

    Args:
        chunks:    List of Document objects to embed.
        provider:  Embedding provider ('huggingface' or 'ollama').
        model:     Embedding model name.
        base_path: Root directory for FAISS indexes. Defaults to config.FAISS_INDEX_PATH.

    Returns:
        An in-memory FAISS vector store ready for search.
    """
    provider = (provider or config.EMBEDDING_PROVIDER).lower()
    model = model or config.EMBEDDING_MODEL
    base_path = base_path or config.FAISS_INDEX_PATH

    embeddings = get_embeddings(provider, model)
    print(f"[VectorStore] Building FAISS index from {len(chunks)} chunk(s)...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    save_path = _index_dir(base_path, model)
    save_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_path))
    _save_model_info(save_path, provider, model)
    print(f"[VectorStore] Index saved to: {save_path}/")

    return vector_store


def load_vector_store(
    provider: str = None,
    model: str = None,
    base_path: str = None,
) -> FAISS:
    """
    Load an existing FAISS index from disk.

    Args:
        provider:  Embedding provider ('huggingface' or 'ollama').
        model:     Embedding model name.
        base_path: Root directory for FAISS indexes.

    Returns:
        A FAISS vector store loaded from disk.

    Raises:
        FileNotFoundError: If no index exists for the given model.
    """
    provider = (provider or config.EMBEDDING_PROVIDER).lower()
    model = model or config.EMBEDDING_MODEL
    base_path = base_path or config.FAISS_INDEX_PATH

    index_path = _index_dir(base_path, model)
    if not index_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{index_path}'. "
            "Run ingestion first to build an index for this model."
        )

    saved_info = _read_model_info(index_path)
    if saved_info and saved_info.get("embedding_model") != model:
        warnings.warn(
            f"Index model mismatch: saved={saved_info.get('embedding_model')!r}, "
            f"requested={model!r}. Results may be wrong.",
            stacklevel=2,
        )

    embeddings = get_embeddings(provider, model)
    print(f"[VectorStore] Loading FAISS index from: {index_path}/")
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def index_exists(model: str = None, base_path: str = None) -> bool:
    """Check if a FAISS index exists on disk for the given model."""
    model = model or config.EMBEDDING_MODEL
    base_path = base_path or config.FAISS_INDEX_PATH
    index_path = _index_dir(base_path, model)
    return (index_path / "index.faiss").exists()
