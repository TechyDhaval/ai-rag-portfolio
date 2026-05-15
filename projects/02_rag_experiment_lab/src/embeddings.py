"""
src/embeddings.py
Multi-provider embedding factory with runtime model switching.

This module centralises embedding creation so the rest of the codebase
never imports provider-specific classes directly. When the user switches
models in the UI, a new embedding object is returned — no restart needed.

Supported providers:
  huggingface — sentence-transformers library, runs 100% locally (default)
  ollama      — models served by Ollama daemon (run `ollama serve` first)

Learning note — why a dedicated module?
  In Project 01, the embedding factory lived inside vector_store.py. Here
  it is extracted because the Experiment Lab needs to create embeddings in
  multiple places: vector store building, experiment evaluation, and
  comparison across different models. A single factory avoids duplication
  and ensures consistent settings (e.g. normalize_embeddings=True).

Learning note — model download behaviour:
  HuggingFace models are downloaded to ~/.cache/huggingface/ the first time
  they are used. Subsequent loads are instant. The download size ranges from
  ~90 MB (all-MiniLM-L6-v2) to ~1.3 GB (bge-large-en-v1.5). Plan accordingly
  when switching models in the UI — the first switch triggers a download.
"""

from src.config import config


# Curated free HuggingFace embedding models with metadata for the UI.
# Keys are the HuggingFace model IDs. Values are dicts with display info.
EMBEDDING_MODELS = {
    "BAAI/bge-small-en-v1.5": {
        "dim": 384, "max_tokens": 512, "size_mb": 130,
        "desc": "Best speed/quality ratio — default choice",
    },
    "BAAI/bge-base-en-v1.5": {
        "dim": 768, "max_tokens": 512, "size_mb": 430,
        "desc": "Balanced step-up in quality",
    },
    "BAAI/bge-large-en-v1.5": {
        "dim": 1024, "max_tokens": 512, "size_mb": 1300,
        "desc": "Near-API quality, large model",
    },
    "nomic-ai/nomic-embed-text-v1": {
        "dim": 768, "max_tokens": 8192, "size_mb": 550,
        "desc": "Best for long documents (8K token context)",
    },
    "all-mpnet-base-v2": {
        "dim": 768, "max_tokens": 514, "size_mb": 420,
        "desc": "Strong general-purpose model",
    },
    "all-MiniLM-L6-v2": {
        "dim": 384, "max_tokens": 256, "size_mb": 90,
        "desc": "Ultra-fast, smallest model",
    },
}


def get_embeddings(provider: str = None, model: str = None):
    """
    Return an embedding model instance.

    Both providers produce floating-point vectors stored identically in FAISS.
    The only difference is in how the embedding is computed — all downstream
    code (vector store, retriever, chain) is provider-agnostic.

    Args:
        provider: 'huggingface' or 'ollama'. Defaults to config.EMBEDDING_PROVIDER.
        model:    Model name/ID. Defaults to config.EMBEDDING_MODEL.

    Returns:
        A LangChain Embeddings instance ready for use.

    Learning note — normalize_embeddings:
        Setting normalize_embeddings=True scales every vector to unit length.
        This converts cosine similarity into a simple dot product (A · B),
        which is faster and makes similarity scores consistent in [0, 1].
        Required for MMR retrieval and score-threshold filtering.
    """
    provider = (provider or config.EMBEDDING_PROVIDER).lower()
    model = model or config.EMBEDDING_MODEL

    if provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        print(f"[Embeddings] Provider: ollama | Model: {model}")
        return OllamaEmbeddings(model=model)

    # Default: huggingface
    from langchain_huggingface import HuggingFaceEmbeddings
    print(f"[Embeddings] Provider: huggingface | Model: {model}")
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={
            "device": "cpu",
            "trust_remote_code": True,
        },
        encode_kwargs={"normalize_embeddings": True},
    )
