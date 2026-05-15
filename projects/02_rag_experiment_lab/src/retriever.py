"""
src/retriever.py
Configurable retriever factory — builds a LangChain retriever from a FAISS
vector store with runtime-selectable search strategy and parameters.

Supported search types:
  similarity                  — plain cosine similarity, returns top-k nearest
  mmr                         — Maximal Marginal Relevance (relevance + diversity)
  similarity_score_threshold  — cosine similarity with a minimum score cutoff

Learning note — when to use each:
  - similarity: simplest; good baseline when your chunks are already diverse
    (e.g. different pages of different documents).
  - mmr: best default for single-document RAG — prevents k near-duplicate chunks
    from the same passage dominating the context.
  - similarity_score_threshold: use when you need to gracefully handle off-topic
    questions. Returns fewer (or zero) results if nothing is relevant, so the
    LLM can say "I don't know" rather than hallucinating from irrelevant chunks.

Learning note — MMR lambda_mult:
  Controls the relevance-vs-diversity tradeoff:
    lambda_mult=1.0 → pure relevance (identical to similarity search)
    lambda_mult=0.0 → pure diversity (maximise distance between selected chunks)
    lambda_mult=0.5 → balanced (default in most LangChain examples)
  Start at 0.5, decrease if you see too many near-duplicate chunks in results.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

from src.config import config


# Mapping of search_type values to human descriptions for UI display
SEARCH_TYPES = {
    "similarity": "Cosine similarity — simple nearest-neighbour",
    "mmr": "MMR — relevance + diversity (recommended)",
    "similarity_score_threshold": "Similarity with minimum score threshold",
}


def build_retriever(
    vector_store: FAISS,
    search_type: str = None,
    top_k: int = None,
    score_threshold: float = None,
    mmr_lambda: float = None,
) -> BaseRetriever:
    """
    Create a retriever from a FAISS vector store with configurable search.

    All parameters default to config values but can be overridden at runtime
    by the Streamlit UI, enabling live experimentation.

    Args:
        vector_store:    A populated FAISS vector store.
        search_type:     'similarity', 'mmr', or 'similarity_score_threshold'.
        top_k:           Number of chunks to retrieve.
        score_threshold: Minimum similarity score (only for threshold search).
        mmr_lambda:      Lambda multiplier for MMR (0=diversity, 1=relevance).

    Returns:
        A LangChain BaseRetriever instance.

    Learning note — fetch_k for MMR:
        MMR first retrieves fetch_k candidates by pure similarity, then
        re-ranks them for diversity to select the final top_k. A pool of
        4× top_k gives enough diversity without excessive computation.
    """
    search_type = search_type or config.SEARCH_TYPE
    top_k = top_k or config.TOP_K
    score_threshold = score_threshold if score_threshold is not None else config.SCORE_THRESHOLD
    mmr_lambda = mmr_lambda if mmr_lambda is not None else config.MMR_LAMBDA

    search_kwargs = {"k": top_k}

    if search_type == "mmr":
        search_kwargs["fetch_k"] = top_k * 4
        search_kwargs["lambda_mult"] = mmr_lambda
    elif search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = score_threshold

    print(f"[Retriever] type={search_type}, k={top_k}, kwargs={search_kwargs}")

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
