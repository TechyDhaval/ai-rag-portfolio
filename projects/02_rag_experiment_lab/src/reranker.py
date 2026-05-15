"""
src/reranker.py
Cross-encoder reranking module — re-scores retrieved chunks using a model
that sees both the query and the document together.

Learning note — bi-encoder vs cross-encoder:
  The embedding model (bi-encoder) encodes query and document independently
  into separate vectors, then compares them with cosine similarity. This is
  fast (embeddings can be precomputed) but approximate — it cannot capture
  fine-grained query-document interactions.

  A cross-encoder takes the (query, document) pair as a single input and
  outputs a relevance score directly. This is much more accurate because it
  can attend to the interaction between query and document tokens, but it
  cannot be precomputed — it must be run at query time for each candidate.

  The standard pattern: use the bi-encoder for fast recall (retrieve top-k),
  then use the cross-encoder to rerank those k candidates for precision.

Learning note — latency impact:
  Cross-encoders add ~100-300ms per query on CPU (for k ≤ 10 candidates).
  This is acceptable for interactive use but can add up in batch evaluation.
  The Experiment Lab lets you toggle reranking on/off to measure the
  quality-vs-latency tradeoff empirically.

Learning note — model choice:
  ms-marco-MiniLM-L-6-v2 is the standard lightweight reranker (~80 MB).
  For higher quality, try ms-marco-MiniLM-L-12-v2 (~130 MB) or
  BAAI/bge-reranker-base (~1.1 GB). The right choice depends on whether
  your bottleneck is retrieval quality or latency.
"""

from typing import List

from langchain.schema import Document

from src.config import config


def rerank(
    question: str,
    docs: List[Document],
    top_n: int = None,
    model_name: str = None,
) -> List[Document]:
    """
    Re-score and re-order documents using a cross-encoder model.

    Args:
        question:   The user's query.
        docs:       List of retrieved Document objects to rerank.
        top_n:      Number of top documents to return after reranking.
                    Defaults to config.RERANKER_TOP_N.
        model_name: Cross-encoder model name. Defaults to config.RERANKER_MODEL.

    Returns:
        A list of the top_n most relevant Documents, ordered by cross-encoder
        score (highest first).

    Learning note — why not rerank inside the retriever?
        LangChain retrievers return Documents but don't expose a hook for
        post-retrieval processing. Reranking is applied after retrieval and
        before the QA chain, as a separate pipeline step. This keeps each
        component single-responsibility and makes it easy to toggle on/off.
    """
    if not docs:
        return docs

    top_n = top_n or config.RERANKER_TOP_N
    model_name = model_name or config.RERANKER_MODEL

    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(model_name)

    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    result = [doc for _, doc in scored_docs[:top_n]]

    print(f"[Reranker] {len(docs)} → {len(result)} docs "
          f"(model={model_name}, top_n={top_n})")
    return result
