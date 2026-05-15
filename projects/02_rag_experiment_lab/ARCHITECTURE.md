# Architecture Decision Record

This document explains how data flows through the system, why each component
exists, and the trade-offs behind the design choices. Read alongside the source
code — each module has inline learning notes explaining the "why".

---

## System Overview

The application has three pipelines:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INGESTION PIPELINE  (re-runs when you change embedding model or chunk size)│
│                                                                             │
│  PDF(s) ──► Loader ──► Splitter ──► Embedder ──► FAISS Index               │
│                │            │            │            │                      │
│          configurable  chunk_size   model choice  per-model dir             │
│                        chunk_overlap                                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  QUERY PIPELINE  (runs on every question)                                   │
│                                                                             │
│  Question ──► Rewriter ──► Retriever ──► Reranker ──► QA Chain ──► Answer   │
│                  │            │              │             │                 │
│            chat_history   search_type     optional    prompt_key            │
│                           top_k         cross-encoder  temperature          │
│                           mmr_lambda                   llm_model            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  EVALUATION PIPELINE  (runs on explicit evaluation requests)                │
│                                                                             │
│  Test Set ──► ask() × N ──► LLM-as-Judge ──► Scores ──► Experiment JSON     │
│                   │               │                          │               │
│            (query pipeline)  3 metrics/question         save & compare      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Design Decisions

### Why separate `embeddings.py` from `vector_store.py`?

In Project 01, `_get_embeddings()` lived inside `vector_store.py`. Here it's
a separate module because:
1. The Experiment Lab needs embeddings in multiple places (building index,
   loading index, switching models in the UI)
2. The embedding model catalogue (6 models with metadata) is a significant
   data structure that deserves its own home
3. When comparing embedding models, you need to iterate the catalogue — that
   logic doesn't belong in the vector store

### Why per-model FAISS index directories?

```
faiss_index/
  bge-small-en-v1.5/
    index.faiss
    index.pkl
  bge-base-en-v1.5/
    index.faiss
    index.pkl
```

Each embedding model produces vectors of different dimensions (384 vs 768 vs
1024). You **cannot** mix them in one FAISS index — the dot products would be
meaningless. Separate directories let you switch models without re-indexing
every time.

### Why a dedicated `retriever.py`?

In Project 01, the retriever was built inline in `rag_chain.py`. Here, retrieval
configuration is a first-class experimental variable, so it gets its own module
with a documented `SEARCH_TYPES` catalogue and configurable `build_retriever()`.

### Why `ask()` is separate from `build_rag_chain()`?

```python
# Conversational (chat tab) — has memory
chain = build_rag_chain(...)
chain.invoke({"input": question}, config={"configurable": {"session_id": "..."}})

# Stateless (evaluation tab) — no memory, returns timing + docs
result = ask(question, vector_store, ...)
```

Evaluation needs:
- No conversation memory (each question is independent)
- Retrieved documents returned (for context relevance scoring)
- Timing breakdown (retrieval vs LLM latency)
- Deterministic results (no prior Q&A influencing the answer)

The chat chain can't provide these, hence the separate `ask()` function.

---

## Evaluation Methodology

### LLM-as-Judge

Each metric uses a separate LLM call with a carefully crafted system prompt
that constrains the judge to return a float between 0.0 and 1.0.

```
┌──────────────────────────────────────────────────────────────────┐
│  Context Relevance                                               │
│  Input:  question + each chunk individually                      │
│  Output: average relevance score across all chunks               │
│  Diagnoses: retrieval quality (embeddings, search type, top_k)   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Faithfulness                                                    │
│  Input:  question + answer + all context chunks                  │
│  Output: single score (is the answer grounded in context?)       │
│  Diagnoses: hallucination (prompt, temperature, model)           │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Answer Relevance                                                │
│  Input:  question + answer (no context)                          │
│  Output: single score (does the answer address the question?)    │
│  Diagnoses: generation quality (LLM model, prompt template)      │
└──────────────────────────────────────────────────────────────────┘
```

### Why these three metrics?

Together they cover every stage of the RAG pipeline:

1. **Context Relevance** → tests the **retriever** (embedding + search)
2. **Faithfulness** → tests **grounding** (prompt + temperature)
3. **Answer Relevance** → tests the **generator** (LLM + prompt)

If you have a bad score, you know exactly where to look:
- Low context relevance → change embedding model, chunk size, or search strategy
- Low faithfulness → use stricter prompt, lower temperature, or enable reranker
- Low answer relevance → use a better LLM model or a different prompt template

---

## Experiment Storage Format

Experiments are saved as JSON files with this structure:

```json
{
  "timestamp": "2024-03-12_14-30-00",
  "label": "baseline_strict",
  "params": {
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "search_type": "mmr",
    "top_k": 6,
    "score_threshold": 0.4,
    "mmr_lambda": 0.5,
    "llm_model": "gpt-4o-mini",
    "temperature": 0.0,
    "prompt_key": "strict",
    "reranker_enabled": false,
    "reranker_top_n": 4
  },
  "aggregates": {
    "avg_context_relevance": 0.82,
    "avg_faithfulness": 0.91,
    "avg_answer_relevance": 0.87,
    "avg_retrieval_time_s": 0.15,
    "avg_llm_time_s": 1.20,
    "avg_total_time_s": 1.35,
    "num_questions": 5
  },
  "results": [
    {
      "question": "What is the main topic?",
      "answer": "...",
      "context_relevance": 0.85,
      "faithfulness": 0.90,
      "answer_relevance": 0.88,
      "retrieval_time_s": 0.12,
      "llm_time_s": 1.10,
      "total_time_s": 1.22
    }
  ]
}
```

This format captures everything needed to reproduce and compare experiments.

---

## Design Trade-offs

### Local embeddings vs. API embeddings

| Factor           | Local (HuggingFace)          | API (OpenAI)                  |
|------------------|------------------------------|-------------------------------|
| Cost             | Free                         | ~$0.02 per 1M tokens          |
| Speed (first)    | Slow (model download)        | Fast (no download)            |
| Speed (ongoing)  | Fast (no network)            | Network-dependent             |
| Quality          | Good to excellent            | Excellent                     |
| Privacy          | Data stays on your machine   | Sent to external API          |
| Offline          | Works offline                | Requires internet             |

This project uses local embeddings because the goal is learning and
experimentation — you don't want API costs piling up while trying
different configurations.

### Reranking: why and when

Cross-encoder reranking is a two-stage pipeline:
1. FAISS retrieves top_k candidates cheaply using bi-encoder similarity
2. Cross-encoder re-scores each candidate with the full question-document pair

This is more accurate because the cross-encoder sees both texts simultaneously
(bi-encoders encode them separately). But it's O(k) expensive — each candidate
requires a model inference.

**Use when:**
- You retrieve many candidates (top_k ≥ 6) and want to keep only the best
- Your embedding model is good at recall but noisy on precision
- You're building a system where precision matters more than latency

**Skip when:**
- Top_k is small (≤ 3) — not enough candidates to rerank
- Latency is critical — reranking adds 100-500ms on CPU
- Your embeddings are already very precise
