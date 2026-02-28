# AI Engineering Concepts — Reference Guide

This file explains every major technique used in this project.
Use it as a cheat-sheet when building your next AI application.

---

## Table of Contents

1. [Retrieval-Augmented Generation (RAG)](#1-retrieval-augmented-generation-rag)
2. [Text Embeddings & Semantic Search](#2-text-embeddings--semantic-search)
3. [Text Chunking](#3-text-chunking)
4. [Vector Stores (FAISS)](#4-vector-stores-faiss)
5. [Maximal Marginal Relevance (MMR)](#5-maximal-marginal-relevance-mmr)
6. [Conversation Memory](#6-conversation-memory)
7. [History-Aware Retrieval](#7-history-aware-retrieval)
8. [LangChain LCEL](#8-langchain-lcel)
9. [LLM Temperature](#9-llm-temperature)
10. [Local vs API Embeddings](#10-local-vs-api-embeddings)

---

## 1. Retrieval-Augmented Generation (RAG)

### What it is
RAG combines a **retrieval system** (finds relevant information) with a **generative LLM** (synthesises an answer). Instead of asking the LLM to answer purely from its training weights, you first fetch relevant text from your own data and inject it into the prompt.

### Why it matters
LLMs hallucinate — they generate plausible-sounding but fabricated answers when they lack knowledge. RAG grounds the model in real, retrieved evidence, making answers both accurate and verifiable.

### The two-phase pipeline

```
OFFLINE (once per document)          ONLINE (per user query)
───────────────────────────          ──────────────────────────────
PDF → Chunk → Embed → Store          Query → Embed → Retrieve → Prompt → LLM → Answer
```

### When to use RAG vs fine-tuning
| Approach | Use when |
|----------|----------|
| RAG | Knowledge changes frequently, documents are proprietary, you need citations |
| Fine-tuning | You want to change the model's style/tone/format, knowledge is stable |
| Both | High-stakes domains where both style and factual grounding matter |

### Key RAG failure modes to know
- **Retrieval failure** — the right chunk was never fetched (fix: better chunking, more top_k)
- **Context window overflow** — too many chunks, LLM loses track (fix: reduce top_k, summarise)
- **Lost-in-the-middle** — LLM ignores chunks in the middle of a long context (fix: reranking, keep top_k small)
- **Irrelevant retrieval** — chunks fetched are not actually useful (fix: MMR, better embeddings, metadata filtering)

---

## 2. Text Embeddings & Semantic Search

### What it is
An embedding is a **fixed-length vector of floating-point numbers** that encodes the semantic meaning of a piece of text. Two texts with similar meaning will have vectors that are close together in the high-dimensional embedding space.

### How similarity is measured
This project uses **cosine similarity** (angle between vectors), normalised to [0, 1]. A score of 1.0 means identical meaning; 0.0 means completely unrelated.

```
similarity = (A · B) / (|A| × |B|)
```

`normalize_embeddings=True` in this project pre-normalises all vectors so cosine similarity reduces to a dot product — fast and memory-efficient.

### The default model: `BAAI/bge-small-en-v1.5`
- **Architecture**: BERT-based, fine-tuned on BAAI's BGE retrieval dataset
- **Output dimension**: 384
- **Max input tokens**: 512 (2× the limit of all-MiniLM-L6-v2)
- **Strength**: best speed/quality ratio among free models; 512-token limit means 1000-char chunks are embedded completely without truncation
- **Weakness**: English-only; lower quality than larger BGE models for very nuanced queries

### Free model catalogue (all local, no API key)

| Model | Dim | Max tok | Size | Best for |
|-------|-----|---------|------|----------|
| `BAAI/bge-small-en-v1.5` *(default)* | 384 | 512 | ~130 MB | Speed + quality balance |
| `BAAI/bge-base-en-v1.5` | 768 | 512 | ~430 MB | Step-up quality, still fast |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 | ~1.3 GB | Near-API quality |
| `nomic-ai/nomic-embed-text-v1` | 768 | 8192 | ~550 MB | Very long documents |
| `all-mpnet-base-v2` | 768 | 514 | ~420 MB | Strong general-purpose |
| `all-MiniLM-L6-v2` | 384 | 256 | ~90 MB | Ultra-fast, limited context |

**Ollama models** (requires `ollama serve`):

| Model | Dim | Max tok | Notes |
|-------|-----|---------|-------|
| `nomic-embed-text` | 768 | 8192 | Best all-around via Ollama |
| `mxbai-embed-large` | 1024 | 512 | High quality |
| `all-minilm` | 384 | 512 | Lightweight |

### How to switch models
Change two lines in `.env` (no code changes needed):
```
EMBEDDING_PROVIDER=huggingface          # or: ollama
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5  # any model from the table above
```
**Important**: after changing the model, delete `faiss_index/` and re-ingest your documents. Different models produce vectors of different dimensions — mixing them causes crashes.

### Key insight
The embedding model is **the most important quality lever in a RAG system**. A better retrieval of fewer chunks beats poor retrieval of many chunks every time.

---

## 3. Text Chunking

### Why chunking is necessary
LLMs have a context window limit. You cannot embed an entire 200-page PDF as one unit — the embedding model also has a 256-token limit, and you want fine-grained retrieval rather than always returning entire chapters.

### The strategy in this project: `RecursiveCharacterTextSplitter`
Tries to split on paragraph breaks first (`\n\n`), then line breaks (`\n`), then spaces, then characters — in that priority order. This preserves semantic coherence better than splitting at a fixed character count.

```
chunk_size    = 1000  chars  (~200 tokens)
chunk_overlap = 200   chars  (~40 tokens)
```

### Why overlap matters
Without overlap, a sentence that straddles a chunk boundary is split, losing context. Overlap ensures each chunk is self-contained.

```
... [chunk 1 content] ... the critical result is →
→ that the model achieves 2.4x speedup [chunk 2 content] ...
```
Without overlap, no single chunk contains both halves of that key sentence.

### Chunking strategies comparison
| Strategy | Pros | Cons |
|----------|------|------|
| `RecursiveCharacterTextSplitter` (this project) | Fast, no model needed, preserves paragraphs | Ignores semantics |
| `SemanticChunker` (LangChain) | Splits on topic boundaries | Requires embedding model, slower |
| Fixed-size (e.g. 512 tokens) | Simple | Breaks mid-sentence |
| Document structure (headings, sections) | Best for structured docs | Requires custom parsing |

### Rules of thumb
- **chunk_size**: match your embedding model's token limit (~200 tokens for MiniLM)
- **chunk_overlap**: 10–20% of chunk_size
- Short chunks → too little context per chunk → LLM struggles
- Long chunks → retrieval is too coarse → irrelevant text injected

---

## 4. Vector Stores (FAISS)

### What it is
A vector store is a database optimised for **nearest-neighbour search** over high-dimensional vectors. Given a query vector, it returns the k most similar stored vectors in milliseconds.

### Why FAISS for this project
FAISS (Facebook AI Similarity Search) is an in-process library — no server, no network, no cost. The entire index lives in a file on disk. Perfect for prototypes, local tools, and small-to-medium datasets (< 1M vectors).

### How FAISS stores and searches
FAISS builds an index from all chunk vectors. For small datasets it uses a flat (brute-force) index — exact nearest-neighbour search. For large datasets you would use IVF (inverted file) or HNSW indexes for approximate search.

### When to graduate from FAISS
| Scenario | Switch to |
|----------|-----------|
| Multiple users, concurrent writes | Chroma (local), Qdrant, Weaviate |
| Cloud deployment, managed infra | Pinecone, Azure AI Search, pgvector |
| Metadata filtering at scale | Qdrant, Weaviate, pgvector |
| > 1M vectors | Any ANN (approximate nearest-neighbour) store |

---

## 5. Maximal Marginal Relevance (MMR)

### The problem MMR solves
Pure similarity search often returns **near-duplicate chunks** — the top-4 results might all be from the same paragraph of the document, just slightly overlapping. The LLM sees the same information repeated and misses the rest of the document.

### What MMR does
MMR balances two objectives:
1. **Relevance** — each selected chunk should be similar to the query
2. **Diversity** — each selected chunk should be dissimilar from already-selected chunks

```
MMR(chunk) = λ · sim(query, chunk) - (1-λ) · max_sim(chunk, selected_chunks)
```

`λ` controls the relevance/diversity trade-off (default 0.5 in LangChain).

### How this project uses MMR
```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 24},
)
```

- `fetch_k=24` — fetch 24 candidates by similarity
- `k=6` — re-rank those 24 using MMR, return best 6

### When to use MMR vs similarity
| Use similarity when | Use MMR when |
|--------------------|--------------|
| Specific factual questions | Broad or exploratory questions |
| Short documents, few chunks | Long documents, many similar passages |
| Speed is critical | Answer quality matters more |

---

## 6. Conversation Memory

### The problem
A vanilla LLM call is **stateless** — each call is independent. If you ask "what is the thesis about?" and then "who wrote it?", the second question provides no context about "it" referring to the thesis.

### How LangChain handles it: `RunnableWithMessageHistory`

```python
RunnableWithMessageHistory(
    chain,
    get_session_history,        # returns ChatMessageHistory for a session
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
```

Every call:
1. Loads the stored `ChatMessageHistory` for the session
2. Appends it to the prompt as `chat_history`
3. After the LLM responds, appends the new (human, AI) turn to the history

### Memory types comparison
| Type | Stores | Grows with | Use case |
|------|--------|-----------|----------|
| `ChatMessageHistory` (this project) | Full transcript | Linearly | Short sessions |
| `ConversationSummaryMemory` | Compressed summary | Slowly | Long sessions |
| `ConversationBufferWindowMemory` | Last N turns | Bounded | When cost matters |
| External DB (Redis, DynamoDB) | Full transcript | Linearly | Multi-user, persistent |

### The context window problem
Storing the full transcript means the prompt grows with every turn. For long conversations you will eventually hit the LLM's context limit. Production systems typically summarise or window the history.

---

## 7. History-Aware Retrieval

### The problem
If a user asks *"explain more about the second one"*, FAISS cannot retrieve relevant chunks — "the second one" has no semantic meaning without the conversation history.

### The solution: `create_history_aware_retriever`
Before retrieving chunks, a preliminary LLM call rewrites the question into a standalone version:

```
Turn 1: "What are the main algorithms?"
Turn 2: "Explain the second one"

                    ↓ LLM rewrite ↓

Standalone query: "Explain the push-relabel algorithm"
```

FAISS then retrieves chunks about push-relabel — which answers the question correctly.

### Cost implication
Every query now makes **two LLM calls**: one to rewrite the question, one to answer it. If the conversation history is empty (first turn), the rewriter simply returns the original question unchanged — no semantic change, but the API call still happens.

### Optimisation: skip rewrite on first turn
You can conditionally skip the history-aware retriever when `chat_history` is empty:
```python
if not chat_history:
    return retriever.invoke(query)
else:
    return history_aware_retriever.invoke(...)
```
This saves one LLM call per session's first question.

---

## 8. LangChain LCEL

### What it is
LangChain Expression Language (LCEL) is a functional composition syntax using the `|` (pipe) operator, similar to Unix pipes. Each component is a `Runnable` with `.invoke()`, `.stream()`, and `.batch()` methods.

### How it looks
```python
chain = retriever | format_docs | prompt | llm | StrOutputParser()
```

Data flows left to right. Each component transforms its input and passes the result to the next.

### Why LCEL over the old Chain classes
| Old chains | LCEL |
|-----------|------|
| `ConversationalRetrievalChain` | `create_retrieval_chain` |
| Black-box, hard to customise | Fully composable, each step is inspectable |
| No streaming support | Built-in streaming at every step |
| Locked into LangChain abstractions | Mix with plain Python functions |

### Key higher-level helpers used in this project
| Helper | What it wires |
|--------|--------------|
| `create_history_aware_retriever` | LLM + retriever + contextualisation prompt |
| `create_stuff_documents_chain` | LLM + QA prompt + document formatter |
| `create_retrieval_chain` | history-aware retriever + stuff-documents chain |

---

## 9. LLM Temperature

### What it controls
Temperature (`0.0` to `2.0`) controls how deterministic vs creative the LLM is:

- **0.0** — always picks the most probable next token (fully deterministic)
- **0.7** — balanced creativity and coherence
- **1.0+** — increasingly random, often incoherent

### Why this project uses `0.0`
For question answering grounded in documents you want **deterministic, factual answers**. At temperature 0.0 the same question always returns the same answer, which makes debugging predictable.

### When to increase temperature
- Creative writing or summarisation where variety is desirable
- Brainstorming / ideation tasks
- Generating diverse test cases

---

## 10. Local vs API Embeddings

### Trade-offs at a glance

| Dimension | Local HuggingFace | Local Ollama | API (`text-embedding-3-small`) |
|-----------|-------------------|--------------|-------------------------------|
| Cost | Free | Free | ~$0.02 per 1M tokens |
| Privacy | Data never leaves machine | Data never leaves machine | Data sent to OpenAI |
| Quality | Good → Near-API (model-dependent) | Good → Near-API | Better baseline |
| Latency | Fast (CPU) | Fast (local HTTP) | Network round-trip |
| Setup | Auto-download (pip) | `ollama serve` + `ollama pull` | API key only |
| Rate limits | None | None | Yes |
| Best model | `BAAI/bge-large-en-v1.5` | `mxbai-embed-large` | `text-embedding-3-large` |

### This project's choice
Local embeddings (HuggingFace by default) are used deliberately:
1. **No API cost** — embedding a 100-page PDF at chunk_size=1000 produces ~300 chunks; zero cost locally
2. **Privacy** — thesis or proprietary documents shouldn't leave your machine
3. **No rate limits** — bulk ingestion completes in seconds, not minutes of throttled API calls
4. **Configurable** — swap models in `.env` without touching any code

### When to switch to API embeddings
Switch when retrieval quality is demonstrably insufficient (answers miss relevant sections)
and the document content is not sensitive. OpenAI's `text-embedding-3-large` (3072 dims)
outperforms all local models on MTEB benchmarks, especially for nuanced technical queries.

### Model quality ladder (free → paid)
```
Fastest / smallest
  all-MiniLM-L6-v2      (384 dim, 256 tok) — original default
  BAAI/bge-small-en-v1.5 (384 dim, 512 tok) — current default ← best starting point
  BAAI/bge-base-en-v1.5  (768 dim, 512 tok) — good for most production needs
  all-mpnet-base-v2      (768 dim, 514 tok) — strong general baseline
  nomic-embed-text        (768 dim, 8192 tok) — best for long docs
  BAAI/bge-large-en-v1.5 (1024 dim, 512 tok) — near-API quality (free)
  text-embedding-3-small  (1536 dim)          — OpenAI API
  text-embedding-3-large  (3072 dim)          — OpenAI API, best available
Highest quality / cost
```
