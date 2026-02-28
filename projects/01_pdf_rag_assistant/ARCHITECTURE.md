# Architecture Decision Record

This document explains how data flows through the system, why each component
exists, and the trade-offs behind the design choices. Use it as a template
when building similar AI applications.

---

## System Overview

The application has two physically separate pipelines that run at different times:

```
┌─────────────────────────────────────────────────────────────────┐
│  INGESTION PIPELINE  (runs once per document set)               │
│                                                                 │
│  PDF(s) ──► Loader ──► Splitter ──► Embedder ──► FAISS Index  │
│                                                                 │
│  Output: faiss_index/ saved to disk                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  QUERY PIPELINE  (runs on every user question)                  │
│                                                                 │
│  Question ──► Rewriter ──► MMR Retriever ──► QA Chain ──► Answer│
│                 │                                    │          │
│          chat_history                         chat_history      │
│                                                                 │
│  Input: question + session memory                              │
│  Output: grounded answer with citations                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ingestion Pipeline — Step by Step

### Step 1: PDF Loading (`src/document_loader.py`)

```
PDF file
  │
  ▼
PyPDFLoader.load()
  │
  └─► List[Document]   (one Document per page)
        metadata: {source, page, filename, page_display}
```

**Design decision — PyPDFLoader vs alternatives:**
`PyPDFLoader` is simple, pure-Python, and handles most PDFs correctly.
Alternatives for harder cases:
- `UnstructuredPDFLoader` — better at tables, images, columns (heavier dependency)
- `PDFMinerLoader` — more accurate text extraction for complex layouts
- `AzureAIDocumentIntelligenceLoader` — cloud-based, best quality but costs money

**Design decision — metadata enrichment:**
`filename` and `page_display` are added at load time so every downstream chunk
carries its origin. This is what powers the `[Source: file.pdf, Page 3]` citation
in answers. Without this, attribution would be impossible without re-parsing.

---

### Step 2: Chunking (`src/document_loader.py`)

```
List[Document] (page-level)
  │
  ▼
RecursiveCharacterTextSplitter
  │
  ├─► Summary chunk (pages 1–2 concatenated, prepended)
  └─► Regular chunks (1000 chars, 200 overlap)
```

**Design decision — summary chunk:**
The splitter has no knowledge of document structure. If the abstract spans
pages 1–2, it might be split into 3 chunks. A broad question like "what is
this about?" would only retrieve one of those 3 fragments. The summary chunk
concatenates the first pages into one unit, ensuring the LLM gets a complete
high-level view for overview questions.

**Design decision — chunk_size=1000 chars:**
`all-MiniLM-L6-v2` has a 256-token limit (~1000 characters). Chunks longer
than this are silently truncated during embedding, losing the tail. Matching
chunk_size to the model's limit avoids silent information loss.

---

### Step 3: Embedding (`src/vector_store.py`)

```
List[Document] (chunk-level)
  │
  ▼
HuggingFaceEmbeddings (all-MiniLM-L6-v2)
  │
  ▼
List[np.ndarray]   (384-dimensional vectors)
```

**Design decision — local embeddings:**
OpenAI's embedding API is higher quality but sends your text to their servers.
For a thesis or proprietary document, a local model (`all-MiniLM-L6-v2`)
keeps data private and eliminates per-call API cost. The quality difference
is acceptable for most RAG use cases.

**Design decision — `normalize_embeddings=True`:**
Normalising vectors to unit length converts cosine similarity to a simple dot
product. FAISS can then use faster integer quantisation or dot-product indexes.
It also makes similarity scores consistent across queries (always in [0, 1]).

---

### Step 4: FAISS Index (`src/vector_store.py`)

```
List[np.ndarray]
  │
  ▼
FAISS.from_documents()   ← builds flat (exact) L2 index
  │
  ▼
vector_store.save_local("faiss_index/")
```

**Design decision — persist to disk:**
Embedding is the slowest step (CPU-bound, seconds to minutes for large PDFs).
Saving the index means subsequent runs skip embedding entirely. Only re-ingest
when the source documents change.

**Design decision — flat index (brute-force):**
FAISS defaults to a flat index for small datasets — this does exact nearest-
neighbour search. For > 100k chunks you would switch to an IVF or HNSW index
for approximate (but much faster) search. For typical PDF RAG workloads
(thousands of chunks), exact search is fast enough.

---

## Query Pipeline — Step by Step

### Step 1: Question Rewriting (`src/rag_chain.py`)

```
(question, chat_history)
  │
  ▼
create_history_aware_retriever(llm, retriever, contextualize_prompt)
  │
  ▼
Standalone question (e.g. "Explain the push-relabel algorithm")
```

**Design decision — separate rewrite step:**
FAISS operates on vector similarity alone — it cannot understand pronouns or
conversational references. "Explain the second one" would retrieve chunks
about "second" or "one", not the previously-mentioned algorithm. A cheap
LLM call (rewrite prompt) resolves the reference before retrieval.

**Cost implication:** First turn has empty history, so the rewriter returns
the question unchanged — but the API call still happens. If cost is critical,
you can skip the rewriter when `chat_history` is empty.

---

### Step 2: MMR Retrieval (`src/rag_chain.py`)

```
Standalone question
  │
  ▼
vector_store.as_retriever(search_type="mmr", k=6, fetch_k=24)
  │
  ├── Similarity search: fetch top-24 candidates
  └── MMR re-rank: select 6 maximally diverse & relevant chunks
```

**Design decision — MMR over similarity:**
An academic thesis has many passages discussing the same algorithm. Pure
similarity search would return 6 near-identical chunks from the same section.
MMR spreads retrieval across the document, giving the LLM a broader view.

**Design decision — fetch_k = k × 4:**
MMR needs a candidate pool to re-rank. A pool of 4× the final count (24 for
k=6) provides enough diversity without fetching so many candidates that
retrieval slows down.

---

### Step 3: Context Assembly (`src/rag_chain.py`)

```
List[Document] (6 retrieved chunks)
  │
  ▼
_DOCUMENT_PROMPT applied to each chunk:
  "[Source: {filename}, Page {page_display}]\n{page_content}"
  │
  ▼
create_stuff_documents_chain
  │
  └─► Formatted context string injected into _QA_PROMPT
```

**Design decision — "stuff" strategy:**
`create_stuff_documents_chain` concatenates all retrieved chunks into one
context string ("stuffs" them into the prompt). This is the simplest strategy
and works well when top_k is small (≤ 6–8 chunks). Alternatives:
- **Map-reduce** — summarise each chunk independently, then combine (for huge docs)
- **Refine** — iteratively update an answer by reading one chunk at a time
- **Rerank + stuff** — use a cross-encoder to reorder chunks before stuffing

---

### Step 4: LLM Generation (`src/rag_chain.py`)

```
Prompt = [system: instructions + context]
         + [chat_history messages]
         + [human: question]
  │
  ▼
ChatOpenAI / AzureChatOpenAI (gpt-4o-mini, temperature=0.0)
  │
  ▼
Answer with inline citations
```

**Design decision — gpt-4o-mini:**
Cheapest OpenAI model that reliably follows complex instructions (citation
format, "say I don't know" when context is insufficient). GPT-4o is ~10×
more expensive and not needed for document Q&A.

**Design decision — temperature=0.0:**
Document Q&A is a retrieval + extraction task, not a creative task. Deterministic
output (temperature=0) makes results reproducible and easy to debug.

---

## Separation of Concerns

Each module has one responsibility:

| Module | Responsibility | Knows about |
|--------|---------------|-------------|
| `config.py` | Load and validate env vars | Environment only |
| `document_loader.py` | PDF → chunks | Files, LangChain splitters |
| `vector_store.py` | chunks → FAISS index | Embeddings, FAISS |
| `rag_chain.py` | Build the full chain | LLM, retriever, prompts |
| `main.py` | CLI, orchestration | All of the above |

This structure means you can swap any single component without touching the
others. For example, replacing FAISS with Chroma only changes `vector_store.py`.

---

## Configuration Flow

```
.env file
  │
  ▼
src/config.py  (Config class, loaded at import time)
  │
  ├── document_loader.py  (CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL)
  ├── vector_store.py     (EMBEDDING_MODEL, FAISS_INDEX_PATH)
  └── rag_chain.py        (LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY,
                            AZURE_* settings, LLM_TEMPERATURE)
```

All settings flow through one object (`config`). This means:
- No magic strings scattered across modules
- Easy to test by monkey-patching `config` attributes
- Easy to extend: add a field in `config.py`, use it anywhere

---

## Template for Your Next RAG Project

When building a new RAG application, the component checklist is:

```
1. Loader          — how do you get raw text? (PDF, web, DB, API)
2. Chunker         — how do you split it? (recursive, semantic, structural)
3. Embedder        — which embedding model? (local vs API, dim, token limit)
4. Vector store    — where do vectors live? (FAISS, Chroma, Pinecone...)
5. Retriever       — how do you fetch chunks? (similarity, MMR, hybrid)
6. Memory          — do you need conversation history?
7. Prompt          — system instructions, context format, output format
8. LLM             — which model and provider?
9. Output parser   — how do you extract the answer? (string, JSON, tool call)
```

Every RAG project makes choices at each of these nine layers. Knowing the
trade-offs (documented in CONCEPTS.md) lets you make those choices deliberately.
