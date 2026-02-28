# Exercises & Experiments Guide

Hands-on challenges to deepen your understanding of AI engineering.
Each exercise has a clear goal, hints, and a "what to observe" section
so you know what you are learning from the change.

Work through them in order — Level 1 only touches config, Level 2 requires
code changes, Level 3 adds new features.

---

## Level 1 — Configuration Experiments (no code changes needed)

### Exercise 1.1 — Chunk size impact on retrieval quality

**Goal:** Understand the relationship between chunk size and answer quality.

**Steps:**
1. In `.env`, set `CHUNK_SIZE=300` and `CHUNK_OVERLAP=50`
2. Re-ingest your PDF: `python main.py --pdf docs/your.pdf --ingest-only`
3. Ask several questions and note the answers
4. Repeat with `CHUNK_SIZE=2000`, `CHUNK_OVERLAP=400`

**What to observe:**
- Small chunks (300): answers may feel "fragmented" because each retrieved
  chunk lacks surrounding context
- Large chunks (2000): answers may feel bloated or include irrelevant text
  from the same page; also hits the embedding model's token limit
- Find the sweet spot for your specific document type

**Concept link:** See CONCEPTS.md §3 (Text Chunking)

---

### Exercise 1.2 — Temperature and answer style

**Goal:** See how temperature affects factual vs generative tasks.

**Steps:**
1. In `.env`, set `LLM_TEMPERATURE=0.0`; ask "summarise Chapter 2"
2. Set `LLM_TEMPERATURE=0.9`; ask the same question
3. Repeat 3 times at each temperature

**What to observe:**
- At 0.0: identical answers every run; conservative, fact-based
- At 0.9: varied phrasing each run; sometimes more fluent, sometimes hallucinates

**Concept link:** See CONCEPTS.md §9 (LLM Temperature)

---

### Exercise 1.3 — Top-k and context breadth

**Goal:** Feel the effect of retrieving more vs fewer chunks.

**Steps:**
1. Run with `--top-k 2` and ask a broad question ("what topics does this cover?")
2. Run with `--top-k 10` and ask the same question

**What to observe:**
- `top-k 2`: fast, but may miss relevant sections
- `top-k 10`: more comprehensive but slower (larger prompt, higher cost)
- The "lost-in-the-middle" effect: very large contexts cause the LLM to
  ignore chunks placed in the middle of the context

---

## Level 2 — Code Modifications

### Exercise 2.1 — Swap the embedding model

**Goal:** Measure quality vs cost trade-off between local and API embeddings.

**File:** `src/vector_store.py` and `src/config.py`

**Steps:**
1. Add `langchain-openai` embedding support in `vector_store.py`:
   ```python
   from langchain_openai import OpenAIEmbeddings

   def _get_embeddings():
       if config.EMBEDDING_PROVIDER == "openai":
           return OpenAIEmbeddings(model="text-embedding-3-small")
       return HuggingFaceEmbeddings(...)
   ```
2. Add `EMBEDDING_PROVIDER=openai` to `.env`
3. Re-ingest and compare answers

**What to observe:**
- Does retrieval quality improve for ambiguous questions?
- How much slower/faster is ingestion?
- Note: you cannot mix indexes built with different embedding models —
  you must re-ingest when switching

**Concept link:** See CONCEPTS.md §10 (Local vs API Embeddings)

---

### Exercise 2.2 — Switch from "stuff" to "map-reduce"

**Goal:** Handle very long documents where context overflows the LLM window.

**File:** `src/rag_chain.py`

**Steps:**
1. Replace `create_stuff_documents_chain` with LangChain's map-reduce chain:
   ```python
   from langchain.chains.combine_documents import create_map_rerank_documents_chain
   ```
   Or implement manually:
   - Step 1: for each retrieved chunk, ask the LLM "answer the question from
     this chunk alone; say N/A if not relevant"
   - Step 2: pass all individual answers to the LLM for a final synthesis

**What to observe:**
- Map-reduce is slower (N+1 LLM calls instead of 1)
- But handles cases where the answer spans multiple sections that don't
  fit in one prompt
- For most PDF RAG use cases, stuff is sufficient with k≤8

---

### Exercise 2.3 — Add similarity score threshold

**Goal:** Prevent retrieval of irrelevant chunks when the question is
completely out of scope.

**File:** `src/rag_chain.py`

**Steps:**
1. Use `score_threshold` in the retriever:
   ```python
   retriever = vector_store.as_retriever(
       search_type="similarity_score_threshold",
       search_kwargs={"score_threshold": 0.4, "k": 6},
   )
   ```
2. Ask an off-topic question (e.g. "what is the capital of France?" to a thesis)

**What to observe:**
- Without threshold: retrieves the 6 most similar chunks regardless of relevance
- With threshold: returns fewer (or zero) chunks if nothing is relevant enough
- The LLM's "I don't know" response is only as good as the retrieval guardrail

**Note:** threshold works with cosine similarity — requires `normalize_embeddings=True`
(already set in this project).

---

### Exercise 2.4 — Add streaming output

**Goal:** Stream the LLM's response token-by-token instead of waiting for the full answer.

**File:** `main.py` and `src/rag_chain.py`

**Steps:**
1. In `chat_loop`, replace `chain.invoke(...)` with `chain.stream(...)`:
   ```python
   for chunk in chain.stream(
       {"input": question},
       config={"configurable": {"session_id": _SESSION_ID}},
   ):
       if "answer" in chunk:
           print(chunk["answer"], end="", flush=True)
   print()
   ```
2. Notice the answer appears progressively

**What to observe:**
- Streaming massively improves perceived responsiveness
- The LLM generates the same tokens either way — streaming just delivers
  them incrementally
- LCEL chains support streaming natively; older LangChain chains do not

---

### Exercise 2.5 — Log retrieved chunks (debug mode)

**Goal:** Understand exactly what context the LLM is seeing before generating an answer.

**File:** `main.py`

**Steps:**
1. After `result = chain.invoke(...)`, add:
   ```python
   if args.debug:
       print("\n[Debug] Retrieved context:")
       for doc in result.get("context", []):
           meta = doc.metadata
           print(f"  [{meta.get('filename')}:p{meta.get('page_display')}] "
                 f"{doc.page_content[:100]}...")
   ```
2. Add `--debug` flag to `parse_args()`

**What to observe:**
- Which chunks are actually retrieved for each question?
- Are they relevant? If not, is the problem in chunking, embedding, or MMR tuning?
- This is the most valuable debugging tool in any RAG system

---

## Level 3 — New Features

### Exercise 3.1 — Add a reranker

**Goal:** Use a cross-encoder to re-order retrieved chunks by true relevance
before passing them to the LLM.

**Background:** Bi-encoders (used for retrieval) encode query and document
independently — fast but approximate. Cross-encoders see both together —
slower but much more accurate at scoring relevance.

**Steps:**
1. Install: `pip install sentence-transformers`
2. After retrieval, add a reranking step:
   ```python
   from sentence_transformers import CrossEncoder
   reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

   def rerank(question, docs, top_n=4):
       pairs = [(question, doc.page_content) for doc in docs]
       scores = reranker.predict(pairs)
       ranked = sorted(zip(scores, docs), reverse=True)
       return [doc for _, doc in ranked[:top_n]]
   ```
3. Call `rerank()` after the retriever and before the QA chain

**What to observe:**
- Reranking often dramatically improves answers on ambiguous questions
- It adds ~200ms latency per query (cross-encoder runs locally on CPU)
- This is the RAG improvement with the highest quality ROI

**Concept link:** In CONCEPTS.md §5, this adds a third stage after MMR retrieval

---

### Exercise 3.2 — Add a FastAPI REST endpoint

**Goal:** Expose the RAG pipeline as an HTTP API so other applications can query it.

**Steps:**
1. Install: `pip install fastapi uvicorn`
2. Create `api.py`:
   ```python
   from fastapi import FastAPI
   from pydantic import BaseModel
   from src.vector_store import load_vector_store
   from src.rag_chain import build_rag_chain

   app = FastAPI()
   vector_store = load_vector_store()
   chain = build_rag_chain(vector_store)

   class QueryRequest(BaseModel):
       question: str
       session_id: str = "default"

   @app.post("/query")
   def query(req: QueryRequest):
       result = chain.invoke(
           {"input": req.question},
           config={"configurable": {"session_id": req.session_id}},
       )
       return {"answer": result["answer"]}
   ```
3. Run: `uvicorn api:app --reload`
4. Test: `curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "what is this about?"}'`

**What to observe:**
- The chain is stateful per `session_id` — multiple "users" can have independent conversations
- `session_id` in the URL/body maps directly to LangChain's `RunnableWithMessageHistory`
- This is the foundation of the FastAPI roadmap item

---

### Exercise 3.3 — Add a Streamlit UI

**Goal:** Build a browser-based chat interface.

**Steps:**
1. Install: `pip install streamlit`
2. Create `app.py`:
   ```python
   import streamlit as st
   from src.vector_store import load_vector_store
   from src.rag_chain import build_rag_chain

   st.title("PDF RAG Assistant")

   if "chain" not in st.session_state:
       vs = load_vector_store()
       st.session_state.chain = build_rag_chain(vs)

   if "messages" not in st.session_state:
       st.session_state.messages = []

   for msg in st.session_state.messages:
       st.chat_message(msg["role"]).write(msg["content"])

   if prompt := st.chat_input("Ask a question..."):
       st.session_state.messages.append({"role": "user", "content": prompt})
       st.chat_message("user").write(prompt)

       result = st.session_state.chain.invoke(
           {"input": prompt},
           config={"configurable": {"session_id": "streamlit"}},
       )
       answer = result["answer"]
       st.session_state.messages.append({"role": "assistant", "content": answer})
       st.chat_message("assistant").write(answer)
   ```
3. Run: `streamlit run app.py`

**What to observe:**
- Streamlit maintains its own session state, which maps naturally to LangChain's session memory
- The chain is initialised once and reused across turns (critical for performance)
- This completes the Streamlit web UI roadmap item

---

### Exercise 3.4 — Hybrid search (BM25 + dense retrieval)

**Goal:** Combine keyword (BM25) and semantic (dense vector) search so exact
terminology matches are not missed by the embedding model.

**Background:** Dense retrieval is great for semantic similarity but can miss
exact keyword matches (e.g. algorithm names, acronyms, version numbers).
BM25 is the reverse. Combining both (hybrid search) gives the best of both worlds.

**Steps:**
1. Install: `pip install rank_bm25`
2. Build a BM25 retriever over your chunks using `BM25Retriever` from `langchain_community`
3. Combine with the FAISS retriever using `EnsembleRetriever`:
   ```python
   from langchain_community.retrievers import BM25Retriever
   from langchain.retrievers import EnsembleRetriever

   bm25 = BM25Retriever.from_documents(chunks, k=6)
   faiss_ret = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 6})

   hybrid = EnsembleRetriever(
       retrievers=[bm25, faiss_ret],
       weights=[0.4, 0.6],   # tune these for your use case
   )
   ```
4. Replace the retriever in `build_rag_chain`

**What to observe:**
- Ask a question containing an exact algorithm name (e.g. "Edmonds-Karp")
- Compare answers with dense-only vs hybrid retrieval
- Hybrid almost always wins for technical documents with specialised vocabulary

---

## Reflection Questions

After completing exercises, think about:

1. **What breaks first as document size grows?** (hint: chunking time? embedding time? context window?)
2. **Which single change had the biggest quality improvement for your document?**
3. **If you had to serve 100 users simultaneously, what would you change?** (hint: FAISS is not thread-safe for concurrent writes; stateful memory needs a shared store like Redis)
4. **What would you add to make hallucination risk even lower?** (hint: score thresholds, citation grounding, separate fact-checking LLM call)
