# Concepts — RAG Experiment Lab

New concepts introduced in Project 02, building on the foundations from
Project 01 and the [shared CONCEPTS.md](../../CONCEPTS.md).

---

## 1. RAG Evaluation — Why It Matters

In Project 01, you could only judge answer quality by reading it yourself.
This works for a demo but doesn't scale — you can't manually check 100
answers across 10 different configurations.

**Systematic evaluation** means:
- Define a test set of questions (preferably with expected answers)
- Run each question through the RAG pipeline
- Score each answer automatically using metrics
- Compare configurations using aggregate scores

This is how production RAG systems are built. Without evaluation, you're
tuning parameters based on gut feeling.

---

## 2. LLM-as-Judge

### What it is

Using an LLM (the "judge") to evaluate another LLM's output. The judge
reads the question, context, and answer, then scores qualities like
faithfulness and relevance.

### Why it works

Evaluation is easier than generation. The judge LLM doesn't need to create
new content — it only needs to verify claims against evidence. This is a
simpler task that even smaller models do reasonably well.

### How it works in this project

```
Score = Judge_LLM(
    system_prompt = "Rate the faithfulness from 0.0 to 1.0...",
    user_prompt   = f"Question: {q}\nAnswer: {a}\nContext: {c}"
) → 0.85
```

Each metric uses a separate system prompt designed to measure one specific
quality. The judge returns a single float.

### Limitations

- The judge can be wrong (especially on domain-specific nuance)
- Different judge models give different scores (GPT-4o > GPT-4o-mini)
- Scores are relative, not absolute — compare configurations, not numbers
- Always spot-check a few LLM-judge scores against your own assessment

### Research

- "Judging LLM-as-a-Judge" (2023) — analysis of when LLM judges fail
- LMSYS Chatbot Arena — uses LLM-as-judge at scale for model comparison
- RAGAS framework — popular library implementing similar metrics

---

## 3. Evaluation Metrics Explained

### Context Relevance (retrieval quality)

**What:** Are the retrieved chunks actually relevant to the question?

**How:** Each chunk is scored individually by the judge, then averaged.

**When low:** Your retriever is fetching irrelevant chunks. Fix by:
- Changing embedding model (maybe the current one doesn't understand your domain)
- Adjusting chunk size (too large = noise; too small = missing context)
- Switching search strategy (MMR for diverse results, similarity for focused)
- Enabling reranking to filter out weak candidates

### Faithfulness (grounding quality)

**What:** Is every claim in the answer supported by the retrieved context?

**How:** The judge checks if the answer contains information not in the context.

**When low:** The LLM is hallucinating. Fix by:
- Using the "strict" prompt template
- Lowering temperature (0.0 is most deterministic)
- Using a more instruction-following model (GPT-4o > GPT-3.5-turbo)
- Enabling reranking to provide better context

### Answer Relevance (generation quality)

**What:** Does the answer actually address the user's question?

**How:** The judge reads only the question and answer (no context).

**When low:** The answer is off-topic even if it's factually correct. Fix by:
- Checking if the relevant information is even in the documents
- Using a different prompt template
- Using a better LLM model

### Diagnostic flowchart

```
Low quality answer?
  │
  ├── Low context relevance? → FIX RETRIEVAL
  │     ├── Try different embedding model
  │     ├── Adjust chunk size/overlap
  │     └── Change search strategy
  │
  ├── Low faithfulness? → FIX GROUNDING
  │     ├── Use stricter prompt
  │     ├── Lower temperature
  │     └── Enable reranker
  │
  └── Low answer relevance? → FIX GENERATION
        ├── Use better LLM
        ├── Try different prompt template
        └── Check if docs contain the answer
```

---

## 4. Cross-Encoder Reranking

### Bi-encoder vs. Cross-encoder

**Bi-encoder** (what FAISS uses):
```
query  ──► Encoder ──► vector_q ─┐
                                  ├──► cosine_sim(vector_q, vector_d) = score
doc    ──► Encoder ──► vector_d ─┘
```
Fast because documents are pre-encoded. But the vectors are computed
independently — the encoder never sees both texts together.

**Cross-encoder** (what the reranker uses):
```
[query, doc] ──► Encoder ──► score
```
Slower but more accurate because it sees the full interaction between
the question and document text.

### Two-stage retrieval

1. **Recall stage:** FAISS retrieves top-K candidates cheaply (bi-encoder)
2. **Precision stage:** Cross-encoder re-scores top-K, keeps top-N (N < K)

The bi-encoder handles the hard part (searching millions of chunks) while
the cross-encoder handles the nuanced part (which of these K candidates
is actually the best match?).

### When to use

| Scenario                        | Use reranker?  |
|---------------------------------|----------------|
| Quick prototype                 | No             |
| top_k ≤ 3                      | No (not enough candidates) |
| top_k ≥ 6 + precision matters  | Yes            |
| Domain-specific content         | Yes (helps with jargon) |
| Real-time chat (latency-sensitive) | Maybe (adds 100-500ms) |

---

## 5. Search Strategies Compared

### Similarity Search
The simplest: return the K chunks with highest cosine similarity to the query.

**Pros:** Fast, intuitive, deterministic.
**Cons:** May return near-duplicate chunks from the same section.

### MMR (Maximal Marginal Relevance)
Balances relevance and diversity using a lambda parameter.

```
MMR_score = λ × sim(query, doc) - (1-λ) × max(sim(doc, already_selected))
```

- λ = 1.0 → pure similarity (no diversity)
- λ = 0.5 → balanced (default)
- λ = 0.0 → maximum diversity (may sacrifice relevance)

**Pros:** Avoids redundant chunks, covers more topics.
**Cons:** Slightly slower (iterative selection), may miss the second-best
match if it's too similar to the first.

### Similarity with Score Threshold
Only returns chunks above a minimum similarity score.

**Pros:** Prevents low-quality chunks from polluting the context.
**Cons:** May return fewer than K chunks (or zero) if no chunk meets the
threshold. Requires tuning the threshold value per embedding model.

---

## 6. Prompt Engineering for RAG

The system prompt is one of the highest-leverage parameters. This project
includes four templates to illustrate different strategies:

### Strict
```
"Answer only from the provided context. Say 'I don't know' otherwise."
```
- Highest faithfulness, lowest hallucination
- May refuse answerable questions if phrased differently than the context
- Best for: legal, medical, compliance — where accuracy > coverage

### Balanced
```
"Use context first. Supplement with general knowledge if needed."
```
- Good faithfulness with graceful degradation
- Must handle the "clearly distinguish" instruction
- Best for: research, learning — where helpful > paranoid

### Concise
```
"Bullet points. Under 150 words."
```
- Forces structured output
- May omit nuance or caveats
- Best for: dashboards, summaries — where brevity > depth

### Detailed
```
"Thorough, comprehensive, include all relevant citations."
```
- Most complete answers
- Uses more tokens (higher cost, slower)
- Best for: research reports, due diligence — where depth > speed

---

## 7. Experiment Design Best Practices

1. **Change one variable at a time.** If you change both embedding model
   and chunk size, you can't tell which change caused the score difference.

2. **Use the same test set.** Always compare experiments run on the same
   questions, otherwise the comparison is meaningless.

3. **Run enough questions.** 5 questions might show noise; 20+ gives
   stable averages. The sample test set has 5 — extend it for your documents.

4. **Check individual questions.** Averages hide outliers. If one question
   scores 0.2 while others score 0.9, investigate that specific failure.

5. **Document your findings.** The Compare tab shows numbers; write down
   your interpretation (e.g., "bge-base scores 10% higher on context
   relevance but takes 3x longer to index — worth it for production").
