# Exercises — RAG Experiment Lab

Guided experiments to build intuition about how each RAG parameter
affects output quality. Each exercise changes ONE variable and measures
the impact.

---

## Prerequisites

1. Index at least one PDF (a 10-30 page academic paper or report works well)
2. Create a test set with 5-10 questions that can be answered from the PDF
3. Save it as `test_sets/my_questions.json`

---

## Exercise 1: Embedding Model Showdown

**Goal:** Discover which embedding model gives the best retrieval quality
for your specific documents.

**Steps:**
1. Use default parameters (strict prompt, chunk_size=1000, top_k=6, MMR)
2. Select `bge-small-en-v1.5` → Index → Run evaluation → Note scores
3. Select `bge-base-en-v1.5` → Re-index → Run evaluation → Note scores
4. Select `bge-large-en-v1.5` → Re-index → Run evaluation → Note scores
5. Go to Compare tab and select all three

**What to look for:**
- Does context relevance increase with model size?
- Is the improvement worth the extra indexing time?
- Check individual questions — are some models better for certain question types?

**Expected learning:**
Larger models generally score higher on context relevance, but the
improvement from base→large is smaller than small→base. For most use cases,
bge-base is the sweet spot.

---

## Exercise 2: Chunk Size Tuning

**Goal:** Find the chunk size that balances context sufficiency with precision.

**Steps:**
1. Fix embedding model to bge-small, use strict prompt, top_k=6
2. Set chunk_size=300, overlap=50 → Re-index → Evaluate
3. Set chunk_size=600, overlap=100 → Re-index → Evaluate
4. Set chunk_size=1000, overlap=200 → Re-index → Evaluate
5. Set chunk_size=2000, overlap=400 → Re-index → Evaluate
6. Compare all four experiments

**What to look for:**
- Small chunks: high context relevance but low answer relevance (not enough info)
- Large chunks: lower context relevance (more noise) but higher answer relevance
- There's usually a sweet spot where faithfulness peaks

**Expected learning:**
500-1000 chars works best for most documents. Very small chunks (<300) lose
important context. Very large chunks (>2000) dilute relevance.

---

## Exercise 3: Temperature Effect on Faithfulness

**Goal:** Demonstrate why low temperatures are preferred for RAG.

**Steps:**
1. Use strict prompt, bge-small, chunk_size=1000, top_k=6
2. Set temperature=0.0 → Evaluate
3. Set temperature=0.3 → Evaluate
4. Set temperature=0.7 → Evaluate
5. Set temperature=1.0 → Evaluate
6. Compare all four

**What to look for:**
- Faithfulness should decrease as temperature increases
- Answer relevance may stay similar or slightly improve
- At temp=1.0, watch for creative hallucinations in the answers

**Expected learning:**
For RAG, temperature 0.0-0.2 gives the most faithful answers. Higher
temperatures add creativity that works against the goal of grounded responses.

---

## Exercise 4: Prompt Template Comparison

**Goal:** See how the system prompt changes answer style AND quality scores.

**Steps:**
1. Fix all other parameters (bge-small, chunk_size=1000, top_k=6, temp=0.0)
2. Select "strict" template → Evaluate
3. Select "balanced" template → Evaluate
4. Select "concise" template → Evaluate
5. Select "detailed" template → Evaluate
6. Compare all four

**What to look for:**
- Strict: highest faithfulness, sometimes lower answer relevance
- Balanced: good on all metrics
- Concise: fast but may miss nuance
- Detailed: highest answer relevance, might have lower faithfulness

**Expected learning:**
The "right" prompt depends on your use case. Strict prevents hallucination
but may refuse valid questions. Balanced is the best general starting point.

---

## Exercise 5: Search Strategy Comparison

**Goal:** Understand when each retrieval strategy shines.

**Steps:**
1. Fix all params, use strict prompt, bge-small, top_k=6
2. Select "similarity" search → Evaluate
3. Select "mmr" search (lambda=0.5) → Evaluate
4. Select "similarity_score_threshold" (threshold=0.4) → Evaluate
5. Compare all three

**What to look for:**
- Similarity: may return near-duplicate chunks
- MMR: more diverse but might miss the second-best relevant chunk
- Threshold: may return fewer chunks (check num_chunks per question)

**Expected learning:**
MMR is usually the best default for RAG. It ensures the LLM sees diverse
information rather than 6 slightly different phrasings of the same fact.

---

## Exercise 6: Reranker Impact

**Goal:** Measure whether cross-encoder reranking improves answer quality.

**Steps:**
1. Fix all params: bge-small, chunk_size=1000, top_k=10, MMR, strict
2. Disable reranker → Evaluate
3. Enable reranker, top_n=4 → Evaluate
4. Enable reranker, top_n=6 → Evaluate
5. Compare all three

**What to look for:**
- Does faithfulness improve with reranking?
- Does context relevance improve?
- How much extra latency does reranking add?

**Expected learning:**
Reranking helps most when top_k is large (≥6) because it filters out
the weakest candidates. With top_k=3, there's not enough to rerank.

---

## Exercise 7: Top-K Sweet Spot

**Goal:** Find the optimal number of retrieved chunks.

**Steps:**
1. Fix all params: bge-small, chunk_size=1000, similarity search, strict
2. Set top_k=2 → Evaluate
3. Set top_k=4 → Evaluate
4. Set top_k=6 → Evaluate
5. Set top_k=10 → Evaluate
6. Set top_k=15 → Evaluate
7. Compare all five

**What to look for:**
- Low top_k: high context relevance (few, precise hits) but maybe low
  answer relevance (not enough info)
- High top_k: lower context relevance (more noise) but more info available
- Token usage increases with top_k (more context = higher cost)

**Expected learning:**
4-6 chunks is typically optimal. Beyond 8-10, you get diminishing returns
and the extra noise can actually hurt faithfulness.

---

## Exercise 8: LLM Model Comparison

**Goal:** Compare different LLM models for answer quality and cost.

**Steps:**
1. Fix retrieval params: bge-small, chunk_size=1000, top_k=6, MMR, strict
2. Select gpt-4o-mini → Evaluate
3. Select gpt-4o → Evaluate
4. Compare

**What to look for:**
- Does gpt-4o significantly improve faithfulness over gpt-4o-mini?
- Is the answer quality improvement worth the ~20x cost difference?
- Check latency: gpt-4o may be slower

**Expected learning:**
For well-constructed RAG pipelines with good retrieval, gpt-4o-mini
often performs nearly as well as gpt-4o — the quality bottleneck is usually
retrieval, not the LLM. This is one of the key insights of RAG engineering.

---

## Exercise 9: End-to-End Optimization

**Goal:** Combine learnings from all exercises to find the best configuration.

**Steps:**
1. Start with the winner from each previous exercise
2. Run a full evaluation with the combined "best" settings
3. Fine-tune any parameter that's not yet optimal
4. Run final evaluation and save as "optimized" experiment

**What to record:**
- Your optimal parameter set
- Final scores vs. the baseline (Exercise 1, run 1)
- Which parameter change gave the biggest improvement
- Total evaluation cost estimate

**Expected outcome:**
A documented, reproducible RAG configuration for your documents, with
evidence for why each parameter value was chosen.

---

## Creating Better Test Sets

The sample test set (`test_sets/sample_questions.json`) has generic
questions. For meaningful results, create domain-specific test sets:

1. **Read your document** and write 10-20 questions of varying difficulty
2. **Include factual questions** ("What is X?") and analytical questions
   ("Why does the author argue X?")
3. **Include unanswerable questions** (2-3 that are NOT in the document)
   to test the prompt's refusal behaviour
4. **Add expected answers** for future automated comparison

```json
[
  {
    "question": "What percentage increase was reported in Q3?",
    "expected_answer": "The report shows a 15% increase in Q3 revenue."
  },
  {
    "question": "What is the capital of Mars?",
    "expected_answer": "This question cannot be answered from the document."
  }
]
```
