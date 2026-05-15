# RAG Experiment Lab

> **Part of the [AI RAG Portfolio](../../README.md)** — Project 02

A **parameter-tuning and evaluation workbench** for RAG systems. Upload documents, tune every parameter at runtime (embedding model, chunk size, search strategy, reranker, LLM, temperature, prompt template), then evaluate and compare configurations with LLM-as-judge scoring.

## Architecture

```
One or more PDF files
   │
   ▼
PyPDFLoader (per PDF)                ← LangChain document loader
   │
   ▼
RecursiveCharacterTextSplitter       ← runtime-configurable chunk size & overlap
   │
   ▼
Embedding Model (selectable)         ← 6 local HuggingFace models to compare
   │
   ▼
FAISS Vector Store                   ← separate index per embedding model
   │
   ▼
Configurable Retriever               ← similarity / MMR / threshold search
   │
   ▼
Cross-Encoder Reranker (optional)    ← re-scores chunks for precision
   │
   ▼
Prompt Template (selectable)         ← strict / balanced / concise / detailed
   │
   ▼
LLM (selectable model & temperature) ← GPT-4o-mini / GPT-4o / GPT-4-turbo
   │
   ▼
LLM-as-Judge Evaluator              ← context relevance / faithfulness / answer relevance
   │
   ▼
Experiment Storage                   ← JSON-based, compare across runs
```

## What You Can Tune

| Parameter         | Options / Range                                    | Where It Affects         |
|-------------------|----------------------------------------------------|--------------------------|
| Embedding model   | bge-small, bge-base, bge-large, nomic, mpnet, MiniLM | Retrieval quality        |
| Chunk size        | 200–3000 chars                                     | Context granularity      |
| Chunk overlap     | 0–500 chars                                        | Boundary coverage        |
| Search strategy   | similarity, MMR, score threshold                   | Result diversity/precision|
| Top-K             | 1–20 chunks                                        | Context breadth          |
| MMR lambda        | 0.0–1.0                                            | Diversity vs relevance   |
| Score threshold   | 0.0–1.0                                            | Minimum quality filter    |
| Reranker          | on/off + top-N                                     | Precision after retrieval|
| LLM model         | gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo   | Answer quality & cost    |
| Temperature       | 0.0–1.0                                            | Creativity vs accuracy   |
| Prompt template   | strict, balanced, concise, detailed                | Answer style & hallucination|

## Evaluation Metrics

| Metric             | What It Measures                       | Low Score Means                    |
|--------------------|----------------------------------------|------------------------------------|
| Context Relevance  | Are retrieved chunks relevant to the Q?| Retrieval problem (fix embeddings) |
| Faithfulness       | Is the answer grounded in context?     | Hallucination (fix prompt/temp)    |
| Answer Relevance   | Does the answer address the question?  | Generation problem (fix LLM/prompt)|

## Tech Stack

| Component     | Library / Tool                               |
|---------------|----------------------------------------------|
| Framework     | LangChain 0.3                                |
| PDF Loader    | PyPDFLoader (pypdf)                          |
| Embeddings    | 6 models via sentence-transformers           |
| Vector DB     | FAISS (local, separate index per model)      |
| Reranker      | cross-encoder/ms-marco-MiniLM-L-6-v2        |
| LLM           | OpenAI GPT-4o-mini / GPT-4o / Azure OpenAI  |
| Evaluation    | LLM-as-judge (context rel, faithfulness, AR) |
| Web UI        | Streamlit (tabs: Chat, Evaluate, Compare)    |
| Config        | python-dotenv                                |

## Quick Start

### 1. Clone and navigate

```bash
git clone https://github.com/YOUR_USERNAME/ai-rag-portfolio.git
cd ai-rag-portfolio/projects/02_rag_experiment_lab

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Use the app

1. **Upload PDFs** in the sidebar
2. **Adjust parameters** using the sidebar controls
3. **Click "Index Documents"** to build the vector store
4. **Chat tab** — ask questions and see answers with source citations
5. **Evaluate tab** — run a test set and get LLM-as-judge scores
6. **Compare tab** — select multiple experiments for side-by-side comparison

### 5. Run a systematic experiment

1. Set baseline parameters, run evaluation → saves as experiment
2. Change ONE parameter (e.g., embedding model), re-index, run evaluation
3. Go to Compare tab, select both experiments
4. See which configuration scores higher

## Project Structure

```
02_rag_experiment_lab/
├── app.py                    # Streamlit UI (3 tabs: Chat, Evaluate, Compare)
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── ARCHITECTURE.md           # Design decisions and data flow
├── CONCEPTS.md               # New concepts introduced in this project
├── EXERCISES.md              # Guided experiments to deepen understanding
├── README.md                 # This file
├── src/
│   ├── __init__.py
│   ├── config.py             # Centralised environment config
│   ├── embeddings.py         # Embedding model catalogue and factory
│   ├── document_loader.py    # PDF loading with configurable chunking
│   ├── vector_store.py       # FAISS build/load, per-model indexes
│   ├── retriever.py          # Configurable retriever (similarity/MMR/threshold)
│   ├── reranker.py           # Cross-encoder reranking
│   ├── rag_chain.py          # Conversational RAG + stateless ask()
│   ├── evaluator.py          # LLM-as-judge scoring
│   └── experiment.py         # Save/load/compare experiment results
├── test_sets/
│   └── sample_questions.json # Example test set
├── experiments/              # Auto-created: saved experiment results
├── faiss_index/              # Auto-created: vector store data
└── docs/                     # Place PDFs here (optional)
```

## Key Differences from Project 01

| Aspect           | Project 01                    | Project 02                          |
|------------------|-------------------------------|-------------------------------------|
| Parameters       | Fixed at startup              | Runtime-configurable via UI         |
| Embedding models | 1 selectable                  | 6 in catalogue with metadata        |
| Search types     | MMR only                      | Similarity, MMR, score threshold    |
| Reranking        | None                          | Cross-encoder with toggle           |
| Prompt templates | 1 fixed prompt                | 4 selectable prompts                |
| Evaluation       | Manual reading                | Automated LLM-as-judge scoring      |
| Experiments      | None                          | Save, load, and compare configs     |
| Goal             | Build a working RAG app       | Understand WHY each parameter matters|
