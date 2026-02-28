# AI RAG Portfolio

A growing collection of **Retrieval-Augmented Generation (RAG)** projects built to learn and demonstrate real-world AI engineering patterns. Each sub-project is self-contained, runs independently, and focuses on a different RAG use case or technique.

---

## Projects

| # | Project | Use Case | Key Techniques |
|---|---------|----------|----------------|
| 01 | [PDF RAG Assistant](projects/01_pdf_rag_assistant/) | Chat with any PDF document | Multi-PDF ingestion, conversation memory, MMR retrieval, source citations |
| 02 | [Library Assistant](projects/02_library_assistant/) | Book discovery & recommendations | Structured data RAG, CSV/JSON ingestion, metadata filtering |

---

## Repository Structure

```
ai-rag-portfolio/
├── README.md                        ← you are here (portfolio overview)
├── CONCEPTS.md                      ← shared AI engineering reference (all projects)
│
└── projects/
    ├── 01_pdf_rag_assistant/        ← Project 1: chat with PDF documents
    │   ├── app.py                   ← Streamlit web UI
    │   ├── main.py                  ← CLI entry point
    │   ├── requirements.txt
    │   ├── .env.example
    │   ├── README.md
    │   ├── ARCHITECTURE.md
    │   ├── EXERCISES.md
    │   ├── docs/                    ← drop PDFs here for CLI mode
    │   └── src/
    │       ├── config.py
    │       ├── document_loader.py
    │       ├── vector_store.py
    │       └── rag_chain.py
    │
    └── 02_library_assistant/        ← Project 2: book discovery RAG
        ├── app.py                   ← Streamlit web UI
        ├── requirements.txt
        ├── .env.example
        ├── README.md
        ├── ARCHITECTURE.md
        ├── data/                    ← book catalog CSV/JSON files
        └── src/
            ├── config.py
            ├── book_loader.py
            ├── vector_store.py
            └── rag_chain.py
```

---

## Shared Reference

**[CONCEPTS.md](CONCEPTS.md)** — covers the AI engineering concepts that apply across all projects:
embeddings, chunking, FAISS, MMR, conversation memory, history-aware retrieval, LangChain LCEL, temperature, and local vs API embeddings.

---

## How Each Project Is Structured

Every project follows the same pattern so it's easy to navigate across them:

```
project/
├── app.py           — Streamlit UI (primary interface)
├── main.py          — CLI interface (where applicable)
├── requirements.txt — isolated dependencies
├── .env.example     — environment variable template
├── README.md        — project overview + quick start
├── ARCHITECTURE.md  — design decisions and pipeline walkthrough
├── EXERCISES.md     — hands-on challenges to extend the project
└── src/
    ├── config.py          — centralised env-var configuration
    ├── *_loader.py        — data ingestion and preprocessing
    ├── vector_store.py    — FAISS build / load helpers
    └── rag_chain.py       — LangChain RAG chain with memory
```

Each project has its own `venv` and `requirements.txt` — dependencies are not shared.

---

## Getting Started

Each project is independent. Pick one, navigate into it, and follow its README:

```bash
# Example: run Project 1
cd projects/01_pdf_rag_assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # fill in OPENAI_API_KEY
streamlit run app.py
```

---

## Learning Path

Start with **01_pdf_rag_assistant** to understand the core RAG loop, then move to **02_library_assistant** to see how RAG adapts to structured data and metadata-based filtering.

Each project builds on concepts from the previous one — check `ARCHITECTURE.md` inside each project to see what new pattern it introduces.

---

## License

MIT
