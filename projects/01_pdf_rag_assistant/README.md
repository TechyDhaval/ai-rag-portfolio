# PDF RAG Assistant

> **Part of the [AI RAG Portfolio](../../README.md)** — Project 01

A production-ready **Retrieval-Augmented Generation (RAG)** application that lets you chat with any PDF document. Demonstrates the core RAG loop with multi-turn conversation memory, MMR retrieval, configurable embedding models, and source citations.

## Architecture

```
One or more PDF files
   │
   ▼
PyPDFLoader (per PDF)          ← LangChain document loader
   │
   ▼
RecursiveCharacterTextSplitter ← splits text into overlapping chunks
   │
   ▼
all-MiniLM-L6-v2               ← local HuggingFace embedding model (no API cost)
   │
   ▼
FAISS Vector Store             ← persisted to disk for reuse
   │
   ▼
History-Aware Retriever        ← rewrites query using prior chat turns
   │
   ▼
Conversational RAG Chain       ← context + chat history + question → LLM
   │
   ▼
GPT-4o-mini                    ← OpenAI LLM answers grounded in retrieved context
```

## Tech Stack

| Component     | Library / Tool                               |
|---------------|----------------------------------------------|
| Framework     | LangChain                                    |
| PDF Loader    | PyPDFLoader (pypdf)                          |
| Embeddings    | `all-MiniLM-L6-v2` via sentence-transformers |
| Vector DB     | FAISS (local, in-process)                    |
| LLM           | OpenAI GPT-4o-mini / Azure OpenAI            |
| Web UI        | Streamlit                                    |
| Config        | python-dotenv                                |

## Quick Start

### 1. Clone and navigate

```bash
git clone https://github.com/YOUR_USERNAME/ai-rag-portfolio.git
cd ai-rag-portfolio/projects/01_pdf_rag_assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and set your OPENAI_API_KEY
```

### 3. Run

**Streamlit web UI (recommended)**
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. Upload PDFs via drag-and-drop,
click **⚡ Embed & index**, then start chatting. Each answer shows expandable
source citations with the exact text and page number used.

**Command-line interface**
```bash
# Ingest a single PDF and start chatting
python main.py --pdf docs/your_document.pdf

# Ingest multiple PDFs into one index
python main.py --pdf docs/report.pdf docs/appendix.pdf

# Ingest every PDF in a directory
python main.py --pdf-dir docs/

# Reuse a previously built index (no re-ingestion)
python main.py

# Build the index only, skip the chat session
python main.py --pdf docs/your_document.pdf --ingest-only

# Control how many chunks are retrieved per query (default: 6)
python main.py --top-k 8
```

### Example session

```
[Loader] Loading PDF: research_paper.pdf
[Loader] Loaded 12 page(s)
[Loader] Split into 47 chunk(s) (size=1000, overlap=200)
[Loader] Total chunks across 1 PDF(s): 47
[VectorStore] Building FAISS index from 47 chunk(s)...
[VectorStore] Index saved to: faiss_index/

============================================================
  PDF RAG Assistant — ready!
  The assistant remembers your conversation history.
  Type your question and press Enter.
  Type 'exit' or press Ctrl+C to quit.
============================================================

You: What is the main contribution of this paper?
Assistant: The paper introduces a novel attention mechanism that reduces
           quadratic complexity to linear, enabling transformers to scale
           to much longer sequences...

You: Can you elaborate on how that works?
Assistant: Building on the mechanism described above, the authors use a
           kernel approximation to rewrite the softmax attention as a
           linear dot-product of feature maps...
```

## Project Structure

```
01_pdf_rag_assistant/
├── app.py                   # Streamlit web UI
├── main.py                  # CLI entry point
├── requirements.txt
├── .env.example             # Template for environment variables
├── ARCHITECTURE.md          # System design and decision rationale
├── EXERCISES.md             # Hands-on challenges to extend the project
├── docs/                    # Drop your PDFs here (CLI mode)
└── src/
    ├── config.py            # Centralised settings (reads .env)
    ├── document_loader.py   # PDF loading and text splitting
    ├── vector_store.py      # FAISS build / load helpers
    └── rag_chain.py         # Conversational RAG chain with memory
```

## Key Concepts Demonstrated

- **RAG pattern**: grounding LLM answers in retrieved document context to reduce hallucination
- **Multi-document ingestion**: combine chunks from multiple PDFs into a single FAISS index
- **Conversation memory**: `RunnableWithMessageHistory` + `ChatMessageHistory` keep the full chat history in session
- **History-aware retrieval**: the query is reformulated with `create_history_aware_retriever` so retrieval resolves pronouns and references across turns
- **Local embeddings**: zero-cost semantic search using a local sentence-transformer model
- **Persistent vector store**: FAISS index saved to disk so ingestion only runs once
- **LangChain LCEL**: modern chain composition with `create_retrieval_chain` and `create_stuff_documents_chain`
- **Separation of concerns**: loader / vector store / chain / config each in their own module

## Roadmap

- [x] Streamlit web UI with source citations
- [x] Support multiple PDFs in one session
- [ ] Swap LLM provider (Anthropic Claude, Google Gemini)
- [x] Add conversation memory / chat history
- [ ] Dockerize the application
- [ ] REST API with FastAPI

## License

MIT
