"""
app.py — Streamlit web interface for the PDF RAG Assistant

Run with:
    streamlit run app.py

Features:
  - Drag-and-drop PDF upload with one-click ingestion
  - Multi-turn chat with conversation memory
  - Expandable source citations showing exact retrieved text + page numbers
  - Response time display
  - Live metrics (documents loaded, chunks indexed, turns, active top_k)
  - Clear chat / full reset controls
  - Auto-loads an existing FAISS index if present
"""

import json
import os
import tempfile
import time
from pathlib import Path

import streamlit as st

from src.config import config
from src.document_loader import load_and_split_pdfs
from src.vector_store import build_vector_store, load_vector_store
from src.rag_chain import build_rag_chain

# Curated free HuggingFace embedding models: name → short description for the UI
_HF_MODELS: dict[str, str] = {
    "BAAI/bge-small-en-v1.5":       "384 dim · 512 tok · ~130 MB  ← default, best speed/quality",
    "BAAI/bge-base-en-v1.5":        "768 dim · 512 tok · ~430 MB  — balanced step-up",
    "BAAI/bge-large-en-v1.5":       "1024 dim · 512 tok · ~1.3 GB — near-API quality",
    "nomic-ai/nomic-embed-text-v1": "768 dim · 8192 tok · ~550 MB — best for long docs",
    "all-mpnet-base-v2":            "768 dim · 514 tok · ~420 MB  — strong general-purpose",
    "all-MiniLM-L6-v2":             "384 dim · 256 tok · ~90 MB   — ultra-fast, small",
}

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 55%, #0f3460 100%);
    border-radius: 16px;
    padding: 28px 36px;
    color: white;
    margin-bottom: 20px;
}
.hero h1 { margin: 0 0 6px; font-size: 2rem; letter-spacing: -0.5px; }
.hero p  { margin: 0; opacity: 0.72; font-size: 1rem; }

/* ── Source citation badge ── */
.src-badge {
    display: inline-flex;
    align-items: center;
    background: #e8eaf6;
    color: #3949ab;
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px 3px 2px 0;
}

/* ── Source text excerpt ── */
.src-excerpt {
    background: #f8f9ff;
    border-left: 3px solid #667eea;
    padding: 8px 14px;
    border-radius: 0 8px 8px 0;
    font-size: 0.84rem;
    color: #333;
    margin: 6px 0 14px;
    line-height: 1.6;
    font-style: italic;
}

/* ── Empty state card ── */
.empty-state {
    text-align: center;
    padding: 48px 24px;
    color: #888;
    border: 2px dashed #ddd;
    border-radius: 16px;
    margin-top: 16px;
}
.empty-state h3 { color: #555; margin-bottom: 8px; }

/* ── Sidebar polish ── */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(102, 126, 234, 0.18);
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "messages": [],           # list[dict]: role, content, sources, elapsed
    "vector_store": None,
    "chain": None,
    "chain_top_k": None,      # top_k the current chain was built with
    "ingested_files": [],     # list[dict]: name, size_kb
    "total_chunks": 0,
    "index_auto_loaded": False,
    # Embedding selection (persisted across re-ingests so the UI remembers choice)
    "emb_provider": config.EMBEDDING_PROVIDER,
    "emb_model": config.EMBEDDING_MODEL,
    # What the current in-memory index was built with (for mismatch warning)
    "index_emb_provider": None,
    "index_emb_model": None,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Internal helpers ──────────────────────────────────────────────────────────

_SESSION_ID = "streamlit"


def _read_index_model_info() -> dict:
    """Return the provider/model recorded in faiss_index/model_info.json, or {}."""
    info_file = Path(config.FAISS_INDEX_PATH) / "model_info.json"
    if info_file.exists():
        try:
            return json.loads(info_file.read_text())
        except Exception:
            pass
    return {}


def _build_chain(top_k: int) -> None:
    """Create / recreate the RAG chain for the current vector store."""
    if st.session_state.vector_store is None:
        return
    st.session_state.chain = build_rag_chain(
        st.session_state.vector_store, top_k=top_k
    )
    st.session_state.chain_top_k = top_k


def _ingest(uploaded_files, top_k: int, emb_provider: str, emb_model: str) -> None:
    """
    Write uploaded file bytes to a temp directory, embed, persist FAISS index,
    and rebuild the RAG chain.  Resets the conversation history.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for uf in uploaded_files:
            dest = os.path.join(tmp_dir, uf.name)
            with open(dest, "wb") as fh:
                fh.write(uf.getbuffer())
            paths.append(dest)

        chunks = load_and_split_pdfs(paths)
        vs = build_vector_store(chunks, provider=emb_provider, model=emb_model)

    st.session_state.vector_store = vs
    st.session_state.total_chunks = len(chunks)
    st.session_state.ingested_files = [
        {"name": uf.name, "size_kb": f"{uf.size / 1024:.1f}"}
        for uf in uploaded_files
    ]
    st.session_state.index_emb_provider = emb_provider
    st.session_state.index_emb_model = emb_model
    # Reset conversation when the document set changes
    st.session_state.messages = []
    _build_chain(top_k)


def _reset_all() -> None:
    for k, v in _DEFAULTS.items():
        st.session_state[k] = (v.copy() if isinstance(v, list) else v)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📁 Document Library")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Drag & drop one or more PDF files here",
    )

    # ── Embedding model selector ───────────────────────────────────────────────
    st.markdown("#### 🧠 Embedding Model")

    emb_provider = st.selectbox(
        "Provider",
        options=["huggingface", "ollama"],
        index=0 if st.session_state.emb_provider == "huggingface" else 1,
        help="huggingface: auto-downloads model locally.  ollama: requires `ollama serve`.",
    )
    st.session_state.emb_provider = emb_provider

    if emb_provider == "huggingface":
        hf_model = st.selectbox(
            "Model",
            options=list(_HF_MODELS.keys()),
            index=list(_HF_MODELS.keys()).index(st.session_state.emb_model)
            if st.session_state.emb_model in _HF_MODELS
            else 0,
            format_func=lambda m: m,
            help="All models are free and run locally on CPU.",
        )
        st.caption(_HF_MODELS.get(hf_model, ""))
        emb_model = hf_model
    else:
        emb_model = st.text_input(
            "Ollama model name",
            value=st.session_state.emb_model
            if st.session_state.emb_provider == "ollama"
            else "nomic-embed-text",
            help="Run `ollama pull <model>` first. Common: nomic-embed-text, mxbai-embed-large",
        )
        st.caption("Requires `ollama serve` to be running.")

    st.session_state.emb_model = emb_model

    # Warn if the selected model differs from what the current index was built with
    _idx_info = _read_index_model_info()
    if _idx_info and (
        _idx_info.get("embedding_model") != emb_model
        or _idx_info.get("embedding_provider") != emb_provider
    ):
        st.warning(
            f"Index was built with **{_idx_info.get('embedding_model')}**. "
            "Re-index after changing the model.",
            icon="⚠️",
        )

    st.divider()

    top_k = st.slider(
        "Chunks retrieved per query",
        min_value=2,
        max_value=12,
        value=st.session_state.chain_top_k or 6,
        step=1,
        help="More chunks → broader context but slower & costlier per query.",
    )

    # Rebuild chain if top_k was adjusted without re-ingesting
    if (
        st.session_state.chain is not None
        and st.session_state.chain_top_k != top_k
    ):
        _build_chain(top_k)

    ingest_clicked = st.button(
        "⚡ Embed & index",
        type="primary",
        use_container_width=True,
        disabled=not uploaded,
        help="Embed the uploaded PDFs and build / replace the FAISS index.",
    )

    if ingest_clicked and uploaded:
        with st.spinner(f"Embedding {len(uploaded)} PDF(s) with **{emb_model}** — this may take a moment…"):
            _ingest(uploaded, top_k, emb_provider, emb_model)
        st.success(
            f"✓ Indexed {len(uploaded)} file(s) · "
            f"{st.session_state.total_chunks} chunks ready"
        )
        st.rerun()

    # Auto-load an existing FAISS index when the app starts
    if (
        st.session_state.vector_store is None
        and not uploaded
        and not st.session_state.index_auto_loaded
    ):
        st.session_state.index_auto_loaded = True
        try:
            with st.spinner("Looking for existing index…"):
                vs = load_vector_store(provider=emb_provider, model=emb_model)
            st.session_state.vector_store = vs
            # Populate index model info from disk so the mismatch warning is accurate
            _saved = _read_index_model_info()
            st.session_state.index_emb_provider = _saved.get("embedding_provider")
            st.session_state.index_emb_model = _saved.get("embedding_model")
            _build_chain(top_k)
            st.info("Loaded existing FAISS index from disk.")
        except FileNotFoundError:
            pass

    # ── Indexed files list ────────────────────────────────────────────────────
    if st.session_state.ingested_files:
        st.divider()
        st.markdown("**Indexed files**")
        for f in st.session_state.ingested_files:
            st.markdown(f"📄 `{f['name']}` &nbsp; {f['size_kb']} KB")
        st.caption(f"Total index chunks: **{st.session_state.total_chunks}**")

    # ── Session controls ──────────────────────────────────────────────────────
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "🗑 Clear chat",
            use_container_width=True,
            help="Clear conversation history (keeps the index)",
        ):
            st.session_state.messages = []
            # Rebuild chain to also reset LangChain's internal memory store
            _build_chain(top_k)
            st.rerun()
    with c2:
        if st.button(
            "🔄 Reset all",
            use_container_width=True,
            help="Remove index and clear conversation",
        ):
            _reset_all()
            st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    if config.LLM_PROVIDER == "azure":
        _model_label = f"Azure · {config.AZURE_OPENAI_DEPLOYMENT}"
    else:
        _model_label = f"OpenAI · {config.LLM_MODEL}"

    st.caption(f"LLM: **{_model_label}**")
    _active_emb = st.session_state.index_emb_model or emb_model
    st.caption(f"Embeddings: **{st.session_state.index_emb_provider or emb_provider}** · `{_active_emb}`")
    st.caption("PDF RAG Assistant · LangChain + FAISS + Streamlit")


# ── Main area — hero ──────────────────────────────────────────────────────────

st.markdown(
    """
<div class="hero">
  <h1>📄 PDF RAG Assistant</h1>
  <p>Upload your documents · Ask natural-language questions ·
     Get grounded answers with page-level source citations</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Metrics row ───────────────────────────────────────────────────────────────

m1, m2, m3, m4 = st.columns(4)
m1.metric("Documents", len(st.session_state.ingested_files))
m2.metric("Index chunks", st.session_state.total_chunks)
m3.metric(
    "Conversation turns",
    sum(1 for m in st.session_state.messages if m["role"] == "user"),
)
m4.metric("Active top-k", st.session_state.chain_top_k or "—")

st.divider()

# ── Chat history display ──────────────────────────────────────────────────────

if not st.session_state.messages and st.session_state.chain is None:
    st.markdown(
        """
<div class="empty-state">
  <h3>No documents loaded yet</h3>
  <p>Upload one or more PDF files in the sidebar and click
     <strong>⚡ Embed &amp; index</strong> to get started.</p>
</div>
""",
        unsafe_allow_html=True,
    )
elif not st.session_state.messages:
    st.markdown(
        """
<div class="empty-state">
  <h3>Ready — ask your first question</h3>
  <p>Type in the chat box below to start exploring your documents.</p>
</div>
""",
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            sources = msg.get("sources", [])
            elapsed = msg.get("elapsed")

            if sources:
                expander_label = f"📚 View sources &nbsp;·&nbsp; {len(sources)} chunks retrieved"
                if elapsed is not None:
                    expander_label += f" &nbsp;·&nbsp; _{elapsed:.1f} s_"

                with st.expander(expander_label):
                    for i, doc in enumerate(sources, 1):
                        fn  = doc.metadata.get("filename", "unknown")
                        pg  = doc.metadata.get("page_display", "?")
                        txt = doc.page_content

                        st.markdown(
                            f'<span class="src-badge">📄 {fn}</span>'
                            f'<span class="src-badge">Page {pg}</span>',
                            unsafe_allow_html=True,
                        )
                        preview = txt[:400] + ("…" if len(txt) > 400 else "")
                        st.markdown(
                            f'<div class="src-excerpt">{preview}</div>',
                            unsafe_allow_html=True,
                        )
                        if i < len(sources):
                            st.markdown("---")
            elif elapsed is not None:
                st.caption(f"_{elapsed:.1f} s_")

# ── Chat input ────────────────────────────────────────────────────────────────

_input_placeholder = (
    "Ask a question about your documents…"
    if st.session_state.chain
    else "Upload PDFs and click '⚡ Embed & index' to begin…"
)

if user_input := st.chat_input(
    _input_placeholder, disabled=not st.session_state.chain
):
    # Render the user bubble immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    st.session_state.messages.append(
        {"role": "user", "content": user_input, "sources": []}
    )

    # Invoke the chain and render the assistant response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            t0 = time.perf_counter()
            result = st.session_state.chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": _SESSION_ID}},
            )
            elapsed = time.perf_counter() - t0

        answer  = result.get("answer", "")
        sources = result.get("context", [])

        st.markdown(answer)

        if sources:
            expander_label = (
                f"📚 View sources &nbsp;·&nbsp; {len(sources)} chunks retrieved"
                f" &nbsp;·&nbsp; _{elapsed:.1f} s_"
            )
            with st.expander(expander_label):
                for i, doc in enumerate(sources, 1):
                    fn  = doc.metadata.get("filename", "unknown")
                    pg  = doc.metadata.get("page_display", "?")
                    txt = doc.page_content

                    st.markdown(
                        f'<span class="src-badge">📄 {fn}</span>'
                        f'<span class="src-badge">Page {pg}</span>',
                        unsafe_allow_html=True,
                    )
                    preview = txt[:400] + ("…" if len(txt) > 400 else "")
                    st.markdown(
                        f'<div class="src-excerpt">{preview}</div>',
                        unsafe_allow_html=True,
                    )
                    if i < len(sources):
                        st.markdown("---")
        else:
            st.caption(f"_{elapsed:.1f} s_")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "elapsed": round(elapsed, 2),
        }
    )
