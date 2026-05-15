"""
app.py — Streamlit web interface for the RAG Experiment Lab

Run with:
    cd projects/02_rag_experiment_lab
    streamlit run app.py

Three tabs:
  1. Chat — talk to your documents with live parameter controls
  2. Evaluate — run a test set and score retrieval/answer quality
  3. Compare — view saved experiments side by side

All RAG parameters are adjustable in the sidebar at runtime.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import config
from src.embeddings import EMBEDDING_MODELS, get_embeddings
from src.document_loader import load_and_split_pdfs
from src.vector_store import build_vector_store, load_vector_store, index_exists
from src.retriever import SEARCH_TYPES
from src.rag_chain import PROMPT_TEMPLATES, build_rag_chain, ask
from src.evaluator import evaluate_single
from src.experiment import (
    save_experiment, list_experiments, load_experiment, delete_experiment,
    compare_experiments, list_test_sets, load_test_set,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Experiment Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 55%, #415a77 100%);
    border-radius: 16px; padding: 28px 36px; color: white; margin-bottom: 20px;
}
.hero h1 { margin: 0 0 6px; font-size: 2rem; letter-spacing: -0.5px; }
.hero p  { margin: 0; opacity: 0.72; font-size: 1rem; }
.metric-card {
    background: #f0f4ff; border-radius: 12px; padding: 14px 18px;
    text-align: center; border: 1px solid #dde4f6;
}
.metric-card h3 { margin: 0; font-size: 1.6rem; color: #3949ab; }
.metric-card p  { margin: 4px 0 0; font-size: 0.82rem; color: #666; }
.src-badge {
    display: inline-flex; align-items: center; background: #e8eaf6;
    color: #3949ab; border-radius: 20px; padding: 3px 11px;
    font-size: 0.78rem; font-weight: 600; margin: 2px 3px 2px 0;
}
.src-excerpt {
    background: #f8f9ff; border-left: 3px solid #667eea;
    padding: 8px 14px; border-radius: 0 8px 8px 0; font-size: 0.84rem;
    color: #333; margin: 6px 0 14px; line-height: 1.6; font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# ── Hero header ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>🧪 RAG Experiment Lab</h1>
    <p>Tune every parameter, evaluate quality, and find the best RAG configuration for your documents.</p>
</div>
""", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────────────────────

def _init_state():
    defaults = {
        "vector_store": None,
        "rag_chain": None,
        "chat_history": [],
        "chain_params_hash": None,
        "docs_loaded": 0,
        "chunks_indexed": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — parameter controls
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Parameters")

    # ── Document upload ──────────────────────────────────────────────────────
    st.subheader("📁 Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True,
        help="Upload one or more PDF files to index.",
    )

    # ── Embedding model ──────────────────────────────────────────────────────
    st.subheader("Embeddings")
    model_names = list(EMBEDDING_MODELS.keys())
    current_model_idx = model_names.index(config.EMBEDDING_MODEL) if config.EMBEDDING_MODEL in model_names else 0
    embedding_model = st.selectbox(
        "Embedding model",
        model_names,
        index=current_model_idx,
        format_func=lambda m: f"{m.split('/')[-1]}  ({EMBEDDING_MODELS[m]['dim']}d, {EMBEDDING_MODELS[m]['size_mb']}MB)",
        help="Changing the embedding model requires re-indexing documents.",
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    st.subheader("Chunking")
    chunk_size = st.slider(
        "Chunk size (chars)", 200, 3000, config.CHUNK_SIZE, step=100,
        help="Larger chunks give more context but reduce precision.",
    )
    chunk_overlap = st.slider(
        "Chunk overlap (chars)", 0, 500, config.CHUNK_OVERLAP, step=50,
        help="Overlap ensures ideas at chunk boundaries aren't lost.",
    )

    # ── Retrieval ────────────────────────────────────────────────────────────
    st.subheader("Retrieval")
    search_type = st.selectbox(
        "Search strategy",
        list(SEARCH_TYPES.keys()),
        index=list(SEARCH_TYPES.keys()).index(config.SEARCH_TYPE),
        format_func=lambda k: SEARCH_TYPES[k],
        help="MMR balances relevance and diversity. Similarity is pure cosine.",
    )
    top_k = st.slider(
        "Top-K chunks", 1, 20, config.TOP_K,
        help="Number of chunks to retrieve. More = broader context, slower.",
    )

    # Conditional controls based on search type
    score_threshold = config.SCORE_THRESHOLD
    mmr_lambda = config.MMR_LAMBDA
    if search_type == "similarity_score_threshold":
        score_threshold = st.slider(
            "Score threshold", 0.0, 1.0, config.SCORE_THRESHOLD, step=0.05,
            help="Minimum similarity score to include a chunk.",
        )
    if search_type == "mmr":
        mmr_lambda = st.slider(
            "MMR lambda", 0.0, 1.0, config.MMR_LAMBDA, step=0.1,
            help="0 = max diversity, 1 = max relevance.",
        )

    # ── Reranker ─────────────────────────────────────────────────────────────
    st.subheader("Reranker")
    reranker_enabled = st.toggle(
        "Enable cross-encoder reranking",
        value=config.RERANKER_ENABLED,
        help="Re-scores retrieved chunks with a cross-encoder for better precision.",
    )
    reranker_top_n = config.RERANKER_TOP_N
    if reranker_enabled:
        reranker_top_n = st.slider(
            "Reranker top-N", 1, top_k, min(config.RERANKER_TOP_N, top_k),
            help="Keep the top N chunks after reranking.",
        )

    # ── LLM ──────────────────────────────────────────────────────────────────
    st.subheader("LLM")
    llm_model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
        help="GPT-4o-mini is cheapest; GPT-4o is most capable.",
    )
    temperature = st.slider(
        "Temperature", 0.0, 1.0, config.LLM_TEMPERATURE, step=0.1,
        help="0 = deterministic, 1 = creative. Low is best for RAG.",
    )

    # ── Prompt ───────────────────────────────────────────────────────────────
    st.subheader("Prompt")
    prompt_key = st.selectbox(
        "Prompt template",
        list(PROMPT_TEMPLATES.keys()),
        format_func=lambda k: PROMPT_TEMPLATES[k]["name"],
        help="System prompt controls answer style and hallucination rate.",
    )
    with st.expander("Preview prompt"):
        st.code(PROMPT_TEMPLATES[prompt_key]["template"], language=None)

    # ── Ingestion button ─────────────────────────────────────────────────────
    st.divider()
    if uploaded_files:
        if st.button("🔄 (Re)Index Documents", use_container_width=True, type="primary"):
            with st.spinner("Loading PDFs and building index..."):
                tmp_paths = []
                for uf in uploaded_files:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(uf.read())
                    tmp.close()
                    tmp_paths.append(tmp.name)

                chunks = load_and_split_pdfs(
                    tmp_paths,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                embeddings = get_embeddings(model_name=embedding_model)
                vs = build_vector_store(chunks, embeddings, model_name=embedding_model)

                st.session_state.vector_store = vs
                st.session_state.docs_loaded = len(uploaded_files)
                st.session_state.chunks_indexed = len(chunks)
                st.session_state.chat_history = []
                st.session_state.rag_chain = None

                for p in tmp_paths:
                    os.unlink(p)

            st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} PDF(s).")

    # Auto-load existing index
    if st.session_state.vector_store is None and index_exists(embedding_model):
        try:
            embeddings = get_embeddings(model_name=embedding_model)
            st.session_state.vector_store = load_vector_store(embeddings, model_name=embedding_model)
            st.sidebar.info("Loaded existing index from disk.")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — three tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_chat, tab_eval, tab_compare = st.tabs(["💬 Chat", "📊 Evaluate", "📈 Compare"])


# ── Helper: current params dict ──────────────────────────────────────────────

def _current_params() -> dict:
    return {
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "search_type": search_type,
        "top_k": top_k,
        "score_threshold": score_threshold,
        "mmr_lambda": mmr_lambda,
        "llm_model": llm_model,
        "temperature": temperature,
        "prompt_key": prompt_key,
        "reranker_enabled": reranker_enabled,
        "reranker_top_n": reranker_top_n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chat
# ══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    if st.session_state.vector_store is None:
        st.info("Upload PDF(s) in the sidebar and click **Index Documents** to start.")
    else:
        # Metrics row
        c1, c2, c3 = st.columns(3)
        c1.metric("Documents", st.session_state.docs_loaded)
        c2.metric("Chunks", st.session_state.chunks_indexed)
        c3.metric("Messages", len(st.session_state.chat_history))

        # Build / rebuild chain if params changed
        params_hash = str(_current_params())
        if st.session_state.chain_params_hash != params_hash:
            st.session_state.rag_chain = build_rag_chain(
                st.session_state.vector_store,
                llm_model=llm_model,
                temperature=temperature,
                search_type=search_type,
                top_k=top_k,
                score_threshold=score_threshold,
                mmr_lambda=mmr_lambda,
                prompt_key=prompt_key,
            )
            st.session_state.chain_params_hash = params_hash

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander(f"📑 {len(msg['sources'])} source(s) · {msg.get('time', '?')}s"):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<span class="src-badge">{src["file"]} · p.{src["page"]}</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="src-excerpt">{src["text"][:400]}...</div>',
                                unsafe_allow_html=True,
                            )

        # Chat input
        if user_input := st.chat_input("Ask a question about your documents..."):
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    t0 = time.perf_counter()
                    response = st.session_state.rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": "streamlit_chat"}},
                    )
                    elapsed = round(time.perf_counter() - t0, 2)

                answer = response["answer"]
                st.markdown(answer)

                sources = []
                for doc in response.get("context", []):
                    sources.append({
                        "file": doc.metadata.get("filename", "unknown"),
                        "page": doc.metadata.get("page_display", "?"),
                        "text": doc.page_content,
                    })

                if sources:
                    with st.expander(f"📑 {len(sources)} source(s) · {elapsed}s"):
                        for src in sources:
                            st.markdown(
                                f'<span class="src-badge">{src["file"]} · p.{src["page"]}</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="src-excerpt">{src["text"][:400]}...</div>',
                                unsafe_allow_html=True,
                            )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "time": elapsed,
                })

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.rag_chain = None
                st.session_state.chain_params_hash = None
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Evaluate
# ══════════════════════════════════════════════════════════════════════════════

with tab_eval:
    if st.session_state.vector_store is None:
        st.info("Index documents first (sidebar) before running evaluations.")
    else:
        st.markdown("### Run evaluation on a test set")
        st.markdown(
            "Select a test set file or enter questions manually. "
            "Each question is answered by the RAG chain, then scored by an LLM judge."
        )

        # Input method
        input_method = st.radio(
            "Question source", ["Manual entry", "Test set file"],
            horizontal=True,
        )

        questions = []
        if input_method == "Test set file":
            available_sets = list_test_sets()
            if not available_sets:
                st.warning(
                    f"No test sets found in `{config.TEST_SETS_DIR}/`. "
                    "Create a JSON file with `[{\"question\": \"...\"}]` format."
                )
            else:
                selected_set = st.selectbox("Test set", available_sets)
                if selected_set:
                    test_data = load_test_set(selected_set)
                    questions = [item["question"] for item in test_data]
                    st.caption(f"{len(questions)} question(s) loaded.")
        else:
            manual_text = st.text_area(
                "Enter questions (one per line)",
                height=150,
                placeholder="What is the main argument?\nWhat methodology was used?\nWhat are the key findings?",
            )
            if manual_text.strip():
                questions = [q.strip() for q in manual_text.strip().split("\n") if q.strip()]

        experiment_label = st.text_input("Experiment label (optional)", placeholder="e.g. baseline_strict")

        if questions and st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
            params = _current_params()
            results = []
            progress = st.progress(0)
            status = st.empty()

            for i, q in enumerate(questions):
                status.text(f"Evaluating question {i+1}/{len(questions)}: {q[:60]}...")
                progress.progress((i) / len(questions))

                # Get RAG answer
                result = ask(
                    question=q,
                    vector_store=st.session_state.vector_store,
                    llm_model=llm_model,
                    temperature=temperature,
                    search_type=search_type,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    mmr_lambda=mmr_lambda,
                    prompt_key=prompt_key,
                    reranker_enabled=reranker_enabled,
                    reranker_model=config.RERANKER_MODEL,
                    reranker_top_n=reranker_top_n,
                )

                # Evaluate
                scores = evaluate_single(q, result["answer"], result["context"])

                results.append({
                    "question": q,
                    "answer": result["answer"],
                    "context_relevance": scores["context_relevance"],
                    "faithfulness": scores["faithfulness"],
                    "answer_relevance": scores["answer_relevance"],
                    "retrieval_time_s": result["retrieval_time_s"],
                    "llm_time_s": result["llm_time_s"],
                    "total_time_s": result["total_time_s"],
                    "num_chunks": len(result["context"]),
                })

            progress.progress(1.0)
            status.text("✅ Evaluation complete!")

            # Save experiment
            exp_name = save_experiment(params, results, label=experiment_label)
            st.success(f"Experiment saved: `{exp_name}`")

            # Display results
            st.markdown("### Results")

            # Aggregate metrics
            n = len(results)
            cols = st.columns(4)
            avg_cr = sum(r["context_relevance"] for r in results) / n
            avg_f = sum(r["faithfulness"] for r in results) / n
            avg_ar = sum(r["answer_relevance"] for r in results) / n
            avg_t = sum(r["total_time_s"] for r in results) / n
            cols[0].metric("Avg Context Relevance", f"{avg_cr:.3f}")
            cols[1].metric("Avg Faithfulness", f"{avg_f:.3f}")
            cols[2].metric("Avg Answer Relevance", f"{avg_ar:.3f}")
            cols[3].metric("Avg Response Time", f"{avg_t:.2f}s")

            # Per-question table
            df = pd.DataFrame(results)
            display_cols = [
                "question", "context_relevance", "faithfulness",
                "answer_relevance", "total_time_s",
            ]
            st.dataframe(
                df[display_cols],
                use_container_width=True,
                hide_index=True,
            )

            # Expandable per-question details
            for i, r in enumerate(results):
                with st.expander(f"Q{i+1}: {r['question'][:80]}"):
                    st.markdown(f"**Answer:** {r['answer']}")
                    st.markdown(
                        f"Context Relevance: `{r['context_relevance']}` · "
                        f"Faithfulness: `{r['faithfulness']}` · "
                        f"Answer Relevance: `{r['answer_relevance']}` · "
                        f"Time: `{r['total_time_s']}s` · "
                        f"Chunks: `{r['num_chunks']}`"
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Compare
# ══════════════════════════════════════════════════════════════════════════════

with tab_compare:
    st.markdown("### Compare Experiments")
    st.markdown("Select two or more saved experiments to compare their configurations and scores.")

    experiments = list_experiments()

    if not experiments:
        st.info(
            "No saved experiments yet. Run an evaluation in the **Evaluate** tab first."
        )
    else:
        # Multi-select with formatted names
        exp_options = {
            e["name"]: f"{e.get('label', '') or e['name'][:30]}  ({e['aggregates'].get('num_questions', '?')} Qs)"
            for e in experiments
        }
        selected = st.multiselect(
            "Select experiments to compare",
            list(exp_options.keys()),
            format_func=lambda k: exp_options[k],
        )

        if len(selected) >= 2:
            comparison = compare_experiments(selected)
            df = pd.DataFrame(comparison)

            # Highlight columns
            score_cols = ["avg_context_relevance", "avg_faithfulness", "avg_answer_relevance"]
            param_cols = ["embedding_model", "chunk_size", "search_type", "top_k", "temperature", "prompt_key", "reranker"]

            st.markdown("#### Parameter Differences")
            st.dataframe(
                df[["label"] + param_cols],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Score Comparison")
            st.dataframe(
                df[["label"] + score_cols + ["avg_total_time_s"]].style.highlight_max(
                    subset=score_cols, color="#c8e6c9"
                ).highlight_min(
                    subset=score_cols, color="#ffcdd2"
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Bar chart of scores
            st.markdown("#### Visual Comparison")
            chart_df = df[["label"] + score_cols].set_index("label")
            st.bar_chart(chart_df)

        elif len(selected) == 1:
            exp = load_experiment(selected[0])
            st.json(exp.get("params", {}))

            results = exp.get("results", [])
            if results:
                df = pd.DataFrame(results)
                display_cols = [c for c in ["question", "context_relevance", "faithfulness", "answer_relevance", "total_time_s"] if c in df.columns]
                st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        # Delete experiments
        if experiments:
            with st.expander("🗑️ Delete experiments"):
                to_delete = st.selectbox("Select experiment to delete", [e["name"] for e in experiments])
                if st.button("Delete", type="secondary"):
                    delete_experiment(to_delete)
                    st.success(f"Deleted: {to_delete}")
                    st.rerun()
