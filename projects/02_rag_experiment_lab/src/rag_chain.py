"""
src/rag_chain.py
Builds a conversational RAG chain with fully configurable LLM, retrieval,
reranking, and prompt template — all swappable at runtime.

Key differences from Project 01:
  - LLM model and temperature are parameters, not fixed at import time
  - Prompt template is selectable from a catalogue of system prompts
  - Reranker can be inserted between retrieval and QA
  - A separate `ask()` function returns structured results including
    retrieved chunks, timings, and the answer — needed for evaluation

Learning note — prompt engineering for RAG:
  The system prompt has outsized impact on answer quality. A prompt that says
  "answer only from context" reduces hallucination but may refuse answerable
  questions. A prompt that says "use your knowledge too" is more helpful but
  hallucinates more. The Experiment Lab gives you multiple prompts to compare
  empirically — this is how you learn prompt engineering for RAG.
"""

import time
from typing import List, Optional

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from src.config import config
from src.retriever import build_retriever
from src.reranker import rerank


# ── Prompt catalogue ─────────────────────────────────────────────────────────
# Each prompt is a (name, description, system_template) tuple.
# The UI shows the name and description; the template is used in the chain.

PROMPT_TEMPLATES = {
    "strict": {
        "name": "Strict (no hallucination)",
        "desc": "Only answers from retrieved context. Says 'I don't know' otherwise.",
        "template": (
            "You are a helpful assistant that answers questions based strictly "
            "on the provided context extracted from one or more documents.\n\n"
            "If the answer cannot be found in the context, say:\n"
            '"I don\'t have enough information in the provided document(s) to answer that."\n\n'
            "Do not make up information. When citing information, include the source "
            "file and page number shown in the context.\n\n"
            "Context:\n{context}"
        ),
    },
    "balanced": {
        "name": "Balanced (context-preferred)",
        "desc": "Prefers context but supplements with general knowledge when helpful.",
        "template": (
            "You are a helpful assistant. Answer questions primarily using the "
            "provided context from the user's documents. You may supplement with "
            "general knowledge when the context is insufficient, but clearly "
            "distinguish between information from the documents and your own knowledge.\n\n"
            "When using document information, cite the source file and page number.\n"
            "When using your own knowledge, prefix with 'Based on my general knowledge: '.\n\n"
            "Context:\n{context}"
        ),
    },
    "concise": {
        "name": "Concise (bullet points)",
        "desc": "Short, structured answers in bullet-point format.",
        "template": (
            "You are a concise assistant. Answer questions using the provided context.\n"
            "Rules:\n"
            "- Use bullet points\n"
            "- Keep answers under 150 words\n"
            "- Cite source files and page numbers\n"
            "- Say 'Not found in documents' if the context lacks the answer\n\n"
            "Context:\n{context}"
        ),
    },
    "detailed": {
        "name": "Detailed (thorough analysis)",
        "desc": "Long-form, thorough answers with explanations and all relevant citations.",
        "template": (
            "You are a thorough research assistant. Provide detailed, comprehensive "
            "answers based on the provided context. Structure your answer with clear "
            "sections when appropriate. Include all relevant citations with source "
            "file names and page numbers. If the context is insufficient, explain "
            "what information is available and what is missing.\n\n"
            "Context:\n{context}"
        ),
    },
}


# ── LLM factory ──────────────────────────────────────────────────────────────

def _build_llm(model: str = None, temperature: float = None):
    """
    Return the appropriate LangChain LLM.

    Args:
        model:       Model name (e.g. 'gpt-4o-mini'). Overrides config.
        temperature: LLM temperature. Overrides config.

    Learning note — temperature effect on RAG:
        For RAG, low temperatures (0.0-0.3) are usually best because you want
        the LLM to faithfully summarise the retrieved context, not creatively
        rephrase it. Higher temperatures increase variety but also increase
        the chance of hallucinating details not in the context.
    """
    model = model or config.LLM_MODEL
    temperature = temperature if temperature is not None else config.LLM_TEMPERATURE

    if config.LLM_PROVIDER == "azure":
        return AzureChatOpenAI(
            azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            temperature=temperature,
        )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=config.OPENAI_API_KEY,
    )


# ── Contextualisation prompt ─────────────────────────────────────────────────

_CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given a chat history and the latest user question which might "
        "reference context in the chat history, formulate a standalone "
        "question that can be understood without the chat history. "
        "Do NOT answer the question — just reformulate it if needed, "
        "otherwise return it as is.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

_DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[Source: {filename}, Page {page_display}]\n{page_content}"
)


# ── Chain builder ────────────────────────────────────────────────────────────

def build_rag_chain(
    vector_store: FAISS,
    llm_model: str = None,
    temperature: float = None,
    search_type: str = None,
    top_k: int = None,
    score_threshold: float = None,
    mmr_lambda: float = None,
    prompt_key: str = "strict",
):
    """
    Build a full conversational RAG chain with configurable parameters.

    Args:
        vector_store:    Populated FAISS vector store.
        llm_model:       LLM model name (e.g. 'gpt-4o-mini').
        temperature:     LLM temperature (0.0-1.0).
        search_type:     Retrieval search type.
        top_k:           Number of chunks to retrieve.
        score_threshold: Minimum similarity score for threshold search.
        mmr_lambda:      MMR lambda multiplier.
        prompt_key:      Key from PROMPT_TEMPLATES dict.

    Returns:
        A RunnableWithMessageHistory chain.
    """
    config.validate()

    llm = _build_llm(model=llm_model, temperature=temperature)
    retriever = build_retriever(
        vector_store,
        search_type=search_type,
        top_k=top_k,
        score_threshold=score_threshold,
        mmr_lambda=mmr_lambda,
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, _CONTEXTUALIZE_PROMPT
    )

    prompt_config = PROMPT_TEMPLATES.get(prompt_key, PROMPT_TEMPLATES["strict"])
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_config["template"]),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(
        llm, qa_prompt, document_prompt=_DOCUMENT_PROMPT
    )
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    store: dict = {}

    def _get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# ── Single-question API (for evaluation) ─────────────────────────────────────

def ask(
    question: str,
    vector_store: FAISS,
    llm_model: str = None,
    temperature: float = None,
    search_type: str = None,
    top_k: int = None,
    score_threshold: float = None,
    mmr_lambda: float = None,
    prompt_key: str = "strict",
    reranker_enabled: bool = False,
    reranker_model: str = None,
    reranker_top_n: int = None,
) -> dict:
    """
    Ask a single question and return a structured result dict.

    Unlike the chain-based approach, this function is stateless (no conversation
    history) and returns timing information — ideal for batch evaluation.

    Args:
        question:         The question to ask.
        vector_store:     Populated FAISS vector store.
        (other args):     Same as build_rag_chain, plus reranker options.

    Returns:
        dict with keys:
          - answer: str
          - context: List[Document] (retrieved/reranked chunks)
          - retrieval_time_s: float
          - llm_time_s: float
          - total_time_s: float
    """
    config.validate()

    top_k = top_k or config.TOP_K

    # 1. Retrieve
    t0 = time.perf_counter()
    retriever = build_retriever(
        vector_store,
        search_type=search_type,
        top_k=top_k,
        score_threshold=score_threshold,
        mmr_lambda=mmr_lambda,
    )
    docs = retriever.invoke(question)
    t_retrieval = time.perf_counter() - t0

    # 2. Rerank (optional)
    if reranker_enabled and docs:
        t_rerank_start = time.perf_counter()
        docs = rerank(
            question, docs,
            top_n=reranker_top_n,
            model_name=reranker_model,
        )
        t_retrieval += time.perf_counter() - t_rerank_start

    # 3. Build context string
    context_str = "\n\n".join(
        _DOCUMENT_PROMPT.format(
            filename=d.metadata.get("filename", "unknown"),
            page_display=d.metadata.get("page_display", "?"),
            page_content=d.page_content,
        )
        for d in docs
    )

    # 4. LLM call
    prompt_config = PROMPT_TEMPLATES.get(prompt_key, PROMPT_TEMPLATES["strict"])
    llm = _build_llm(model=llm_model, temperature=temperature)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_config["template"]),
        ("human", "{input}"),
    ])
    chain = qa_prompt | llm

    t1 = time.perf_counter()
    result = chain.invoke({"context": context_str, "input": question})
    t_llm = time.perf_counter() - t1

    return {
        "answer": result.content,
        "context": docs,
        "retrieval_time_s": round(t_retrieval, 3),
        "llm_time_s": round(t_llm, 3),
        "total_time_s": round(t_retrieval + t_llm, 3),
    }
