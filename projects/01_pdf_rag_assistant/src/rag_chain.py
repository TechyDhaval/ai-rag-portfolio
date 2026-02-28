"""
src/rag_chain.py
Builds a conversational RAG chain with memory using LangChain.

Flow:
  User question + chat history
    → history-aware retriever (contextualises query)
    → MMR FAISS retriever (top-k diverse chunks)
    → QA prompt (context + history + question)
    → LLM
    → answer string with source citations

Learning note — why two separate prompts?
  _CONTEXTUALIZE_PROMPT rewrites the question into a standalone form before
  retrieval. _QA_PROMPT then answers the question given the retrieved context.
  Separating them is important: the retriever needs a clean, self-contained
  query string, while the LLM answerer needs the full conversation history to
  maintain coherent multi-turn dialogue. Mixing both concerns into one prompt
  degrades both retrieval and answer quality.

Learning note — two LLM calls per turn:
  Every user query triggers two LLM calls — one to rewrite the question
  (contextualisation) and one to generate the answer (QA). On the first turn,
  chat_history is empty so the rewriter just echoes the question unchanged,
  but the API call still happens. If you want to cut cost, you can skip the
  rewriter when len(chat_history) == 0.
"""

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS

from src.config import config


# ── LLM factory ────────────────────────────────────────────────────────────

def _build_llm():
    """
    Return the appropriate LangChain LLM based on LLM_PROVIDER in config.

    Supported providers:
      openai — ChatOpenAI (direct OpenAI API)
      azure  — AzureChatOpenAI (Azure OpenAI Service / Azure AI Foundry)

    Learning note — why a factory function?
      Both providers implement the same LangChain BaseChatModel interface, so
      the rest of the chain is provider-agnostic. Adding a new provider (e.g.
      Anthropic Claude, Google Gemini) only requires adding a branch here and
      the corresponding config variables — zero changes to prompts or chains.
    """
    if config.LLM_PROVIDER == "azure":
        print(
            f"[Chain] Using Azure OpenAI — deployment: {config.AZURE_OPENAI_DEPLOYMENT}"
        )
        return AzureChatOpenAI(
            azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            temperature=config.LLM_TEMPERATURE,
        )

    print(f"[Chain] Using OpenAI — model: {config.LLM_MODEL}")
    return ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
    )


# ── Contextualisation prompt ────────────────────────────────────────────────
# [Learning] FAISS retrieves by vector similarity — it cannot resolve pronouns
# or conversational references ("it", "the second one", "explain more").
# This prompt instructs the LLM to rewrite such questions into standalone
# queries before they reach the retriever. Example:
#   Turn 2 input : "explain more about the second one"
#   Rewritten    : "explain the push-relabel algorithm"
# The rewriter uses the SAME LLM as the answerer, so choose a fast model
# (gpt-4o-mini is fine) to keep per-turn latency low.

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


# ── QA prompt ───────────────────────────────────────────────────────────────

# [Learning] _DOCUMENT_PROMPT formats each retrieved chunk before it is
# injected into the context block. The {filename} and {page_display} variables
# come from metadata set in document_loader.py. This is what gives the LLM
# the information it needs to write grounded citations like
# "According to thesis.pdf, page 5…"
_DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[Source: {filename}, Page {page_display}]\n{page_content}"
)

# [Learning] The system prompt has two jobs:
#   1. Ground the LLM in the retrieved context (prevents hallucination)
#   2. Tell it to cite sources (improves verifiability)
# The "I don't know" instruction is critical — without it, the LLM will often
# guess an answer rather than admit it lacks enough context.
_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based
strictly on the provided context extracted from one or more documents.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided document(s) to answer that."

Do not make up information. When citing information, include the source file
and page number shown in the context (e.g. "According to report.pdf, page 3…").

Context:
{context}
"""

# [Learning] MessagesPlaceholder("chat_history") injects all prior turns of
# the conversation into the prompt. The LLM can refer to them when answering
# follow-up questions ("as I mentioned…"). This grows with each turn — for
# very long conversations you would need to summarise or window the history.
_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# ── Public API ──────────────────────────────────────────────────────────────

def build_rag_chain(vector_store: FAISS, top_k: int = 6):
    """
    Construct a conversational RAG chain with in-session memory.

    The chain:
    1. Rewrites the user's question using prior chat history so retrieval
       is context-aware (history-aware retriever).
    2. Retrieves top-k diverse chunks from FAISS using MMR.
    3. Answers the question grounded in those chunks, with the full
       conversation history visible to the LLM.

    Args:
        vector_store: A populated FAISS vector store.
        top_k:        Number of document chunks to retrieve per query.

    Returns:
        A RunnableWithMessageHistory chain. Invoke it with:
            chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": "cli"}},
            )
        The return value is a dict; the answer is at result["answer"].
    """
    config.validate()

    llm = _build_llm()

    # [Learning] MMR retrieval vs pure similarity:
    #   Pure similarity returns the k most-similar chunks, which for a long
    #   document often means k near-duplicates from the same passage. MMR
    #   adds a diversity penalty: after selecting the first chunk, each
    #   subsequent chunk is chosen to be both relevant AND different from
    #   what is already selected. fetch_k is the candidate pool size — MMR
    #   re-ranks those candidates to pick the final k. A pool of 4× final k
    #   gives enough diversity without fetching too many vectors.
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 4},
    )

    # Step 1: retriever that understands prior turns
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, _CONTEXTUALIZE_PROMPT
    )

    # Step 2: QA chain — formats each chunk, injects into prompt, calls LLM
    # [Learning] "stuff" strategy = all chunks concatenated into one context
    # string. Simple and effective for k ≤ 8. For very long documents where
    # k must be large, consider map-reduce (summarise each chunk, then combine)
    # or refine (iteratively update an answer chunk by chunk).
    qa_chain = create_stuff_documents_chain(
        llm, _QA_PROMPT, document_prompt=_DOCUMENT_PROMPT
    )

    # Step 3: wire retriever → QA chain into the full RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Step 4: attach per-session in-memory message history
    # [Learning] store is a plain dict keyed by session_id, holding a
    # ChatMessageHistory per session. This is in-process memory — it is lost
    # when the process exits. For persistent or multi-user memory, back this
    # with a database (Redis, DynamoDB, PostgreSQL) using LangChain's
    # RedisChatMessageHistory or DynamoDBChatMessageHistory equivalents.
    store: dict = {}

    def _get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # [Learning] RunnableWithMessageHistory wraps the chain and automatically:
    #   1. Loads history for the given session_id before each call
    #   2. Passes it as chat_history to the prompt
    #   3. Appends the new (human, AI) turn to history after each call
    # The session_id is supplied at invoke time via config["configurable"],
    # which allows one chain object to serve multiple independent sessions.
    return RunnableWithMessageHistory(
        rag_chain,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
