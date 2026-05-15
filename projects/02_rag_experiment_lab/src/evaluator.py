"""
src/evaluator.py
RAG evaluation metrics — measures retrieval quality, answer faithfulness,
and answer relevance using LLM-as-judge scoring.

This is the core new concept in Project 02. In Project 01, you could only
judge answer quality by reading it yourself. Here, you get quantifiable
scores that let you compare configurations objectively.

Learning note — LLM-as-judge:
  Using an LLM to evaluate another LLM's output is a well-established
  technique (see: "Judging LLM-as-a-Judge" paper, LMSYS arena). It works
  because evaluation is easier than generation — the judge LLM only needs
  to verify claims against evidence, not create new content.

  Caveats:
  - The judge can be wrong (especially for nuanced domain-specific questions)
  - GPT-4o is a better judge than GPT-4o-mini; use the best model you can afford
  - Always spot-check LLM-judge scores against your own human judgment

Learning note — metrics explained:
  1. Context Relevance: Are the retrieved chunks actually relevant to the question?
     Low score = retrieval problem (fix embeddings, chunk size, or search strategy)

  2. Faithfulness: Is the answer grounded in the retrieved context?
     Low score = hallucination problem (fix prompt, lower temperature, or add reranker)

  3. Answer Relevance: Does the answer actually address the question?
     Low score = prompt problem or LLM quality issue

  These three metrics together diagnose WHERE in the RAG pipeline the problem
  is — retrieval, grounding, or generation. This is the key diagnostic skill
  in RAG engineering.

Learning note — cost:
  Each metric requires one LLM call to the judge model. Evaluating N questions
  with 3 metrics = 3N LLM calls. For a 20-question test set, that's 60 calls
  (~$0.01 with GPT-4o-mini, ~$0.30 with GPT-4o). Budget accordingly.
"""

from typing import List

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

from src.config import config


def _get_judge_llm():
    """
    Return the LLM used for evaluation judging.

    Uses the same provider as the main LLM but always uses the configured model.
    For best evaluation quality, set LLM_MODEL=gpt-4o in your .env when running
    evaluations, even if you use gpt-4o-mini for the RAG chain itself.
    """
    if config.LLM_PROVIDER == "azure":
        return AzureChatOpenAI(
            azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            temperature=0.0,
        )
    return ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0.0,
        api_key=config.OPENAI_API_KEY,
    )


def score_context_relevance(question: str, docs: List[Document]) -> float:
    """
    Score how relevant the retrieved chunks are to the question (0.0-1.0).

    Asks the judge LLM to rate each chunk's relevance, then averages.

    Learning note — why per-chunk scoring?
      Scoring each chunk individually (then averaging) reveals whether the
      retriever is fetching irrelevant chunks. A low average with some high
      scores suggests the top results are good but lower-ranked chunks are
      noise — reduce top_k or enable reranking.
    """
    if not docs:
        return 0.0

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an evaluation judge. Given a question and a text chunk, "
         "rate the relevance of the chunk to answering the question.\n"
         "Reply with ONLY a number from 0.0 to 1.0 where:\n"
         "  0.0 = completely irrelevant\n"
         "  0.5 = somewhat relevant\n"
         "  1.0 = highly relevant and directly answers the question\n"
         "Reply with just the number, nothing else."),
        ("human", "Question: {question}\n\nText chunk:\n{chunk}"),
    ])

    llm = _get_judge_llm()
    chain = prompt | llm

    scores = []
    for doc in docs:
        try:
            result = chain.invoke({
                "question": question,
                "chunk": doc.page_content[:1500],  # limit chunk length for judge
            })
            score = float(result.content.strip())
            scores.append(max(0.0, min(1.0, score)))
        except (ValueError, TypeError):
            scores.append(0.5)  # default if parsing fails

    return round(sum(scores) / len(scores), 3)


def score_faithfulness(question: str, answer: str, docs: List[Document]) -> float:
    """
    Score whether the answer is grounded in the retrieved context (0.0-1.0).

    A faithful answer only contains claims supported by the context. An
    unfaithful answer hallucinates information not present in any chunk.

    Learning note — this is the hallucination detector:
      If faithfulness is low but context relevance is high, the retriever
      is doing its job but the LLM is ignoring the context and making things
      up. Fixes: stricter system prompt, lower temperature, or a more
      instruction-following model.
    """
    if not docs or not answer:
        return 0.0

    context_text = "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an evaluation judge. Given a question, an answer, and the "
         "source context that was provided to the answering model, rate the "
         "faithfulness of the answer.\n\n"
         "Faithfulness means: every claim in the answer is supported by the "
         "provided context. The answer should not contain information that "
         "cannot be found in or reasonably inferred from the context.\n\n"
         "Reply with ONLY a number from 0.0 to 1.0 where:\n"
         "  0.0 = answer contains many claims not in the context (hallucination)\n"
         "  0.5 = answer is partially grounded but has some unsupported claims\n"
         "  1.0 = answer is fully grounded in the provided context\n"
         "Reply with just the number, nothing else."),
        ("human",
         "Question: {question}\n\n"
         "Answer: {answer}\n\n"
         "Context provided to the model:\n{context}"),
    ])

    llm = _get_judge_llm()
    chain = prompt | llm

    try:
        result = chain.invoke({
            "question": question,
            "answer": answer,
            "context": context_text[:4000],  # limit context for judge
        })
        score = float(result.content.strip())
        return round(max(0.0, min(1.0, score)), 3)
    except (ValueError, TypeError):
        return 0.5


def score_answer_relevance(question: str, answer: str) -> float:
    """
    Score whether the answer actually addresses the question (0.0-1.0).

    This metric is independent of the context — it purely measures whether
    the answer is helpful for the question asked.

    Learning note — when this is low:
      If answer relevance is low but faithfulness is high, the model is
      faithfully summarising the context but the context doesn't contain
      what the user needs. This indicates a retrieval problem, not a
      generation problem — the right fix is better chunking or embeddings,
      not a better prompt.
    """
    if not answer:
        return 0.0

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an evaluation judge. Given a question and an answer, "
         "rate how well the answer addresses the question.\n\n"
         "Reply with ONLY a number from 0.0 to 1.0 where:\n"
         "  0.0 = answer is completely off-topic or useless\n"
         "  0.5 = answer partially addresses the question\n"
         "  1.0 = answer fully and helpfully addresses the question\n"
         "Reply with just the number, nothing else."),
        ("human", "Question: {question}\n\nAnswer: {answer}"),
    ])

    llm = _get_judge_llm()
    chain = prompt | llm

    try:
        result = chain.invoke({
            "question": question,
            "answer": answer,
        })
        score = float(result.content.strip())
        return round(max(0.0, min(1.0, score)), 3)
    except (ValueError, TypeError):
        return 0.5


def evaluate_single(
    question: str,
    answer: str,
    docs: List[Document],
    expected_answer: str = None,
) -> dict:
    """
    Run all evaluation metrics on a single question-answer pair.

    Args:
        question:        The user's question.
        answer:          The RAG chain's answer.
        docs:            Retrieved documents used to generate the answer.
        expected_answer: Optional ground-truth answer (for future metrics).

    Returns:
        dict with keys: context_relevance, faithfulness, answer_relevance
    """
    return {
        "context_relevance": score_context_relevance(question, docs),
        "faithfulness": score_faithfulness(question, answer, docs),
        "answer_relevance": score_answer_relevance(question, answer),
    }
