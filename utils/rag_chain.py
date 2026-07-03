"""
utils/rag_chain.py
===================
Builds the LangChain Expression Language (LCEL) Retrieval-Augmented
Generation (RAG) pipeline that powers the assistant's answers.

Pipeline overview:

    user question
         │
         ▼
    retriever.invoke(question)  ──►  List[Document]  (semantic search in ChromaDB)
         │
         ▼
    format_docs()  ──►  concatenated context string (+ citation metadata kept separately)
         │
         ▼
    ChatPromptTemplate (system + chat_history + question)
         │
         ▼
    ChatGoogleGenerativeAI (Gemini 2.5 Flash)  ──►  streamed answer tokens

The chain is built with LCEL's `RunnablePassthrough` / `RunnableParallel`
composition so retrieval, prompting, and generation are wired together
declaratively and support both `.invoke()` and `.stream()`.

Author: Senior AI Engineering Team
"""

from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from utils.helper import get_logger
from utils.prompt import get_rag_prompt

logger = get_logger(__name__)


def format_docs(docs: Sequence[Document]) -> str:
    """
    Concatenate retrieved chunks into a single context string for the LLM.

    Each chunk is prefixed with a lightweight source tag so the model has
    an implicit sense of provenance, even though explicit citations are
    rendered separately in the UI from `docs` metadata.

    Args:
        docs: Retrieved Document chunks.

    Returns:
        str: Formatted context block.
    """
    formatted = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Excerpt {i} | {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def convert_history_to_messages(
    history: Sequence[Tuple[str, str]]
) -> List[BaseMessage]:
    """
    Convert a list of (role, content) tuples from Streamlit session state
    into LangChain BaseMessage objects for the prompt's chat_history slot.

    Args:
        history: Sequence of ("user"/"assistant", message_text) tuples.

    Returns:
        List[BaseMessage]: LangChain message objects.
    """
    messages: List[BaseMessage] = []
    for role, content in history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    """
    Instantiate the Gemini 2.5 Flash chat model.

    Args:
        api_key: Google Generative AI API key.

    Returns:
        ChatGoogleGenerativeAI: Configured chat model instance.
    """
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=api_key,
        temperature=settings.temperature,
        convert_system_message_to_human=False,
        streaming=True,
    )


def build_rag_chain(retriever: VectorStoreRetriever, api_key: str):
    """
    Compose the full LCEL RAG chain.

    The chain accepts a dict input: {"question": str, "chat_history": List[BaseMessage]}
    and returns a streamed string output.

    Args:
        retriever: A LangChain retriever bound to the ChromaDB collection.
        api_key: Google Generative AI API key used to authenticate Gemini calls.

    Returns:
        Runnable: An LCEL runnable chain supporting `.invoke()` and `.stream()`.
    """
    llm = get_llm(api_key)
    prompt = get_rag_prompt()

    chain = (
        RunnableParallel(
            {
                "context": (lambda x: x["question"]) | retriever | format_docs,
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def retrieve_source_documents(retriever: VectorStoreRetriever, question: str) -> List[Document]:
    """
    Independently run retrieval to obtain source Documents for citation
    rendering in the UI (kept separate from the LCEL generation chain so
    citations are available even while the answer is still streaming).

    Args:
        retriever: A LangChain retriever bound to the ChromaDB collection.
        question: The user's current question.

    Returns:
        List[Document]: Retrieved chunks with source/page metadata.
    """
    try:
        return retriever.invoke(question)
    except Exception as exc:  # noqa: BLE001
        logger.error("Retrieval failed for question '%s': %s", question, exc)
        return []


def stream_answer(
    chain,
    question: str,
    chat_history: Sequence[Tuple[str, str]],
) -> Iterator[str]:
    """
    Stream the assistant's answer token-by-token from the LCEL chain.

    Args:
        chain: The compiled LCEL RAG chain (from `build_rag_chain`).
        question: The user's current question.
        chat_history: Prior conversation turns as (role, content) tuples.

    Yields:
        str: Successive text chunks of the streamed answer.
    """
    messages = convert_history_to_messages(chat_history)
    payload = {"question": question, "chat_history": messages}

    try:
        for token in chain.stream(payload):
            yield token
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM streaming failed: %s", exc)
        yield (
            "\n\n⚠️ I ran into an error while generating a response. "
            "This is often caused by an invalid API key, a network issue, "
            "or a temporary Gemini service disruption. Please verify your "
            "API key and try again."
        )
