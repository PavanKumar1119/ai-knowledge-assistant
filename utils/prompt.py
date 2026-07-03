"""
utils/prompt.py
================
Defines the prompt templates used by the RAG pipeline.

The system prompt instructs the LLM to:
    - Answer strictly using the provided context (grounded generation).
    - Politely decline when the answer is not present in the context.
    - Maintain a professional, enterprise-assistant tone.
    - Consider prior conversation turns for follow-up questions.

Author: Senior AI Engineering Team
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are an Enterprise AI Knowledge Assistant. Your job is to answer \
questions accurately and concisely using ONLY the information provided in the \
context below, which was retrieved from the organization's internal documents.

Guidelines:
- Base your answer strictly on the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer, clearly say so \
instead of guessing or fabricating an answer.
- Be concise, professional, and well-structured. Use bullet points or short \
paragraphs where helpful.
- If relevant, synthesize information from multiple parts of the context.
- Do not mention "the context" explicitly in your answer; just answer naturally \
as a knowledgeable assistant.

Context:
{context}
"""


def get_rag_prompt() -> ChatPromptTemplate:
    """
    Build the chat prompt template used by the RAG chain.

    Includes a placeholder for prior conversation turns (`chat_history`)
    so the assistant can handle natural follow-up questions, and a
    `{question}` slot for the current user query.

    Returns:
        ChatPromptTemplate: The composed prompt template.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
