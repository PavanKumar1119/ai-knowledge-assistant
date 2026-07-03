"""
utils/splitter.py
==================
Wraps LangChain's `RecursiveCharacterTextSplitter` to chunk ingested
documents into overlapping windows suitable for embedding + retrieval.

Chunking strategy:
    - RecursiveCharacterTextSplitter tries a cascade of separators
      (paragraph -> sentence -> word -> char) so chunks respect natural
      text boundaries as much as possible.
    - Overlap preserves context across chunk boundaries, which improves
      retrieval recall for answers that span a boundary.

Author: Senior AI Engineering Team
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from utils.helper import get_logger

logger = get_logger(__name__)


def get_text_splitter(
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> RecursiveCharacterTextSplitter:
    """
    Build a configured RecursiveCharacterTextSplitter instance.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        RecursiveCharacterTextSplitter: Configured splitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_documents(
    documents: List[Document],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> List[Document]:
    """
    Split a list of page-level Documents into smaller overlapping chunks.

    Each output chunk retains the original document's metadata
    (source filename, page number) so citations remain accurate
    even after splitting.

    Args:
        documents: List of Document objects (typically one per PDF page).
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List[Document]: Flattened list of chunked Document objects.
    """
    if not documents:
        logger.warning("split_documents called with an empty document list.")
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    logger.info(
        "Split %d source pages into %d chunks (chunk_size=%d, overlap=%d).",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
