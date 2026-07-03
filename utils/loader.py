"""
utils/loader.py
================
Handles ingestion of PDF documents uploaded via the Streamlit UI.

Responsibilities:
    - Persist uploaded files (in-memory BytesIO from Streamlit) to disk
      so that PyPDFLoader can read them.
    - Parse PDFs into LangChain `Document` objects with page-level
      metadata (source filename + page number) for later citation.
    - Gracefully handle corrupted / invalid / empty PDF files without
      crashing the whole ingestion batch.

Author: Senior AI Engineering Team
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from config import settings
from utils.helper import get_logger

logger = get_logger(__name__)


@dataclass
class LoadResult:
    """
    Result of loading a single uploaded PDF file.

    Attributes:
        filename: Original filename of the uploaded PDF.
        documents: List of parsed page-level Document objects (empty on failure).
        success: Whether parsing succeeded.
        error_message: Human-readable error description if success is False.
        num_pages: Number of pages successfully extracted.
    """

    filename: str
    documents: List[Document]
    success: bool
    error_message: str = ""
    num_pages: int = 0


def _save_uploaded_file(uploaded_file) -> Path:
    """
    Persist a Streamlit `UploadedFile` object to the local `data/` directory.

    Args:
        uploaded_file: A Streamlit UploadedFile instance (has `.name`, `.getbuffer()`).

    Returns:
        Path: Filesystem path where the file was written.
    """
    from config import DATA_DIR  # local import keeps module import order simple

    destination = DATA_DIR / uploaded_file.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return destination


def load_single_pdf(uploaded_file) -> LoadResult:
    """
    Load and parse a single uploaded PDF file into page-level Documents.

    Each resulting Document's metadata is normalized to contain:
        - "source": the original filename (not the full disk path)
        - "page": 1-indexed page number for human-friendly citation

    Args:
        uploaded_file: A Streamlit UploadedFile instance.

    Returns:
        LoadResult: Structured result including success/failure state.
    """
    filename = getattr(uploaded_file, "name", "unknown.pdf")

    if not filename.lower().endswith(".pdf"):
        msg = f"'{filename}' is not a PDF file and was skipped."
        logger.warning(msg)
        return LoadResult(filename=filename, documents=[], success=False, error_message=msg)

    try:
        file_path = _save_uploaded_file(uploaded_file)
    except Exception as exc:  # noqa: BLE001 - we want to surface any IO error to the UI
        msg = f"Failed to save '{filename}' to disk: {exc}"
        logger.error(msg)
        return LoadResult(filename=filename, documents=[], success=False, error_message=msg)

    try:
        loader = PyPDFLoader(str(file_path))
        raw_docs = loader.load()
    except Exception as exc:  # noqa: BLE001 - PyPDF raises various exception types on bad PDFs
        msg = f"'{filename}' could not be parsed (corrupted or invalid PDF): {exc}"
        logger.error(msg)
        return LoadResult(filename=filename, documents=[], success=False, error_message=msg)

    if not raw_docs:
        msg = f"'{filename}' appears to be empty (no extractable text)."
        logger.warning(msg)
        return LoadResult(filename=filename, documents=[], success=False, error_message=msg)

    normalized_docs: List[Document] = []
    for i, doc in enumerate(raw_docs):
        page_number = doc.metadata.get("page", i)
        # PyPDFLoader returns 0-indexed pages; normalize to 1-indexed for display.
        doc.metadata = {
            "source": filename,
            "page": int(page_number) + 1,
        }
        if doc.page_content and doc.page_content.strip():
            normalized_docs.append(doc)

    if not normalized_docs:
        msg = f"'{filename}' contains no extractable text (possibly a scanned image PDF)."
        logger.warning(msg)
        return LoadResult(filename=filename, documents=[], success=False, error_message=msg)

    logger.info("Successfully loaded '%s' with %d pages.", filename, len(normalized_docs))
    return LoadResult(
        filename=filename,
        documents=normalized_docs,
        success=True,
        num_pages=len(normalized_docs),
    )


def load_multiple_pdfs(uploaded_files: Sequence) -> Tuple[List[Document], List[LoadResult]]:
    """
    Load multiple uploaded PDF files, collecting successes and failures.

    Args:
        uploaded_files: Sequence of Streamlit UploadedFile instances.

    Returns:
        Tuple[List[Document], List[LoadResult]]:
            - Flat list of all successfully parsed Documents (across all files).
            - Per-file LoadResult objects (for UI reporting of failures/successes).
    """
    all_documents: List[Document] = []
    results: List[LoadResult] = []

    if not uploaded_files:
        logger.warning("load_multiple_pdfs called with an empty file list.")
        return all_documents, results

    for uploaded_file in uploaded_files:
        result = load_single_pdf(uploaded_file)
        results.append(result)
        if result.success:
            all_documents.extend(result.documents)

    logger.info(
        "Ingestion complete: %d/%d files succeeded, %d total pages extracted.",
        sum(1 for r in results if r.success),
        len(results),
        len(all_documents),
    )
    return all_documents, results
