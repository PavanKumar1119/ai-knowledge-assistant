"""
utils/vector_store.py
======================
Wraps a persistent ChromaDB collection used as the semantic vector
store for the RAG pipeline.

Responsibilities:
    - Create/load a persistent Chroma collection on disk (`chroma_db/`).
    - Add newly ingested + chunked documents to the collection.
    - Expose a LangChain-compatible retriever for the RAG chain.
    - Support clearing the knowledge base (delete + recreate collection).
    - Report basic statistics (document/chunk counts) for the UI.

Author: Senior AI Engineering Team
"""

from __future__ import annotations

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from config import settings, CHROMA_DIR
from utils.helper import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """
    Manages the lifecycle of a persistent Chroma vector store collection.

    This class is intentionally NOT a Streamlit-cached singleton because
    it holds mutable state (the underlying Chroma client) that must be
    re-creatable on demand (e.g. after "Clear Knowledge Base").
    """

    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = settings.collection_name,
        persist_directory: str = str(CHROMA_DIR),
    ) -> None:
        """
        Initialize the manager and load (or create) the Chroma collection.

        Args:
            embedding_function: The embeddings model used to vectorize text.
            collection_name: Name of the Chroma collection on disk.
            persist_directory: Directory where Chroma persists its data.
        """
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._store: Chroma = self._load_store()

    def _load_store(self) -> Chroma:
        """
        Instantiate the underlying Chroma vector store client.

        Returns:
            Chroma: LangChain Chroma vector store bound to the persistent directory.
        """
        logger.info(
            "Loading Chroma collection '%s' from '%s'.",
            self.collection_name,
            self.persist_directory,
        )
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )

    def add_documents(self, documents: List[Document]) -> int:
        """
        Embed and persist a batch of document chunks into the vector store.

        Args:
            documents: List of chunked Document objects to index.

        Returns:
            int: Number of chunks successfully added.
        """
        if not documents:
            logger.warning("add_documents called with an empty document list.")
            return 0

        self._store.add_documents(documents)
        logger.info("Added %d chunks to collection '%s'.", len(documents), self.collection_name)
        return len(documents)

    def get_retriever(self, k: int = settings.retriever_k) -> VectorStoreRetriever:
        """
        Build a semantic similarity retriever from the vector store.

        Args:
            k: Number of top-matching chunks to retrieve per query.

        Returns:
            VectorStoreRetriever: A LangChain-compatible retriever.
        """
        return self._store.as_retriever(search_type="similarity", search_kwargs={"k": k})

    def document_count(self) -> int:
        """
        Return the number of chunks currently stored in the collection.

        Returns:
            int: Count of vectors/chunks in the collection.
        """
        try:
            return self._store._collection.count()  # noqa: SLF001 - Chroma has no public count() API
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch document count: %s", exc)
            return 0

    def clear(self) -> None:
        """
        Delete all vectors from the collection and reinitialize an empty store.

        This effectively resets the knowledge base while keeping the same
        collection name and persist directory.
        """
        try:
            self._store.delete_collection()
            logger.info("Cleared Chroma collection '%s'.", self.collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error while clearing collection: %s", exc)
        finally:
            self._store = self._load_store()


def get_vector_store_manager(
    embedding_function: Embeddings,
    collection_name: str = settings.collection_name,
) -> VectorStoreManager:
    """
    Factory function returning a fresh VectorStoreManager instance.

    Kept outside Streamlit's cache decorators intentionally: the caller
    (app.py) manages this object's lifetime via `st.session_state` so it
    can be explicitly rebuilt after a "Clear Knowledge Base" action.

    Args:
        embedding_function: The embeddings model used to vectorize text.
        collection_name: Name of the Chroma collection on disk.

    Returns:
        VectorStoreManager: A ready-to-use manager instance.
    """
    return VectorStoreManager(embedding_function=embedding_function, collection_name=collection_name)
