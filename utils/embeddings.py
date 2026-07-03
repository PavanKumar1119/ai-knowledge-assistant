"""
utils/embeddings.py
====================
Provides a cached factory for the HuggingFace sentence-transformers
embedding model used to vectorize document chunks and queries.

The model is loaded once per Streamlit server process via
`st.cache_resource`, avoiding an expensive reload (~100-400ms + model
weights download on cold start) on every UI re-run.

Author: Senior AI Engineering Team
"""

from __future__ import annotations

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from utils.helper import get_logger

logger = get_logger(__name__)


@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str = settings.embedding_model) -> HuggingFaceEmbeddings:
    """
    Instantiate (and cache) the HuggingFace embedding model.

    Uses `sentence-transformers/all-MiniLM-L6-v2` by default: a compact,
    fast, high-quality general-purpose embedding model (384 dimensions)
    well suited for semantic search over enterprise documents.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        HuggingFaceEmbeddings: A LangChain-compatible embeddings object.
    """
    logger.info("Loading embedding model '%s' (this happens once per session)...", model_name)
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model loaded successfully.")
    return model
