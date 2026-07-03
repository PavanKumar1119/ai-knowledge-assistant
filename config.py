"""
config.py
=========
Centralized configuration for the Enterprise AI Knowledge Assistant.

All tunable parameters, paths, and environment-derived settings live here
so that the rest of the codebase never hardcodes "magic values". This
follows the 12-factor app principle of externalized configuration.

Author: Senior AI Engineering Team
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env (if present). In production
# (e.g. Streamlit Community Cloud), secrets are injected via st.secrets
# or platform environment variables instead of a physical .env file.
# ---------------------------------------------------------------------------
load_dotenv(override=False)

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
CHROMA_DIR: Path = BASE_DIR / "chroma_db"
ASSETS_DIR: Path = BASE_DIR / "assets"

for _dir in (DATA_DIR, CHROMA_DIR, ASSETS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def _get_api_key() -> str:
    """
    Resolve the Google Generative AI API key from multiple possible sources.

    Priority:
        1. Streamlit secrets (st.secrets) - used on Streamlit Community Cloud
        2. Environment variable GOOGLE_API_KEY - used locally via .env

    Returns:
        str: The resolved API key, or an empty string if not found.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st  # local import to avoid hard dependency at import time

            api_key = st.secrets.get("GOOGLE_API_KEY", "")  # type: ignore[assignment]
        except Exception:
            # st.secrets raises if no secrets.toml exists; that's fine, fall through.
            api_key = ""
    return api_key or ""


@dataclass(frozen=True)
class Settings:
    """
    Immutable application settings object.

    Attributes:
        google_api_key: API key used to authenticate with Gemini.
        llm_model: Gemini model identifier used for generation.
        embedding_model: HuggingFace sentence-transformers model name.
        chunk_size: Max characters per text chunk during splitting.
        chunk_overlap: Character overlap between consecutive chunks.
        retriever_k: Number of top-k chunks retrieved per query.
        collection_name: ChromaDB collection name.
        temperature: LLM sampling temperature (0 = deterministic).
        max_upload_size_mb: Soft limit for a single uploaded PDF, in MB.
        app_title: Display title of the Streamlit application.
    """

    google_api_key: str = field(default_factory=_get_api_key)
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    retriever_k: int = int(os.getenv("RETRIEVER_K", "4"))
    collection_name: str = os.getenv("COLLECTION_NAME", "enterprise_knowledge_base")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    app_title: str = "Enterprise AI Knowledge Assistant"


# Singleton settings instance used across the application.
settings = Settings()
