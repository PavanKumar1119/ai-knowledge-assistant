"""
app.py
======
Enterprise AI Knowledge Assistant - Streamlit Application Entry Point.

A production-style Retrieval-Augmented Generation (RAG) chat application
that lets users upload multiple PDF documents, builds a persistent
semantic knowledge base in ChromaDB, and answers questions grounded in
that knowledge base using Gemini 2.5 Flash via LangChain's LCEL.

Run locally:
    streamlit run app.py

Author: Pavan
"""

from __future__ import annotations

from typing import List

import streamlit as st

from config import settings
from utils.embeddings import get_embedding_model
from utils.helper import get_logger, format_file_size, truncate_text, validate_api_key
from utils.loader import load_multiple_pdfs, LoadResult
from utils.rag_chain import build_rag_chain, retrieve_source_documents, stream_answer
from utils.splitter import split_documents
from utils.vector_store import get_vector_store_manager

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page configuration - must be the first Streamlit call.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=settings.app_title,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS - dark, ChatGPT-style responsive theme.
# ---------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Inject custom CSS for a polished, dark, ChatGPT-style UI."""
    st.markdown(
        """
        <style>
        /* ---- Global ---- */
        .stApp {
            background-color: #0f1117;
            color: #e6e6e6;
        }
        [data-testid="stSidebar"] {
            background-color: #14161f;
            border-right: 1px solid #2a2d3a;
        }
        /* ---- Headings ---- */
        h1, h2, h3, h4 {
            color: #f5f5f7;
            font-family: 'Segoe UI', sans-serif;
        }
        /* ---- Chat bubbles ---- */
        [data-testid="stChatMessage"] {
            background-color: #1a1d29;
            border-radius: 14px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
            border: 1px solid #262a3a;
        }
        /* ---- Buttons ---- */
        div.stButton > button {
            border-radius: 10px;
            border: 1px solid #3a3f55;
            background-color: #1f2333;
            color: #e6e6e6;
            font-weight: 500;
            transition: all 0.15s ease-in-out;
        }
        div.stButton > button:hover {
            background-color: #2c3350;
            border-color: #5865f2;
            color: #ffffff;
        }
        /* ---- Metrics ---- */
        [data-testid="stMetricValue"] {
            color: #7c8cff;
        }
        /* ---- Badges for citations ---- */
        .citation-badge {
            display: inline-block;
            background-color: #1f2333;
            border: 1px solid #3a3f55;
            border-radius: 8px;
            padding: 2px 10px;
            margin: 3px 4px 0 0;
            font-size: 0.78rem;
            color: #9aa5ff;
        }
        .app-subtitle {
            color: #9aa0b5;
            font-size: 0.95rem;
            margin-top: -8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize all required keys in `st.session_state` exactly once."""
    defaults = {
        "chat_history": [],          # List[Tuple[str, str]] -> ("user"/"assistant", content)
        "vector_store_manager": None,
        "rag_chain": None,
        "processed_files": [],       # List[str] filenames indexed into the KB
        "total_chunks": 0,
        "last_upload_results": [],   # List[LoadResult] from most recent processing run
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def ensure_vector_store() -> None:
    """
    Lazily create the VectorStoreManager (and its dependent RAG chain)
    on first use, storing it in session_state so it persists across
    Streamlit re-runs within the same browser session.
    """
    if st.session_state.vector_store_manager is None:
        embedding_model = get_embedding_model()
        st.session_state.vector_store_manager = get_vector_store_manager(embedding_model)
        st.session_state.total_chunks = st.session_state.vector_store_manager.document_count()


def rebuild_rag_chain() -> None:
    """Rebuild the LCEL RAG chain bound to the current retriever + API key."""
    if not validate_api_key(settings.google_api_key):
        st.session_state.rag_chain = None
        return
    retriever = st.session_state.vector_store_manager.get_retriever()
    st.session_state.rag_chain = build_rag_chain(retriever, settings.google_api_key)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    """Render the professional sidebar: branding, upload, stats, controls."""
    with st.sidebar:
        st.markdown("## 🧠 Knowledge Assistant")
        st.markdown(
            '<p class="app-subtitle">Enterprise-grade RAG over your PDF documents</p>',
            unsafe_allow_html=True,
        )
        st.divider()

        # --- API key status ---
        if validate_api_key(settings.google_api_key):
            st.success("Gemini API key detected ✅", icon="🔑")
        else:
            st.error(
                "No valid `GOOGLE_API_KEY` found. Add it to a `.env` file "
                "(see `.env.example`) or Streamlit secrets.",
                icon="🚫",
            )

        st.divider()
        st.markdown("### 📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Max recommended size per file: {settings.max_upload_size_mb} MB",
        )

        process_clicked = st.button("⚙️ Process Documents", use_container_width=True, type="primary")

        if process_clicked:
            handle_processing(uploaded_files)

        # --- Uploaded / indexed file list ---
        if st.session_state.processed_files:
            st.markdown("### 📚 Indexed Files")
            for fname in st.session_state.processed_files:
                st.markdown(f"- 📄 {truncate_text(fname, 40)}")

        # --- Last run report (failures etc.) ---
        render_last_upload_report()

        st.divider()
        st.markdown("### 📊 Knowledge Base Stats")
        ensure_vector_store()
        col1, col2 = st.columns(2)
        col1.metric("Indexed Files", len(st.session_state.processed_files))
        col2.metric("Chunks Stored", st.session_state.total_chunks)

        st.divider()
        st.markdown("### 🛠️ Controls")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with c2:
            if st.button("🗑️ Clear KB", use_container_width=True):
                clear_knowledge_base()
                st.rerun()

        st.divider()
        st.caption(
            f"Model: `{settings.llm_model}` · Embeddings: `{settings.embedding_model.split('/')[-1]}`"
        )
        st.caption("Built with LangChain LCEL · ChromaDB · Streamlit")


def render_last_upload_report() -> None:
    """Display success/failure feedback for the most recent processing run."""
    results: List[LoadResult] = st.session_state.last_upload_results
    if not results:
        return

    failures = [r for r in results if not r.success]
    successes = [r for r in results if r.success]

    if successes:
        st.success(f"{len(successes)} file(s) indexed successfully.", icon="✅")
    if failures:
        with st.expander(f"⚠️ {len(failures)} file(s) had issues", expanded=False):
            for r in failures:
                st.warning(f"**{r.filename}**: {r.error_message}")


# ---------------------------------------------------------------------------
# Processing pipeline (ingest -> split -> embed -> store)
# ---------------------------------------------------------------------------
def handle_processing(uploaded_files) -> None:
    """
    Orchestrate the full ingestion pipeline for newly uploaded files.

    Steps: load PDFs -> split into chunks -> embed & persist to ChromaDB
    -> refresh RAG chain -> update session_state stats.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects (may be empty/None).
    """
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF file before processing.", icon="⚠️")
        return

    with st.spinner("📄 Reading and parsing PDF files..."):
        documents, results = load_multiple_pdfs(uploaded_files)
        st.session_state.last_upload_results = results

    if not documents:
        st.sidebar.error("No valid text could be extracted from the uploaded file(s).", icon="🚫")
        return

    with st.spinner("✂️ Splitting documents into chunks..."):
        chunks = split_documents(documents)

    with st.spinner("🧬 Generating embeddings and updating the knowledge base..."):
        ensure_vector_store()
        added = st.session_state.vector_store_manager.add_documents(chunks)
        st.session_state.total_chunks = st.session_state.vector_store_manager.document_count()

    newly_indexed = {r.filename for r in results if r.success}
    for fname in newly_indexed:
        if fname not in st.session_state.processed_files:
            st.session_state.processed_files.append(fname)

    rebuild_rag_chain()
    st.sidebar.success(f"Indexed {added} chunks from {len(newly_indexed)} file(s).", icon="🎉")


def clear_knowledge_base() -> None:
    """Wipe the ChromaDB collection and reset all related session state."""
    ensure_vector_store()
    st.session_state.vector_store_manager.clear()
    st.session_state.processed_files = []
    st.session_state.total_chunks = 0
    st.session_state.last_upload_results = []
    st.session_state.rag_chain = None
    st.session_state.chat_history = []
    logger.info("Knowledge base cleared by user.")


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
def render_chat_history() -> None:
    """Render all prior turns from `st.session_state.chat_history`."""
    for role, content in st.session_state.chat_history:
        avatar = "🧑‍💼" if role == "user" else "🧠"
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)


def render_citations(source_docs) -> None:
    """
    Render source citation badges (filename + page number) below an answer.

    Args:
        source_docs: List of retrieved Document chunks used to ground the answer.
    """
    if not source_docs:
        return
    seen = set()
    badges = []
    for doc in source_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            badges.append(f'<span class="citation-badge">📄 {source} · p.{page}</span>')
    if badges:
        st.markdown("**Sources:**<br>" + "".join(badges), unsafe_allow_html=True)


def handle_chat_input() -> None:
    """Render the chat input box and orchestrate a single Q&A turn."""
    placeholder = (
        "Ask a question about your uploaded documents..."
        if st.session_state.processed_files
        else "Upload and process documents first, then ask a question..."
    )
    user_question = st.chat_input(placeholder)

    if not user_question:
        return

    if not validate_api_key(settings.google_api_key):
        st.error(
            "Cannot generate a response: no valid Gemini API key configured. "
            "Please set `GOOGLE_API_KEY` in your `.env` file or Streamlit secrets.",
            icon="🚫",
        )
        return

    if not st.session_state.processed_files:
        st.warning(
            "Your knowledge base is empty. Upload and process at least one PDF "
            "before asking a question.",
            icon="⚠️",
        )
        return

    ensure_vector_store()
    if st.session_state.rag_chain is None:
        rebuild_rag_chain()

    st.session_state.chat_history.append(("user", user_question))
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_question)

    retriever = st.session_state.vector_store_manager.get_retriever()
    source_docs = retrieve_source_documents(retriever, user_question)

    with st.chat_message("assistant", avatar="🧠"):
        try:
            response_text = st.write_stream(
                stream_answer(
                    st.session_state.rag_chain,
                    user_question,
                    st.session_state.chat_history[:-1],  # exclude the just-added user turn
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error during answer generation: %s", exc)
            response_text = (
                "⚠️ Something went wrong while generating a response. "
                "Please check your API key and network connection, then try again."
            )
            st.error(response_text)

        render_citations(source_docs)

    st.session_state.chat_history.append(("assistant", response_text))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Application entry point: wires up session state, sidebar, and chat UI."""
    inject_custom_css()
    init_session_state()

    st.markdown("# 🧠 Enterprise AI Knowledge Assistant")
    st.markdown(
        '<p class="app-subtitle">Ask questions across all of your organization\'s '
        "PDF knowledge, grounded and cited, powered by Gemini 2.5 Flash + RAG.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    render_sidebar()

    chat_container = st.container()
    with chat_container:
        render_chat_history()

    handle_chat_input()


if __name__ == "__main__":
    main()
