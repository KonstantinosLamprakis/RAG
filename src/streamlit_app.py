"""Streamlit UI for company knowledge RAG system"""

from datetime import datetime

import streamlit as st

from config import (
    COMPANY_KNOWLEDGE_COLLECTION,
    DATA_DIRECTORY,
    FILE_METADATA_PATH,
    MAX_RELEVANCE_DISTANCE,
    PERSISTENT_DB_PATH,
    TOP_K_RESULTS,
    EmbeddingsType,
    LLMType,
)
from document_manager import DocumentManager
from models import EmbeddingModel, LLMModel
from rag_pipeline import rag_pipeline, validate_environment
from vector_store import setup_persistent_chroma, update_vector_database


def run_streamlit_app():
    llm_type = LLMType.OPENAI.value
    embedding_type = EmbeddingsType.OPENAI.value

    if not validate_environment(llm_type):
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.set_page_config(page_title="Company Knowledge RAG", layout="wide")
    st.title("🏢 Company Knowledge RAG System with Persistent Storage")

    if "doc_manager" not in st.session_state:
        st.session_state.doc_manager = DocumentManager(
            DATA_DIRECTORY, FILE_METADATA_PATH
        )

    st.sidebar.title("Vector Database Management")
    if st.sidebar.button("🔄 Update Vector Database"):
        with st.spinner("Checking for file changes..."):
            embedding_model = EmbeddingModel(embedding_type)
            collection = setup_persistent_chroma(embedding_model)

            processed_count = update_vector_database(
                collection,
                st.session_state.doc_manager,
            )

            if processed_count > 0:
                st.sidebar.success(
                    f"Updated {processed_count} files in vector database!"
                )
                # Force reinitialization
                if "initialized" in st.session_state:
                    del st.session_state["initialized"]
            else:
                st.sidebar.info("Vector database is already up to date!")

    if st.sidebar.button("📊 Show Database Stats"):
        try:
            embedding_model = EmbeddingModel(embedding_type)
            collection = setup_persistent_chroma(embedding_model)
            count = collection.count()
            st.sidebar.write(f"**Total vectors in database:** {count}")

            tracked_files = len(st.session_state.doc_manager.file_metadata)
            st.sidebar.write(f"**Tracked files:** {tracked_files}")

            changed_files = st.session_state.doc_manager.get_changed_files()
            st.sidebar.write(f"**Files needing updates:** {len(changed_files)}")

        except Exception as e:
            st.sidebar.error(f"Error getting database stats: {e}")

    if "initialized" not in st.session_state or st.sidebar.button(
        "🚀 Initialize RAG System"
    ):
        st.session_state.initialized = False

        with st.spinner("Initializing RAG system with persistent storage..."):
            try:
                st.session_state.llm_model = LLMModel(llm_type)
                st.session_state.embedding_model = EmbeddingModel(embedding_type)
                st.session_state.collection = setup_persistent_chroma(
                    st.session_state.embedding_model
                )

                # Auto-update vector database on initialization
                processed_count = update_vector_database(
                    st.session_state.collection,
                    st.session_state.doc_manager,
                )

                vector_count = st.session_state.collection.count()
                st.session_state.initialized = True

                st.success(
                    f"RAG system initialized! Database contains {vector_count} vectors."
                )
                if processed_count > 0:
                    st.info(
                        f"Processed {processed_count} changed files during initialization."
                    )

            except Exception as e:
                st.error(f"Error initializing RAG system: {e}")
                return

    if not st.session_state.get("initialized", False):
        st.info("Please initialize the RAG system to start querying company knowledge.")
        return

    with st.expander("🗄️ Vector Database Information", expanded=False):
        try:
            vector_count = st.session_state.collection.count()
            st.write(f"**Total vectors in database:** {vector_count}")
            st.write(f"**Database path:** {PERSISTENT_DB_PATH}")
            st.write(f"**Collection name:** {COMPANY_KNOWLEDGE_COLLECTION}")

            tracked_files = st.session_state.doc_manager.file_metadata
            if tracked_files:
                st.write("**Tracked Files:**")
                for file_path, metadata in tracked_files.items():
                    mod_time = datetime.fromtimestamp(
                        metadata["modification_time"]
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    st.write(f"📄 {metadata['filename']} (Modified: {mod_time})")
        except Exception as e:
            st.error(f"Error displaying database info: {e}")

    st.markdown("### 💬 Ask Questions About Company Knowledge")
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the company policies? How does the technical procedure work?",
    )

    if query:
        with st.spinner("Searching company knowledge base..."):
            try:
                response, references, augmented_prompt = rag_pipeline(
                    query,
                    st.session_state.collection,
                    st.session_state.llm_model,
                    TOP_K_RESULTS,
                    MAX_RELEVANCE_DISTANCE,
                )

                st.session_state.chat_history.append({"role": "user", "content": query})
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📝 Response")
                    st.write(response)

                with col2:
                    st.markdown("### 📚 References Used")
                    for ref in references:
                        st.write(f"📄 {ref[:200]}...")

                with st.expander("🔧 Technical Details", expanded=False):
                    st.markdown("### Augmented Prompt")
                    st.code(augmented_prompt)

                    st.markdown("### Model Configuration")
                    st.write(f"LLM Model: {llm_type.upper()}")
                    st.write(f"Embedding Model: {embedding_type.upper()}")

                with st.expander("💬 Chat History", expanded=False):
                    for msg in st.session_state.chat_history:
                        if msg["role"] == "user":
                            st.write(f"👤 **Employee:** {msg['content']}")
                        else:
                            st.write(f"🤖 **Company Assistant:** {msg['content']}")

            except Exception as e:
                st.error(f"Error processing query: {e}")
