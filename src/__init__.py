"""
Company RAG System - Retrieval-Augmented Generation for Employee Knowledge Access

This package provides RAG functionality for company document processing and employee queries.
Supports CSV, PDF, TXT files for company policies, procedures, and technical documentation.
"""

__version__ = "1.0.0"
__author__ = "Company RAG Team"

# Make key classes available at package level for easy imports
from .document_manager import DocumentManager
from .models import EmbeddingModel, LLMModel
from .rag_pipeline import rag_pipeline
from .vector_store import setup_persistent_chroma, update_vector_database

__all__ = [
    "DocumentManager",
    "EmbeddingModel",
    "LLMModel",
    "setup_persistent_chroma",
    "update_vector_database",
    "rag_pipeline",
]
