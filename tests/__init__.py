"""
Test suite for Company RAG System

Tests cover:
- Document processing (CSV, PDF, TXT company files)
- Vector database operations (ChromaDB)
- Employee query handling
- RAG pipeline integration
- Company knowledge retrieval
"""

# Test configuration
import os
import sys

# Ensure src is importable during testing
TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
