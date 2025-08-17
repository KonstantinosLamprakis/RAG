import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path for RAG imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def temp_dir():
    """Create temporary directory for company RAG system tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def temp_data_dir(temp_dir):
    """Create temporary data directory with sample company documents for employee queries"""
    data_dir = os.path.join(temp_dir, "company_data")
    os.makedirs(data_dir)
    
    # Company employee CSV - for org chart and employee directory queries
    csv_content = (
        "employee_id,name,department,role,manager,email\n"
        "1,John Doe,Engineering,Senior Developer,Jane Smith,john.doe@company.com\n"
        "2,Jane Smith,Engineering,Team Lead,Bob Wilson,jane.smith@company.com"
    )
    with open(os.path.join(data_dir, "employees.csv"), "w") as f:
        f.write(csv_content)
    
    # Company security policy TXT - for employee security procedure queries
    txt_content = (
        "Company Policy: All employees must follow security protocols including "
        "two-factor authentication and incident reporting procedures."
    )
    with open(os.path.join(data_dir, "policy.txt"), "w") as f:
        f.write(txt_content)
    
    return data_dir

@pytest.fixture
def temp_metadata_file(temp_dir):
    """Create temporary metadata file for company document tracking"""
    metadata_path = os.path.join(temp_dir, "company_metadata.json")
    
    # Create metadata that matches the test expectation
    metadata = {
        "/path/to/old_file.txt": {
            "filename": "old_file.txt",
            "modification_time": 1234567890,
            "size": 100,
            "hash": "abcd1234"
        }
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    return metadata_path

@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for company document similarity search"""
    mock = Mock()
    mock.embedding_fn = Mock()
    mock.embedding_fn.return_value = [[0.1, 0.2, 0.3] * 512]  # Mock 1536-dim OpenAI embedding
    mock.embed_query = Mock(return_value=[0.1] * 1536)
    mock.embed_documents = Mock(return_value=[[0.1] * 1536])
    return mock

@pytest.fixture
def mock_llm_model():
    """Mock ChatOpenAI for employee query responses"""
    mock = Mock()
    mock.invoke = Mock(return_value=Mock(content="Based on company policy, employees should follow procedures."))
    return mock

@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection for company knowledge base"""
    mock = Mock()
    mock.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    mock.query.return_value = {
        "ids": [["doc1", "doc2"]],
        "documents": [["Company policy document content", "HR procedure content"]],
        "metadatas": [[{"source": "policy.pdf", "type": "pdf"}, {"source": "hr.txt", "type": "txt"}]]
    }
    mock.add = Mock()
    mock.delete = Mock()
    return mock

@pytest.fixture
def sample_file_info():
    """Sample file info for testing company document processing"""
    return {
        'path': '/test/path/file.txt',
        'filename': 'file.txt',
        'modification_time': 1692000000,
        'size': 1024,
        'hash': 'test_hash_123'
    }

@pytest.fixture
def sample_company_documents():
    """Sample company documents for RAG ingestion testing"""
    return [
        {
            "content": "Company vacation policy: Employees receive 20 days annual leave and must submit requests via HR portal",
            "metadata": {"source_file": "hr_policy.pdf", "document_type": "pdf", "department": "HR"}
        },
        {
            "content": "IT Security Policy: All employees must use strong passwords, 2FA, and VPN for remote access",
            "metadata": {"source_file": "it_security.txt", "document_type": "txt", "department": "IT"}
        },
        {
            "content": "Employee Directory: John Doe, Engineering Department, Senior Developer, Reports to Jane Smith",
            "metadata": {"source_file": "employees.csv", "document_type": "csv", "department": "HR"}
        }
    ]

@pytest.fixture
def sample_employee_questions():
    """Common employee questions about company policies and procedures"""
    return [
        "What is the company vacation policy and how do I request time off?",
        "How do I report a security incident or suspicious email?",
        "What IT equipment can I request for my home office setup?",
        "Who is my manager and what's the reporting structure?",
        "What are the password requirements for company systems?",
        "How do I access company VPN for remote work?",
        "What training is required for new employees?"
    ]