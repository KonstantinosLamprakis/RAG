import pytest
import os
import tempfile
from unittest.mock import Mock, patch
from src.document_manager import DocumentManager
from src.vector_store import setup_persistent_chroma, update_vector_database
from src.models import EmbeddingModel, LLMModel
from src.rag_pipeline import rag_pipeline
from src.config import LLMType, TOP_K_RESULTS, MAX_RELEVANCE_DISTANCE

@pytest.mark.integration
class TestCompanyRAGPipeline:
    """Integration tests for company RAG system - employees querying company documents"""
    
    def test_company_document_ingestion_pipeline(self, temp_data_dir, temp_dir):
        """Test complete company document ingestion: CSV/PDF/TXT -> ChromaDB vector storage"""
        # Setup company document management for employee access
        metadata_path = os.path.join(temp_dir, "company_metadata.json")
        doc_manager = DocumentManager(temp_data_dir, metadata_path)
        
        # Mock ChromaDB collection for company knowledge base
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        
        # Ingest company documents (policies, procedures, technical docs) for employee queries
        result = update_vector_database(mock_collection, doc_manager)
        
        # Verify company documents are processed and available to employees via RAG
        assert result > 0  # Should process company files (CSV, PDF, TXT)
        assert mock_collection.add.called
        
        # Verify proper metadata for employee document search
        call_args = mock_collection.add.call_args
        assert 'documents' in call_args[1]  
        assert 'metadatas' in call_args[1]  

    @pytest.mark.document_processing
    @pytest.mark.company_knowledge
    def test_multi_format_company_documents(self, temp_dir):
        """Test processing company documents in various formats for employee RAG queries"""
        # Create realistic company document repository for RAG ingestion
        data_dir = os.path.join(temp_dir, "company_documents")
        os.makedirs(data_dir)
        
        employee_csv = (
            "employee_id,name,department,role,manager,email,location\n"
            "1,John Smith,Engineering,Senior Developer,Jane Doe,john.smith@company.com,Remote\n"
            "2,Jane Doe,Engineering,Team Lead,Bob Wilson,jane.doe@company.com,NYC Office\n"
            "3,Alice Johnson,HR,HR Specialist,Carol Brown,alice.johnson@company.com,LA Office"
        )
        with open(os.path.join(data_dir, "employee_directory.csv"), "w") as f:
            f.write(employee_csv)
        
        security_policy = (
            "Company IT Security Policy:\n"
            "1. Password Requirements: Minimum 12 characters, updated every 90 days\n"
            "2. Two-Factor Authentication: Required for all company systems\n"
            "3. VPN Access: Mandatory for remote work and external access\n"
            "4. Incident Reporting: Contact security@company.com for breaches"
        )
        with open(os.path.join(data_dir, "it_security_policy.txt"), "w") as f:
            f.write(security_policy)
        
        hr_procedures = (
            "HR Employee Procedures:\n"
            "Vacation Policy: 20 days PTO annually, submit via HR portal\n"
            "Onboarding: Complete I-9, orientation, and training within 30 days\n"
            "Equipment Requests: Submit via IT portal with manager approval"
        )
        with open(os.path.join(data_dir, "hr_procedures.txt"), "w") as f:
            f.write(hr_procedures)
        
        metadata_path = os.path.join(temp_dir, "company_metadata.json")
        doc_manager = DocumentManager(data_dir, metadata_path)
        
        # Process all company documents for employee RAG access
        changed_files = doc_manager.get_changed_files()
        
        # Verify all company document types are ready for employee queries
        assert len(changed_files) == 3
        file_names = [os.path.basename(cf['file_path']) for cf in changed_files]
        assert "employee_directory.csv" in file_names
        assert "it_security_policy.txt" in file_names
        assert "hr_procedures.txt" in file_names

    @pytest.mark.vector_store
    @pytest.mark.company_knowledge
    def test_company_knowledge_vector_operations(self, temp_data_dir, temp_dir):
        """Test ChromaDB vector operations for company knowledge base used by employees"""
        metadata_path = os.path.join(temp_dir, "company_metadata.json")
        doc_manager = DocumentManager(temp_data_dir, metadata_path)
        
        # Mock ChromaDB for company knowledge collection
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        
        # Initial processing of company documents for RAG
        result = update_vector_database(mock_collection, doc_manager)
        
        # Verify company documents are vectorized for employee search
        assert result > 0
        assert mock_collection.add.called
        
        # Test company policy update scenario (common in real companies)
        mock_collection.reset_mock()
        mock_collection.get.return_value = {"ids": ["policy_1", "policy_2"], "documents": [], "metadatas": []}
        
        # Simulate company document update (new employee hired)
        csv_file = os.path.join(temp_data_dir, "employees.csv")
        with open(csv_file, "a") as f:
            f.write("\n3,Sarah Davis,Marketing,Marketing Manager,CEO,sarah.davis@company.com")
        
        # Re-process updated company documents for RAG
        result = update_vector_database(mock_collection, doc_manager)
        
        # Verify company knowledge base is updated for employee queries
        assert mock_collection.delete.called  # Remove outdated company data
        assert mock_collection.add.called     # Add updated company information