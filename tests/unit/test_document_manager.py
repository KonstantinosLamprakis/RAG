import pytest
import os
import json
from unittest.mock import patch, mock_open, Mock
from src.document_manager import (
    DocumentManager, 
    extract_text_from_pdf, 
    chunk_pdf_text,
    process_file_for_rag
)

class TestDocumentManager:
    
    def test_init(self, temp_data_dir, temp_metadata_file):
        """Test DocumentManager initialization"""
        dm = DocumentManager(temp_data_dir, temp_metadata_file)
        assert dm.data_directory == temp_data_dir
        assert dm.metadata_path == temp_metadata_file
        assert isinstance(dm.file_metadata, dict)
    
    def test_load_file_metadata_existing(self, temp_data_dir, temp_metadata_file):
        """Test loading existing metadata"""
        dm = DocumentManager(temp_data_dir, temp_metadata_file)
        assert len(dm.file_metadata) > 0
        assert "/path/to/old_file.txt" in dm.file_metadata
    
    def test_load_file_metadata_nonexistent(self, temp_data_dir, temp_dir):
        """Test loading metadata when file doesn't exist"""
        nonexistent_path = os.path.join(temp_dir, "nonexistent.json")
        dm = DocumentManager(temp_data_dir, nonexistent_path)
        assert dm.file_metadata == {}
    
    def test_calculate_file_hash(self, temp_data_dir):
        """Test file hash calculation"""
        dm = DocumentManager(temp_data_dir, "")
        csv_file = os.path.join(temp_data_dir, "employees.csv")
        hash1 = dm.calculate_file_hash(csv_file)
        hash2 = dm.calculate_file_hash(csv_file)
        
        assert hash1 is not None
        assert hash1 == hash2  # Same file should have same hash
        assert len(hash1) == 64  # SHA-256 hex string length
    
    def test_get_file_info(self, temp_data_dir):
        """Test getting file information"""
        dm = DocumentManager(temp_data_dir, "")
        csv_file = os.path.join(temp_data_dir, "employees.csv")
        file_info = dm.get_file_info(csv_file)
        
        assert file_info is not None
        assert file_info['filename'] == "employees.csv"
        assert file_info['path'] == csv_file
        assert 'modification_time' in file_info
        assert 'size' in file_info
        assert 'hash' in file_info
    
    def test_has_file_changed_new_file(self, temp_data_dir, temp_metadata_file):
        """Test detecting new file"""
        dm = DocumentManager(temp_data_dir, temp_metadata_file)
        csv_file = os.path.join(temp_data_dir, "employees.csv")
        
        has_changed, file_info = dm.has_file_changed(csv_file)
        assert has_changed is True
        assert file_info is not None
    
    def test_has_file_changed_unchanged_file(self, temp_data_dir, temp_metadata_file):
        """Test detecting unchanged file"""
        dm = DocumentManager(temp_data_dir, temp_metadata_file)
        csv_file = os.path.join(temp_data_dir, "employees.csv")
        
        # First check - file is new
        has_changed, file_info = dm.has_file_changed(csv_file)
        assert has_changed is True
        
        # Update metadata
        dm.update_file_metadata(csv_file, file_info)
        
        # Second check - file should be unchanged
        has_changed, file_info = dm.has_file_changed(csv_file)
        assert has_changed is False

class TestPDFProcessing:
    
    @patch('src.document_manager.PyPDF2.PdfReader')
    def test_extract_text_from_pdf_success(self, mock_pdf_reader):
        """Test successful PDF text extraction"""
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample PDF content"
        mock_pdf_reader.return_value.pages = [mock_page]
        
        with patch('builtins.open', mock_open()):
            result = extract_text_from_pdf("test.pdf")
            
        assert "Sample PDF content" in result
        assert "Page 1" in result
    
    def test_chunk_pdf_text_small_text(self):
        """Test chunking small text that doesn't need splitting"""
        text = "Short text content"
        chunks = chunk_pdf_text(text, 1000, 100)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_pdf_text_large_text(self):
        """Test chunking large text"""
        text = "This is a sentence. " * 100  # Create long text
        chunks = chunk_pdf_text(text, 200, 50)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 250 for chunk in chunks)  # Account for overlap
    
    def test_chunk_pdf_text_empty(self):
        """Test chunking empty text"""
        chunks = chunk_pdf_text("", 1000, 100)
        assert chunks == []

class TestFileProcessing:
    
    def test_process_csv_file(self, temp_data_dir, sample_file_info):
        """Test processing CSV file"""
        csv_file = os.path.join(temp_data_dir, "employees.csv")
        sample_file_info['path'] = csv_file
        sample_file_info['filename'] = "employees.csv"
        
        documents = process_file_for_rag(csv_file, sample_file_info)
        
        assert len(documents) == 2  # Two rows in CSV
        assert documents[0]['metadata']['document_type'] == 'csv'
        assert 'John Doe' in documents[0]['content']
        assert 'Jane Smith' in documents[1]['content']
    
    def test_process_txt_file(self, temp_data_dir, sample_file_info):
        """Test processing TXT file"""
        txt_file = os.path.join(temp_data_dir, "policy.txt")
        sample_file_info['path'] = txt_file
        sample_file_info['filename'] = "policy.txt"
        
        documents = process_file_for_rag(txt_file, sample_file_info)
        
        assert len(documents) == 1
        assert documents[0]['metadata']['document_type'] == 'txt'
        assert 'Company Policy' in documents[0]['content']