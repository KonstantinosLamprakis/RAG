"""Document management for company RAG system"""
import os
import json
import hashlib
import pandas as pd
import PyPDF2
from pathlib import Path
from datetime import datetime
from config import ( FileExtensions,PDF_CHUNK_SIZE, PDF_OVERLAP)

class DocumentManager:
    """Manages document processing and metadata tracking for company RAG system"""
    
    def __init__(self, data_directory, metadata_path):
        self.data_directory = data_directory
        self.metadata_path = metadata_path
        self.file_metadata = self.load_file_metadata()
        
    def load_file_metadata(self):
        """Load existing file metadata from JSON file"""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading file metadata: {e}")
            return {}
    
    def save_file_metadata(self):
        """Save file metadata to JSON file"""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.file_metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving file metadata: {e}")
    
    def calculate_file_hash(self, file_path):
        """Calculate SHA-256 hash of file content for change detection"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def get_file_info(self, file_path):
        """Get file information including modification time and hash"""
        try:
            stat = os.stat(file_path)
            return {
                'path': file_path,
                'filename': os.path.basename(file_path),
                'modification_time': stat.st_mtime,
                'size': stat.st_size,
                'hash': self.calculate_file_hash(file_path)
            }
        except Exception as e:
            print(f"Error getting file info for {file_path}: {e}")
            return None
    
    def has_file_changed(self, file_path):
        """Check if file has changed since last processing"""
        current_info = self.get_file_info(file_path)
        if not current_info:
            return False, None
            
        stored_info = self.file_metadata.get(file_path)
        if not stored_info:
            return True, current_info  # New file
            
        if (current_info['modification_time'] != stored_info['modification_time'] or
            current_info['hash'] != stored_info['hash']):
            return True, current_info
            
        return False, current_info
    
    def get_changed_files(self):
        """Get list of files that need to be processed (new/modified)"""
        changed_files = []
        
        # Check existing files in data directory
        for ext in FileExtensions:
            pattern = f"*{ext.value}"
            for file_path in Path(self.data_directory).glob(pattern):
                file_path_str = str(file_path)
                has_changed, file_info = self.has_file_changed(file_path_str)
                if has_changed:
                    changed_files.append({
                        'action': 'update',
                        'file_path': file_path_str,
                        'file_info': file_info
                    })
        
        # Check for deleted files
        for stored_path in list(self.file_metadata.keys()):
            if not os.path.exists(stored_path):
                changed_files.append({
                    'action': 'delete',
                    'file_path': stored_path,
                    'file_info': None
                })
        
        return changed_files
    
    def update_file_metadata(self, file_path, file_info):
        """Update metadata for a processed file"""
        self.file_metadata[file_path] = file_info
        self.save_file_metadata()
    
    def remove_file_metadata(self, file_path):
        """Remove metadata for a deleted file"""
        if file_path in self.file_metadata:
            del self.file_metadata[file_path]
            self.save_file_metadata()


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        text_content = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content += f"\n--- Page {page_num} ---\n{page_text}\n"
                except Exception as e:
                    print(f"Error extracting text from page {page_num} in {pdf_path}: {str(e)}")
                    continue
        return text_content.strip()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {str(e)}")
        return None

def chunk_pdf_text(text, chunk_size, overlap):
    """Split large PDF text into manageable chunks for RAG processing"""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            for i in range(min(100, chunk_size // 4)):
                if end - i > start and text[end - i] in ['.', '!', '?', '\n\n']:
                    end = end - i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else len(text)
    
    return chunks

def process_file_for_rag(file_path, file_info):
    """Process a single file and return documents with metadata"""
    documents = []
    file_ext = Path(file_path).suffix.lower()
    
    try:
        if file_ext == FileExtensions.CSV.value:
            df = pd.read_csv(file_path)
            for index, row in df.iterrows():
                doc_parts = []
                for column, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        doc_parts.append(f"{column}: {value}")
                
                if doc_parts:
                    doc_content = "\n".join(doc_parts)
                    documents.append({
                        'content': doc_content,
                        'metadata': {
                            'source_file': file_info['filename'],
                            'file_path': file_path,
                            'file_hash': file_info['hash'],
                            'modification_time': file_info['modification_time'],
                            'document_type': 'csv',
                            'row_index': index
                        }
                    })
        
        elif file_ext == FileExtensions.TXT.value:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    documents.append({
                        'content': content,
                        'metadata': {
                            'source_file': file_info['filename'],
                            'file_path': file_path,
                            'file_hash': file_info['hash'],
                            'modification_time': file_info['modification_time'],
                            'document_type': 'txt'
                        }
                    })
        
        elif file_ext == FileExtensions.PDF.value:
            pdf_text = extract_text_from_pdf(file_path)
            if pdf_text:
                if len(pdf_text) > PDF_CHUNK_SIZE:
                    chunks = chunk_pdf_text(pdf_text, PDF_CHUNK_SIZE, PDF_OVERLAP)
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            'content': chunk,
                            'metadata': {
                                'source_file': file_info['filename'],
                                'file_path': file_path,
                                'file_hash': file_info['hash'],
                                'modification_time': file_info['modification_time'],
                                'document_type': 'pdf',
                                'chunk_index': i,
                                'total_chunks': len(chunks)
                            }
                        })
                else:
                    documents.append({
                        'content': pdf_text,
                        'metadata': {
                            'source_file': file_info['filename'],
                            'file_path': file_path,
                            'file_hash': file_info['hash'],
                            'modification_time': file_info['modification_time'],
                            'document_type': 'pdf'
                        }
                    })
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
    return documents

def generate_document_id(file_path, chunk_index):
    """Generate unique document ID for vector storage"""
    return f"{file_path}::chunk_{chunk_index}"