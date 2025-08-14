import streamlit as st
import csv
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os 
from dotenv import load_dotenv
from enum import Enum
import PyPDF2
from io import BytesIO
import hashlib
from datetime import datetime
from pathlib import Path
import json

class FileExtensions(Enum):
    CSV = ".csv"
    PDF = ".pdf"
    TXT = ".txt"
    
# Constants for RAG system configuration
DATA_DIRECTORY = "./data"
COMPANY_KNOWLEDGE_COLLECTION = "company_knowledge"
PERSISTENT_DB_PATH = "./chroma_db"  # Persistent database path
FILE_METADATA_PATH = "./file_metadata.json"  # File tracking metadata
MAX_TOKENS = 500
TEMPERATURE = 0.1
TOP_K_RESULTS = 2

# PDF processing
PDF_CHUNK_SIZE = 1000  
PDF_OVERLAP = 200      

# Model configuration constants
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OPENAI_LLM_MODEL = "gpt-4o-mini"
OLLAMA_LLM_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

class DocumentManager:
    """Manages document processing and metadata tracking for company RAG system"""
    
    def __init__(self, data_directory=DATA_DIRECTORY, metadata_path=FILE_METADATA_PATH):
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
            
        # Check if modification time or hash changed
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

class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"), 
                model_name=OPENAI_EMBEDDING_MODEL
            )
        elif model_type == "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "nomic":
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OLLAMA_API_KEY,
                api_base=OLLAMA_BASE_URL,
                model_name=OLLAMA_EMBEDDING_MODEL, 
            )

class LLMModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = OPENAI_LLM_MODEL
        else:
            self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
            self.model_name = OLLAMA_LLM_MODEL

    def generate_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating completion: {str(e)}"

def setup_persistent_chroma(embedding_model):
    """Setup persistent ChromaDB with embeddings"""
    client = chromadb.PersistentClient(path=PERSISTENT_DB_PATH)
    
    try:
        collection = client.get_collection(
            name=COMPANY_KNOWLEDGE_COLLECTION,
            embedding_function=embedding_model.embedding_fn
        )
        print(f"Using existing collection: {COMPANY_KNOWLEDGE_COLLECTION}")
    except:
        collection = client.create_collection(
            name=COMPANY_KNOWLEDGE_COLLECTION,
            embedding_function=embedding_model.embedding_fn
        )
        print(f"Created new collection: {COMPANY_KNOWLEDGE_COLLECTION}")
    
    return collection

def generate_document_id(file_path, chunk_index=0):
    """Generate unique document ID for vector storage"""
    return f"{file_path}::chunk_{chunk_index}"

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

def chunk_pdf_text(text, chunk_size=PDF_CHUNK_SIZE, overlap=PDF_OVERLAP):
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
                    chunks = chunk_pdf_text(pdf_text)
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

def update_vector_database(collection, doc_manager):
    """Update vector database with changed files"""
    changed_files = doc_manager.get_changed_files()
    
    if not changed_files:
        print("No file changes detected. Vector database is up to date.")
        return 0
    
    total_processed = 0
    
    for change in changed_files:
        file_path = change['file_path']
        action = change['action']
        
        if action == 'delete':
            try:
                existing_docs = collection.get(where={"file_path": file_path})
                if existing_docs['ids']:
                    collection.delete(ids=existing_docs['ids'])
                    print(f"Deleted {len(existing_docs['ids'])} vectors for deleted file: {file_path}")
                
                doc_manager.remove_file_metadata(file_path)
                total_processed += 1
            except Exception as e:
                print(f"Error deleting vectors for {file_path}: {e}")
        
        elif action == 'update':
            file_info = change['file_info']
            try:
                existing_docs = collection.get(where={"file_path": file_path})
                if existing_docs['ids']:
                    collection.delete(ids=existing_docs['ids'])
                    print(f"Removed {len(existing_docs['ids'])} existing vectors for: {file_path}")
            except Exception as e:
                print(f"Warning: Could not remove existing vectors for {file_path}: {e}")
            
            documents = process_file_for_rag(file_path, file_info)
            
            if documents:
                doc_contents = []
                doc_ids = []
                doc_metadatas = []
                
                for i, doc in enumerate(documents):
                    doc_id = generate_document_id(file_path, i)
                    doc_contents.append(doc['content'])
                    doc_ids.append(doc_id)
                    doc_metadatas.append(doc['metadata'])
                
                collection.add(
                    documents=doc_contents,
                    ids=doc_ids,
                    metadatas=doc_metadatas
                )
                
                print(f"Added {len(documents)} vectors for: {file_path}")
                
                doc_manager.update_file_metadata(file_path, file_info)
                total_processed += 1
    
    print(f"Vector database update complete. Processed {total_processed} files.")
    return total_processed

def find_related_chunks(query, collection, top_k=TOP_K_RESULTS):
    """Find related document chunks using vector similarity search"""
    results = collection.query(query_texts=[query], n_results=top_k)
    
    for doc in results['documents'][0]:
        print(f"Found related document: {doc[:100]}...")

    return list(zip(
        results['documents'][0], 
        results['metadatas'][0] if results['metadatas'][0] else [{}] * len(results['documents'][0])
    ))

def augment_prompt(query, related_chunks):
    """Create augmented prompt with context from documents"""
    context = "\n".join(chunk[0] for chunk in related_chunks)
    augmented_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return augmented_prompt

def rag_pipeline(query, collection, llm_model, top_k=TOP_K_RESULTS):
    """Execute RAG pipeline for company knowledge queries"""
    print(f"Running RAG pipeline for query: {query}")
    related_chunks = find_related_chunks(query, collection, top_k)
    augmented_prompt = augment_prompt(query, related_chunks)

    response = llm_model.generate_completion([
        {"role": "system", "content": "You are a helpful assistant for company knowledge and procedures."},
        {"role": "user", "content": augmented_prompt},
    ])

    references = [chunk[0] for chunk in related_chunks]
    return response, references, augmented_prompt

def streamlit_app():
    st.set_page_config(page_title="Company Knowledge RAG", layout="wide")
    st.title("🏢 Company Knowledge RAG System with Persistent Storage")
    
    if 'doc_manager' not in st.session_state:
        st.session_state.doc_manager = DocumentManager()
    
    st.sidebar.title("Model Configuration")
    llm_type = st.sidebar.radio(
        "Select LLM Model", 
        ["openai", "ollama"],
        format_func=lambda x: "OpenAI GPT-4" if x == "openai" else "Ollama"
    )
    embedding_type = st.sidebar.radio(
        "Select Embedding Model",
        ["openai", "chroma", "nomic"],
        format_func=lambda x: {
            "openai": "OpenAI Embeddings",
            "chroma": "Chroma Default",
            "nomic": "Nomic Embed Text (Ollama)", 
        }[x],
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
                st.sidebar.success(f"Updated {processed_count} files in vector database!")
                # Force reinitialization
                if 'initialized' in st.session_state:
                    del st.session_state['initialized']
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

    if "initialized" not in st.session_state or st.sidebar.button("🚀 Initialize RAG System"):
        st.session_state.initialized = False
        
        with st.spinner("Initializing RAG system with persistent storage..."):
            try:
                st.session_state.llm_model = LLMModel(llm_type)
                st.session_state.embedding_model = EmbeddingModel(embedding_type)
                st.session_state.collection = setup_persistent_chroma(st.session_state.embedding_model)
                
                # Auto-update vector database on initialization
                processed_count = update_vector_database(
                    st.session_state.collection,
                    st.session_state.doc_manager,
                )
                
                vector_count = st.session_state.collection.count()
                st.session_state.initialized = True
                
                st.success(f"RAG system initialized! Database contains {vector_count} vectors.")
                if processed_count > 0:
                    st.info(f"Processed {processed_count} changed files during initialization.")
                    
            except Exception as e:
                st.error(f"Error initializing RAG system: {e}")
                return

    if not st.session_state.get('initialized', False):
        st.info("Please initialize the RAG system to start querying company knowledge.")
        return

    # Update models if selection changed
    # TODO(KL) Shouldn't we delete all embeddings and reprocess all files?
    if (hasattr(st.session_state, 'llm_model') and 
        st.session_state.llm_model.model_type != llm_type):
        st.session_state.llm_model = LLMModel(llm_type)
    
    if (hasattr(st.session_state, 'embedding_model') and 
        st.session_state.embedding_model.model_type != embedding_type):
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
        st.session_state.collection = setup_persistent_chroma(st.session_state.embedding_model)

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
                    mod_time = datetime.fromtimestamp(metadata['modification_time']).strftime('%Y-%m-%d %H:%M:%S')
                    st.write(f"📄 {metadata['filename']} (Modified: {mod_time})")
        except Exception as e:
            st.error(f"Error displaying database info: {e}")

    st.markdown("### 💬 Ask Questions About Company Knowledge")
    query = st.text_input(
        "Enter your question:", 
        placeholder="e.g., What are the company policies? How does the technical procedure work?"
    )

    if query:
        with st.spinner("Searching company knowledge base..."):
            try:
                response, references, augmented_prompt = rag_pipeline(
                    query, 
                    st.session_state.collection, 
                    st.session_state.llm_model
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
                    
            except Exception as e:
                st.error(f"Error processing query: {e}")

if __name__ == "__main__":
    streamlit_app()