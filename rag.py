import streamlit as st
import csv
import pandas as pd
import chromadb
from chromadb.utils import  embedding_functions
from openai import OpenAI
import os 
from dotenv import load_dotenv
from enum import Enum
import PyPDF2
from io import BytesIO

class FileExtensions(Enum):
    CSV = ".csv"
    PDF = ".pdf"
    TXT = ".txt"
    
# Constants for RAG system configuration
DATA_DIRECTORY = "./data"
COMPANY_KNOWLEDGE_COLLECTION = "company_knowledge"
MAX_TOKENS = 500
TEMPERATURE = 0.3
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

class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name=OPENAI_EMBEDDING_MODEL)
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

    def embed(self, text):
        response = self.client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return response['data'][0]['embedding']

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

def load_pdf():
    """Load all .pdf files from directory"""
    documents = []
    pdf_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith(FileExtensions.PDF.value)]
    
    if not pdf_files:
        print("No PDF files found in company data directory")
        return documents
    
    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_DIRECTORY, pdf_file)
        print(f"Processing company PDF: {pdf_file}")
        
        pdf_text = extract_text_from_pdf(file_path)
        
        if pdf_text:
            if len(pdf_text) > PDF_CHUNK_SIZE:
                chunks = chunk_pdf_text(pdf_text)
                
                for i, chunk in enumerate(chunks, 1):
                    chunk_with_metadata = f"Source: {pdf_file} (Part {i}/{len(chunks)})\n\n{chunk}"
                    documents.append(chunk_with_metadata)
                    print(f"Loaded PDF chunk {i}/{len(chunks)} from {pdf_file}: {chunk[:100]}...")
            else:
                doc_with_metadata = f"Source: {pdf_file}\n\n{pdf_text}"
                documents.append(doc_with_metadata)
                print(f"Loaded complete PDF from {pdf_file}: {pdf_text[:100]}...")
        else:
            print(f"Failed to extract text from {pdf_file}")
    
    print(f"Successfully loaded {len(documents)} PDF document chunks for company RAG")
    return documents

def validate_pdf_processing():
    """Validate PDF processing functionality for company RAG system"""
    pdf_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith(FileExtensions.PDF.value)]
    
    if not pdf_files:
        return False, "No PDF files found in company data directory"
    
    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_DIRECTORY, pdf_file)
        
        # Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                if len(pdf_reader.pages) == 0:
                    return False, f"PDF {pdf_file} appears to be empty"
        except Exception as e:
            return False, f"Cannot read PDF {pdf_file}: {str(e)}"
    
    return True, f"All {len(pdf_files)} PDF files are valid for processing"

def load_csv():
    """Load all .csv files from directory for company documents - handles any CSV structure"""
    all_csvs = []
    
    csv_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith(FileExtensions.CSV.value)]
    for csv_file in csv_files:     
        file_path = os.path.join(DATA_DIRECTORY, csv_file)   
        try:
            df = pd.read_csv(file_path)
            
            for index, row in df.iterrows():
                doc_parts = []
                
                for column, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        doc_parts.append(f"{column}: {value}")
                
                if doc_parts:
                    doc_content = "\n".join(doc_parts)
                    all_csvs.append(doc_content)
                    print(f"Loaded CSV document from {csv_file}: {doc_content[:100]}...")
                    
        except Exception as e:
            print(f"Error loading CSV {csv_file}: {str(e)}")
            continue
    
    return all_csvs

def load_txt_files():
    """Load all .txt files from directory for company documents"""
    documents = []
    txt_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith(FileExtensions.TXT.value)]
    
    for file_name in txt_files:
        file_path = os.path.join(DATA_DIRECTORY, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    documents.append(content)
                    print(f"Loaded document from {file_name}: {content[:100]}...")
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")
    
    return documents

def load_all_company_documents():
    """Load all supported document types for the RAG system"""
    all_documents = []
    
    pdf_valid, pdf_message = validate_pdf_processing()
    print(f"PDF Validation: {pdf_message}")
    
    csv_docs = load_csv()
    all_documents.extend(csv_docs)
    print(f"Loaded {len(csv_docs)} CSV documents")
    
    txt_docs = load_txt_files()
    all_documents.extend(txt_docs)
    print(f"Loaded {len(txt_docs)} TXT documents")
    
    if pdf_valid:
        pdf_docs = load_pdf()
        all_documents.extend(pdf_docs)
        print(f"Loaded {len(pdf_docs)} PDF document chunks")
    else:
        print("Skipping PDF processing due to validation issues")
    
    print(f"Total company documents loaded for RAG: {len(all_documents)}")
    return all_documents

def setup_chroma(documents, embedding_model):
    client = chromadb.Client()
    try:
        client.delete_collection(COMPANY_KNOWLEDGE_COLLECTION)
    except:
        pass

    collection = client.create_collection(
        name=COMPANY_KNOWLEDGE_COLLECTION, embedding_function=embedding_model.embedding_fn
    )

    collection.add(documents=documents, ids=[str(i) for i in range(len(documents))])
    print(f"Collection {COMPANY_KNOWLEDGE_COLLECTION} created with {len(documents)} documents.")
    
    return collection

def find_related_chunks(query, collection, top_k=TOP_K_RESULTS):
    results = collection.query(query_texts=[query], n_results=top_k) 

    for doc in results['documents'][0]:
        print(f"Found related document: {doc}")

    return list(
        zip(results['documents'][0], (
            results['metadatas'][0]
            if results['metadatas'][0] 
            else [{}] * len(results['documents'][0])
            )
        ),
    )

def augment_prompt(query, related_chunks):
    context = "\n".join(chunk[0] for chunk in related_chunks)
    augmented_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    print(f"Augmented prompt: {augmented_prompt}")
    return augmented_prompt

def rag_pipeline(query, collection, llm_model, top_k=2):
    print(f"Running RAG pipeline for query: {query}")
    related_chunks = find_related_chunks(query, collection, top_k)
    augmented_prompt = augment_prompt(query, related_chunks)

    response = llm_model.generate_completion(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": augmented_prompt},
        ]
    )

    print(f"Generated response: {response}")
    references = [chunk[0] for chunk in related_chunks]
    return response, references, augmented_prompt

def streamlit_app():
    st.set_page_config(page_title="Company Knowledge RAG", layout="wide")
    st.title("🏢 Company Knowledge RAG System")
    
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

    if st.sidebar.button("Refresh Document Stats"):
        csv_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith(FileExtensions.CSV.value)]
        txt_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith(FileExtensions.TXT.value)]
        pdf_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith(FileExtensions.PDF.value)]
        
        st.sidebar.write(f"📊 **Document Statistics:**")
        st.sidebar.write(f"CSV files: {len(csv_files)}")
        st.sidebar.write(f"TXT files: {len(txt_files)}")
        st.sidebar.write(f"PDF files: {len(pdf_files)}")

    if "initialized" not in st.session_state or st.sidebar.button("Reload Documents"):
        st.session_state.initialized = False
        
        with st.spinner("Loading company documents..."):
            all_documents = load_all_company_documents()
            
            if not all_documents:
                st.warning("No documents found. Please add csv, txt or pdf files to the data directory.")
                return
                
            st.session_state.documents = all_documents
            st.session_state.llm_model = LLMModel(llm_type)
            st.session_state.embedding_model = EmbeddingModel(embedding_type)

            st.session_state.collection = setup_chroma(all_documents, st.session_state.embedding_model)
            st.session_state.initialized = True
            
            st.success(f"Loaded {len(all_documents)} company documents.")

    if not st.session_state.get('initialized', False):
        st.info("Please load company documents to start using the RAG system.")
        return

    # Update models if selection changed
    if (st.session_state.llm_model.model_type != llm_type or 
        st.session_state.embedding_model.model_type != embedding_type):
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)

    with st.expander("📁 Available Company Knowledge", expanded=False):
        pdf_count = sum(1 for doc in st.session_state.documents if "Source:" in doc and ".pdf" in doc)
        csv_count = sum(1 for doc in st.session_state.documents if not ("Source:" in doc and ".pdf" in doc))
        
        st.write(f"**Document Summary:** {len(st.session_state.documents)} total documents")
        st.write(f"📄 PDF documents: {pdf_count} chunks")
        st.write(f"📝 CSV/TXT documents: {csv_count}")
        
        for i, doc in enumerate(st.session_state.documents):
            if "Source:" in doc and ".pdf" in doc:
                st.write(f"📄 **PDF Document {i+1}:** {doc[:200]}...")
            else:
                st.write(f"📝 **Document {i+1}:** {doc[:150]}...")

    st.markdown("### 💬 Ask Questions About Company Knowledge")
    query = st.text_input(
        "Enter your question:", 
        placeholder="e.g., What does the employee handbook say about vacation?"
    )

    if query:
        with st.spinner("Searching company documents..."):
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
                    if "Source:" in ref and ".pdf" in ref:
                        st.write(f"📄 PDF: {ref[:200]}...")
                    else:
                        st.write(f"📝 Document: {ref[:150]}...")
            
            with st.expander("🔧 Technical Details", expanded=False):
                st.markdown("### Augmented Prompt")
                st.code(augmented_prompt)

                st.markdown("### Model Configuration")
                st.write(f"LLM Model: {llm_type.upper()}")
                st.write(f"Embedding Model: {embedding_type.upper()}")
                st.write(f"Total Documents: {len(st.session_state.documents)}")
                st.write(f"PDF Chunks: {sum(1 for doc in st.session_state.documents if 'Source:' in doc and '.pdf' in doc)}")

if __name__ == "__main__":
    streamlit_app()