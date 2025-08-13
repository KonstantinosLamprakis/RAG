import streamlit as st
import csv
import pandas as pd
import chromadb
from chromadb.utils import  embedding_functions
from openai import OpenAI
import os 
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
directory_path="./data"

class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-ada-002")
        elif model_type == "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "nomic":
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key="ollama",
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text", 
            )

class LLMModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = "gpt-4o-mini"
        else:
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model_name = "llama3.2"

    def generate_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=500,
                temperature=0.7
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

def load_csv():
    """Load all .csv files from directory for company documents - handles any CSV structure"""
    all_csvs = []
    
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    for csv_file in csv_files:     
        file_path = os.path.join(directory_path, csv_file)   
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

def load_pdf():
    """Load all .pdf files from directory for company documents"""
    # TODO: Implement PDF parsing for company procedures and policies
    # This will be needed for technical documentation, employee handbooks, etc.
    documents = []
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        # Future implementation with PyPDF2 or similar
        print(f"PDF support coming soon for: {pdf_file}")
    
    return documents

def load_all_company_documents():
    """Load all supported document types for the RAG system"""
    all_documents = []
    
    csv_docs = load_csv()
    all_documents.extend(csv_docs)
    
    txt_docs = load_txt_files()
    all_documents.extend(txt_docs)
    
    pdf_docs = load_pdf()
    all_documents.extend(pdf_docs)
    
    print(f"Total company documents loaded: {len(all_documents)}")
    return all_documents

def load_txt_files():
    """Load all .txt files from directory for company documents"""
    documents = []
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    
    for file_name in txt_files:
        file_path = os.path.join(directory_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    documents.append(content)
                    print(f"Loaded document from {file_name}: {content[:100]}...")
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")
    
    return documents

def setup_chroma(documents, embedding_model):
    client = chromadb.Client()
    try:
        client.delete_collection("company_knowledge")
    except:
        pass

    collection = client.create_collection(
        name="company_knowledge", embedding_function=embedding_model.embedding_fn
    )

    collection.add(documents=documents, ids=[str(i) for i in range(len(documents))])
    print(f"Collection 'company_knowledge' created with {len(documents)} documents.")
    
    return collection

def find_related_chunks(query, collection, top_k=2):
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

    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        
        all_documents = load_all_company_documents()
        
        if not all_documents:
            st.warning("No documents found. Please generate sample data or add CSV/TXT files.")
            return
            
        st.session_state.documents = all_documents
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)

        st.session_state.collection = setup_chroma(all_documents, st.session_state.embedding_model)
        st.session_state.initialized = True

    if (st.session_state.llm_model.model_type == llm_type
        or st.session_state.embedding_model.model_type == embedding_type):
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
    

   

    with st.expander("📁 Available Company Knowledge", expanded=False):
        for i, doc in enumerate(st.session_state.documents):
            st.write(f"**Document {i+1}:** {doc[:150]}...")

        st.markdown("### 💬 Ask Questions About Company Procedures & Policies")
        query = st.text_input(
            "Enter your question:", 
            placeholder="e.g., What is the remote work policy? How does code review work?"
        )

        if query:
            with st.spinner("Processing your query..."):
                response, reference, augmented_prompt = rag_pipeline( 
                    query, 
                    st.session_state.collection, 
                    st.session_state.llm_model
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Response")
                    st.write(response)
                with col2:
                    st.markdown("### References Used")
                    for ref in reference:
                        st.write(f"- {ref}")
                
                with st.expander("Technical Details", expanded=False):
                    st.markdown("### Augmented Prompt")
                    st.code(augmented_prompt)

                    st.markdown("### Model Configuration")
                    st.write(f"LLM Model: {llm_type.upper()}")
                    st.write(f"Embedding Model: {embedding_type.upper()}")

if __name__ == "__main__":
    streamlit_app()