"""Vector store management for company knowledge base"""
import chromadb
from config import (
    PERSISTENT_DB_PATH, COMPANY_KNOWLEDGE_COLLECTION
)
from document_manager import process_file_for_rag, generate_document_id, DocumentManager
from models import EmbeddingModel

def setup_persistent_chroma(embedding_model: EmbeddingModel):
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

def update_vector_database(collection, doc_manager: DocumentManager):
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
