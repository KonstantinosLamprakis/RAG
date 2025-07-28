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
                model_name="nomic-embed-text", # you can find this in ollama websites searching for models for embeddings
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
    
def generate_csv(): 
    facts = [
        {"id": 1, "fact": "The Sun accounts for about 99.86% of the total mass of the Solar System."},
        {"id": 2, "fact": "A day on Venus is longer than a year on Venus. It takes about 243 Earth days to rotate once on its axis, but only about 225 Earth days to orbit the Sun."},
        {"id": 3, "fact": "Jupiter is so large that it could fit all the other planets inside it. It is more than 11 times the diameter of Earth."},
        {"id": 4, "fact": "Saturn's rings are made mostly of ice particles, with a smaller amount of rocky debris and dust."},
        {"id": 5, "fact": "Mars has the largest volcano in the Solar System, Olympus Mons, which is about 13.6 miles (22 kilometers) high."},
        {"id": 6, "fact": "Neptune has the strongest winds in the Solar System, with speeds reaching up to 1,200 miles per hour (2,000 kilometers per hour)."},
        {"id": 7, "fact": "Pluto was reclassified as a dwarf planet in 2006 by the International Astronomical Union."},
        {"id": 8, "fact": "The Voyager spacecraft have traveled farther than any other human-made objects in space."},
        {"id": 9, "fact": "A year on Mercury is only about 88 Earth days long."},
        {"id": 10, "fact": "The Great Red Spot on Jupiter is a giant storm that has been raging for at least 350 years."}
    ]

    with open("space_facts.csv", mode="w", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["id", "fact"])
        writer.writeheader()
        writer.writerows(facts)

    return facts


# def load_csv(file_path):
#     df = pd.read_csv(file_path)
#     documents = df['fact'].tolist()
#     for doc in documents:
#         print (f"Loaded document: {doc}")
#     return documents

def setup_chroma(documents, embedding_model):
    client = chromadb.Client()
    try:
        client.delete_collection("space_facts")
    except:
        pass

    collection = client.create_collection(
        name="space_facts", embedding_function=embedding_model.embedding_fn
    )

    collection.add(documents=documents, ids=[str(i) for i in range(len(documents))])
    print(f"Collection 'space_facts' created with {len(documents)} documents.")
    
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
    st.set_page_config(page_title="Space Facts RAG", layout="wide")
    st.title("Space Facts RAG System")
    
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
        st.session_state.facts = generate_csv()
        
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)

        documents = [fact["fact"] for fact in st.session_state.facts]
        st.session_state.collection = setup_chroma(documents, st.session_state.embedding_model)
        st.session_state.initialized = True

    if (st.session_state.llm_model.model_type == llm_type
        or st.session_state.embedding_model.model_type == embedding_type):
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
    
    with st.expander("Available Space Facts", expanded=False):
        for fact in st.session_state.facts:
            st.write(f"- {fact['fact']}")

        query = st.text_input("Enter your question about space:", placeholder="e.g., What is the largest planet in our solar system?")
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