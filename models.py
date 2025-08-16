"""LLM and Embedding models for company RAG system"""
import os
from openai import OpenAI
from chromadb.utils import embedding_functions
from config import (
    LLMType, EmbeddingsType, OPENAI_API_KEY_ENV_VAR,
    OPENAI_EMBEDDING_MODEL, OPENAI_LLM_MODEL,
    MAX_TOKENS, TEMPERATURE
)

class EmbeddingModel:
    def __init__(self, model_type):
        self.model_type = model_type
        if model_type == EmbeddingsType.OPENAI.value:
            self.client = OpenAI(api_key=os.getenv(OPENAI_API_KEY_ENV_VAR))
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv(OPENAI_API_KEY_ENV_VAR), 
                model_name=OPENAI_EMBEDDING_MODEL
            )

class LLMModel:
    def __init__(self, model_type):
        self.model_type = model_type
        if model_type == LLMType.OPENAI.value:
            self.client = OpenAI(api_key=os.getenv(OPENAI_API_KEY_ENV_VAR))
            self.model_name = OPENAI_LLM_MODEL

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