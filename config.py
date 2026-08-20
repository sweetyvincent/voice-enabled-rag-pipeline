"""Configuration module for the application."""
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    SARVAM_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    SARVAM_STT_URL: str = "https://api.sarvam.ai/speech-to-text"
    SARVAM_MODEL: str = "saaras:v3"
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    FAISS_INDEX_PATH: str = "data/faiss.index"
    CHUNKS_METADATA_PATH: str = "data/chunks_metadata.jsonl"
    BENCHMARK_RESULTS_PATH: str = "data/benchmark_results.json"
    FIXED_CHUNK_SIZE: int = 256
    FIXED_CHUNK_OVERLAP: int = 64
    SEMANTIC_CHUNK_SOFT_LIMIT: int = 200
    PASSAGE_MAX_TOKENS: int = 300
    RECURSIVE_MAX_TOKENS: int = 400
    RETRIEVAL_TOP_K: int = 20
    RETRIEVAL_FINAL_K: int = 5
    RELEVANCE_THRESHOLD: float = 0.35
    OFF_TOPIC_THRESHOLD: float = 0.10
    GROUNDING_MIN_OVERLAP: float = 0.30
    QUERY_MIN_LENGTH: int = 3
    QUERY_MAX_LENGTH: int = 500
    LLM_MAX_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT: float = 5.0
    LLM_MAX_RETRIES: int = 3
    DATASET_NAME: str = "ai4bharat/MSMARCO-XI"
    DATASET_SPLIT: str = "validation"
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000

    class Config:
        env_file = '.env'

@lru_cache()
def get_settings() -> Settings:
    """Returns cached settings instance."""
    return Settings()
