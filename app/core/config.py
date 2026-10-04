import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project metadata
    PROJECT_NAME: str = "IPO Prospectus Analyst"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    # LLM API Keys
    GEMINI_API_KEY: str = Field(default="", description="Google AI Studio API Key")
    GROQ_API_KEY: str = Field(default="", description="Groq Cloud API Key")

    # LLM Model Names
    PRIMARY_LLM_MODEL: str = "gemini-3.8-flash"
    ROUTER_LLM_MODEL: str = "gemini-3.8-flash-lite"
    FALLBACK_LLM_MODEL: str = "qwen/qwen3.8-27b"
    EVAL_JUDGE_MODEL: str = "qwen/qwen3.8-27b"

    # Embedding & Reranker
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"

    # Vector Storage (Qdrant)
    # If QDRANT_URL is empty, embedded mode (local directory) is used
    QDRANT_URL: str = Field(default="", description="Qdrant Cloud URL or empty for embedded")
    QDRANT_API_KEY: str = Field(default="", description="Qdrant Cloud API Key if using cloud")
    QDRANT_STORAGE_DIR: str = str(BASE_DIR / "data" / "qdrant_storage")

    # Knowledge Graph (Neo4j)
    NEO4J_URI: str = Field(default="", description="Neo4j AuraDB URI or bolt://localhost:7687")
    NEO4J_USERNAME: str = Field(default="neo4j", description="Neo4j username")
    NEO4J_PASSWORD: str = Field(default="", description="Neo4j password")

    # Local Paths
    DATA_DIR: str = str(BASE_DIR / "data")
    RAW_PDFS_DIR: str = str(BASE_DIR / "data" / "raw_pdfs")
    PROCESSED_DIR: str = str(BASE_DIR / "data" / "processed")
    CACHE_DIR: str = str(BASE_DIR / "data" / "cache")

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for path_str in [
            self.DATA_DIR,
            self.RAW_PDFS_DIR,
            self.PROCESSED_DIR,
            self.CACHE_DIR,
            self.QDRANT_STORAGE_DIR,
        ]:
            Path(path_str).mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()
settings.ensure_directories()
