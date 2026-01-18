"""
Configuration management for RAG Document Analyzer.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Application configuration with environment variable support."""
    
    # Data directories
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
    RAW_DOCS_DIR: Path = DATA_DIR / "raw_docs"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    
    # PDF-specific paths
    PDF_UPLOAD_DIR: Path = RAW_DOCS_DIR / "uploads"
    PDF_STORAGE_DIR: Path = RAW_DOCS_DIR / "pdfs"
    
    # CUAD dataset path
    CUAD_JSON_PATH: Path = RAW_DOCS_DIR / "CUADv1.json"
    
    # Embedding files
    EMBEDDINGS_FILE: str = os.getenv("EMBEDDINGS_FILE", "unified_embeddings.npy")
    METADATA_FILE: str = os.getenv("METADATA_FILE", "unified_metadata.json")
    
    # Chunking parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # PDF processing
    MAX_PDF_SIZE_MB: int = int(os.getenv("MAX_PDF_SIZE_MB", "50"))
    MAX_PDF_SIZE_BYTES: int = MAX_PDF_SIZE_MB * 1024 * 1024
    
    # Embedding model
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))
    
    # LLM model
    LLM_MODEL: str = os.getenv("LLM_MODEL", "google/flan-t5-base")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "256"))
    
    # API configuration
    ENABLE_CUAD_PRELOAD: bool = os.getenv("ENABLE_CUAD_PRELOAD", "true").lower() == "true"
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Vector store
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "faiss")
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DOCS_DIR,
            cls.EMBEDDINGS_DIR,
            cls.PROCESSED_DIR,
            cls.PDF_UPLOAD_DIR,
            cls.PDF_STORAGE_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_embeddings_path(cls) -> Path:
        """Get full path to embeddings file."""
        return cls.EMBEDDINGS_DIR / cls.EMBEDDINGS_FILE
    
    @classmethod
    def get_metadata_path(cls) -> Path:
        """Get full path to metadata file."""
        return cls.EMBEDDINGS_DIR / cls.METADATA_FILE


# Initialize directories on import
Config.ensure_directories()
