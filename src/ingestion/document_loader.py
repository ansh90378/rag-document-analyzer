"""
Unified document loader interface supporting multiple document formats.
"""
from pathlib import Path
from typing import List, Dict, Optional
import json

from src.ingestion.json_loader import ingest_cuad_documents
from src.ingestion.pdf_loader import ingest_pdf_documents


def get_supported_formats() -> List[str]:
    """
    Return list of supported document formats.
    
    Returns:
        List of format strings: ["json", "pdf"]
    """
    return ["json", "pdf"]


def detect_document_type(source: str) -> Optional[str]:
    """
    Auto-detect document type from source path.
    
    Args:
        source: Path to file or directory
        
    Returns:
        Document type string ("json", "pdf") or None if cannot detect
    """
    source_path = Path(source)
    
    if not source_path.exists():
        return None
    
    # Check file extension
    if source_path.is_file():
        suffix = source_path.suffix.lower()
        if suffix == ".json":
            return "json"
        elif suffix == ".pdf":
            return "pdf"
    
    # Check directory contents
    elif source_path.is_dir():
        json_files = list(source_path.glob("*.json"))
        pdf_files = list(source_path.glob("*.pdf"))
        
        if json_files and not pdf_files:
            return "json"
        elif pdf_files and not json_files:
            return "pdf"
        elif json_files and pdf_files:
            # Mixed directory - default to PDF if both exist
            # User should specify doc_type explicitly
            return None
    
    return None


def load_documents(
    source: str,
    doc_type: str = "auto",
    **kwargs
) -> List[Dict]:
    """
    Auto-detect and load documents from various formats.
    
    Args:
        source: Path to document file or directory
        doc_type: Document type ("json", "pdf", "auto")
        **kwargs: Additional arguments passed to ingestion functions
                  (e.g., chunk_size, overlap for PDF)
        
    Returns:
        List of document dictionaries with 'text' and 'metadata' keys
        
    Raises:
        ValueError: If document type cannot be determined or is unsupported
        FileNotFoundError: If source path doesn't exist
    """
    if doc_type == "auto":
        doc_type = detect_document_type(source)
        
        if doc_type is None:
            raise ValueError(
                f"Cannot auto-detect document type for {source}. "
                f"Please specify doc_type explicitly (json or pdf)."
            )
    
    if doc_type not in get_supported_formats():
        raise ValueError(
            f"Unsupported document type: {doc_type}. "
            f"Supported formats: {get_supported_formats()}"
        )
    
    if doc_type == "json":
        # For JSON, we assume it's CUAD format
        # CUAD loader doesn't accept source path, it uses hardcoded path
        # So we'll use the existing function as-is
        return ingest_cuad_documents()
    
    elif doc_type == "pdf":
        # Extract PDF-specific kwargs
        chunk_size = kwargs.get("chunk_size", 800)
        overlap = kwargs.get("overlap", 200)
        
        return ingest_pdf_documents(
            pdf_path=source,
            chunk_size=chunk_size,
            overlap=overlap
        )
    
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")


def load_documents_batch(
    sources: List[str],
    doc_types: Optional[List[str]] = None,
    **kwargs
) -> List[Dict]:
    """
    Load multiple document sources and combine them.
    
    Args:
        sources: List of paths to document files or directories
        doc_types: List of document types (one per source), or None for auto-detect
        **kwargs: Additional arguments passed to ingestion functions
        
    Returns:
        Combined list of all documents
    """
    if doc_types is None:
        doc_types = ["auto"] * len(sources)
    
    if len(doc_types) != len(sources):
        raise ValueError(
            f"Number of doc_types ({len(doc_types)}) must match "
            f"number of sources ({len(sources)})"
        )
    
    all_documents = []
    
    for source, doc_type in zip(sources, doc_types):
        try:
            documents = load_documents(source, doc_type=doc_type, **kwargs)
            all_documents.extend(documents)
        except Exception as e:
            print(f"Warning: Failed to load documents from {source}: {e}")
            continue
    
    return all_documents
