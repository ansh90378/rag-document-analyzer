import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.ingestion.json_loader import chunk_text


def load_pdf(pdf_path: str) -> Dict:
    """
    Extract text and metadata from a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with 'text' and 'metadata' keys
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        PdfReadError: If PDF is corrupted or encrypted
        ValueError: If PDF extraction fails
    """
    pdf_path_obj = Path(pdf_path)
    
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    if not pdf_path_obj.suffix.lower() == '.pdf':
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    try:
        reader = PdfReader(str(pdf_path_obj))
        
        # Check if PDF is encrypted
        if reader.is_encrypted:
            raise PdfReadError(f"PDF is encrypted and cannot be read: {pdf_path}")
        
        # Extract text from all pages
        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    pages_text.append((page_num, page_text))
            except Exception as e:
                print(f"Warning: Failed to extract text from page {page_num} of {pdf_path}: {e}")
                continue
        
        if not pages_text:
            raise ValueError(f"No text could be extracted from PDF: {pdf_path}")
        
        # Combine all page texts
        full_text = "\n\n".join([text for _, text in pages_text])
        
        # Calculate file size
        file_size = pdf_path_obj.stat().st_size
        
        # Generate unique document ID from file path hash
        doc_id = hashlib.md5(str(pdf_path_obj.resolve()).encode()).hexdigest()
        
        # Extract PDF metadata if available
        pdf_metadata = {}
        if reader.metadata:
            pdf_metadata = {
                "title": reader.metadata.get("/Title", ""),
                "author": reader.metadata.get("/Author", ""),
                "subject": reader.metadata.get("/Subject", ""),
                "creator": reader.metadata.get("/Creator", ""),
            }
        
        return {
            "text": full_text,
            "metadata": {
                "source": "PDF",
                "document_id": doc_id,
                "filename": pdf_path_obj.name,
                "file_path": str(pdf_path_obj),
                "total_pages": len(reader.pages),
                "file_size": file_size,
                "pages_with_text": len(pages_text),
                **pdf_metadata
            }
        }
        
    except PdfReadError as e:
        raise PdfReadError(f"Failed to read PDF {pdf_path}: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error processing PDF {pdf_path}: {e}")


def load_pdf_directory(directory: str) -> List[Dict]:
    """
    Batch process all PDF files in a directory.
    
    Args:
        directory: Path to directory containing PDF files
        
    Returns:
        List of dictionaries with 'text' and 'metadata' keys
    """
    directory_path = Path(directory)
    
    if not directory_path.exists() or not directory_path.is_dir():
        raise ValueError(f"Directory does not exist or is not a directory: {directory}")
    
    pdf_files = list(directory_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"Warning: No PDF files found in {directory}")
        return []
    
    documents = []
    failed_files = []
    
    for pdf_file in pdf_files:
        try:
            doc = load_pdf(str(pdf_file))
            documents.append(doc)
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            failed_files.append(str(pdf_file))
            continue
    
    print(f"Successfully processed {len(documents)} PDF(s) from {directory}")
    if failed_files:
        print(f"Failed to process {len(failed_files)} PDF(s)")
    
    return documents


def ingest_pdf_documents(
    pdf_path: str,
    chunk_size: int = 800,
    overlap: int = 200
) -> List[Dict]:
    """
    Main PDF ingestion function - parallel to ingest_cuad_documents.
    
    Loads PDF(s), chunks text, and returns standardized document format.
    
    Args:
        pdf_path: Path to PDF file or directory containing PDFs
        chunk_size: Size of text chunks in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of document dictionaries with 'text' and 'metadata' keys
    """
    pdf_path_obj = Path(pdf_path)
    
    # Determine if it's a file or directory
    if pdf_path_obj.is_file():
        pdf_docs = [load_pdf(pdf_path)]
    elif pdf_path_obj.is_dir():
        pdf_docs = load_pdf_directory(pdf_path)
    else:
        raise ValueError(f"Path does not exist: {pdf_path}")
    
    if not pdf_docs:
        return []
    
    # Chunk each PDF document
    documents = []
    
    for pdf_doc in pdf_docs:
        text = pdf_doc["text"]
        base_metadata = pdf_doc["metadata"]
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        
        # Create document entries for each chunk
        for chunk_id, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "metadata": {
                    **base_metadata,
                    "chunk_id": chunk_id,
                    "total_chunks": len(chunks),
                    # For backward compatibility with Source model
                    "contract_id": base_metadata["document_id"],
                    "paragraph_id": 0,  # PDFs don't have paragraph structure like CUAD
                }
            })
    
    return documents
