from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from pathlib import Path
import time
from datetime import datetime

from src.config import Config
from src.ingestion.json_loader import ingest_cuad_documents
from src.ingestion.pdf_loader import ingest_pdf_documents
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import FAISSVectorStore
from src.llm.qa_chain import RAGQAChain

app = FastAPI(
    title="RAG Document Analyzer API",
    description="Generative AI powered contract question answering system with PDF support",
    version="2.0"
)

# ---------- Global Components (initialized on startup) ----------

embedder: Optional[EmbeddingGenerator] = None
vector_store: Optional[FAISSVectorStore] = None
qa_chain: Optional[RAGQAChain] = None
documents_registry: List[dict] = []  # Track uploaded documents


def initialize_components():
    """Initialize global components."""
    global embedder, vector_store, qa_chain
    
    # Initialize embedding generator
    embedder = EmbeddingGenerator(model_name=Config.EMBEDDING_MODEL)
    
    # Initialize vector store
    vector_store = FAISSVectorStore(embedding_dim=Config.EMBEDDING_DIM)
    
    # Try to load existing embeddings
    embeddings_path = Config.get_embeddings_path()
    metadata_path = Config.get_metadata_path()
    
    if embeddings_path.exists() and metadata_path.exists():
        try:
            vector_store.load_embeddings(
                embeddings_path=str(embeddings_path),
                metadata_path=str(metadata_path)
            )
            print(f"Loaded existing index with {vector_store.index.ntotal} vectors")
        except Exception as e:
            print(f"Warning: Failed to load existing embeddings: {e}")
    
    # Initialize QA chain
    qa_chain = RAGQAChain(
        model_name=Config.LLM_MODEL,
        max_tokens=Config.LLM_MAX_TOKENS,
        max_input_tokens=Config.LLM_MAX_INPUT_TOKENS
    )


def load_cuad_if_enabled():
    """Load CUAD documents if enabled in config."""
    if not Config.ENABLE_CUAD_PRELOAD:
        print("CUAD pre-loading is disabled. Skipping...")
        return
    
    try:
        print("Loading CUAD documents...")
        documents = ingest_cuad_documents()
        
        if not documents:
            print("Warning: No CUAD documents loaded")
            return
        
        # Generate embeddings
        embeddings = embedder.generate_embeddings(documents)
        
        metadata = [
            {**doc["metadata"], "text": doc["text"]}
            for doc in documents
        ]
        
        # Add to vector store
        if vector_store.index.ntotal == 0:
            # First load - use load_embeddings
            embedder.save_embeddings(
                embeddings,
                metadata,
                output_dir=str(Config.EMBEDDINGS_DIR),
                filename_prefix="unified"
            )
            vector_store.load_embeddings(
                embeddings_path=str(Config.get_embeddings_path()),
                metadata_path=str(Config.get_metadata_path())
            )
        else:
            # Incremental add
            vector_store.add_embeddings(embeddings, metadata)
            vector_store.save_index(
                embeddings_path=str(Config.get_embeddings_path()),
                metadata_path=str(Config.get_metadata_path())
            )
        
        print(f"Loaded {len(documents)} CUAD documents")
    except FileNotFoundError as e:
        print(f"Warning: CUAD file not found: {e}")
    except Exception as e:
        print(f"Error loading CUAD documents: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize components on API startup."""
    initialize_components()
    load_cuad_if_enabled()


# ---------- Request / Response Models ----------

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 4


class Source(BaseModel):
    contract_id: str
    paragraph_id: int
    score: float
    source_type: Optional[str] = None
    filename: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]


class PDFUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_created: int
    processing_time_seconds: float
    message: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    source_type: str
    upload_time: str
    chunks_count: int
    file_size: Optional[int] = None


class DocumentListResponse(BaseModel):
    total_documents: int
    documents: List[DocumentInfo]


class StatsResponse(BaseModel):
    total_vectors: int
    embedding_dim: int
    metadata_count: int
    index_type: str


# ---------- Helper Functions ----------

def validate_pdf_file(file: UploadFile) -> tuple:
    """Validate uploaded PDF file."""
    # Check file extension
    if not file.filename:
        return False, "No filename provided"
    
    if not file.filename.lower().endswith('.pdf'):
        return False, "File must be a PDF (.pdf extension)"
    
    # Check MIME type if available
    if file.content_type and file.content_type != 'application/pdf':
        return False, f"Invalid MIME type: {file.content_type}"
    
    return True, ""


def process_pdf_file(file_path: Path, document_id: str) -> dict:
    """Process a single PDF file and add to index."""
    start_time = time.time()
    
    try:
        # Ingest PDF
        documents = ingest_pdf_documents(
            pdf_path=str(file_path),
            chunk_size=Config.CHUNK_SIZE,
            overlap=Config.CHUNK_OVERLAP
        )
        
        if not documents:
            raise ValueError("No text could be extracted from PDF")
        
        # Generate embeddings
        embeddings = embedder.generate_embeddings_incremental(
            documents,
            show_progress=False
        )
        
        # Prepare metadata
        metadata = [
            {**doc["metadata"], "text": doc["text"]}
            for doc in documents
        ]
        
        # Add to vector store
        vector_store.add_embeddings(embeddings, metadata)
        
        # Save updated index
        vector_store.save_index(
            embeddings_path=str(Config.get_embeddings_path()),
            metadata_path=str(Config.get_metadata_path())
        )
        
        processing_time = time.time() - start_time
        
        return {
            "document_id": document_id,
            "chunks_created": len(documents),
            "processing_time": processing_time,
            "status": "success"
        }
    
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )


# ---------- Endpoints ----------

@app.get("/")
def health_check():
    """Health check endpoint."""
    stats = vector_store.get_stats() if vector_store else {}
    return {
        "status": "RAG API is running 🚀",
        "version": "2.0",
        "index_loaded": vector_store is not None and vector_store.index.ntotal > 0,
        "total_vectors": stats.get("total_vectors", 0)
    }


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """Get vector store statistics."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    stats = vector_store.get_stats()
    return StatsResponse(**stats)


@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a single PDF file for processing.
    
    The PDF will be processed, chunked, embedded, and added to the vector index.
    """
    if not embedder or not vector_store:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Validate file
    is_valid, error_msg = validate_pdf_file(file)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Check file size
    file_contents = await file.read()
    file_size = len(file_contents)
    
    if file_size > Config.MAX_PDF_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {Config.MAX_PDF_SIZE_MB}MB"
        )
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    
    # Generate document ID
    document_id = f"{int(time.time())}_{file.filename}"
    
    # Save uploaded file
    upload_path = Config.PDF_UPLOAD_DIR / file.filename
    
    # Handle filename conflicts
    counter = 1
    original_upload_path = upload_path
    while upload_path.exists():
        stem = original_upload_path.stem
        suffix = original_upload_path.suffix
        upload_path = Config.PDF_UPLOAD_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    
    # Write file
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    with open(upload_path, "wb") as f:
        f.write(file_contents)
    
    # Move to permanent storage
    storage_path = Config.PDF_STORAGE_DIR / upload_path.name
    shutil.move(str(upload_path), str(storage_path))
    
    try:
        # Process PDF
        result = process_pdf_file(storage_path, document_id)
        
        # Register document
        documents_registry.append({
            "document_id": document_id,
            "filename": file.filename,
            "source_type": "PDF",
            "upload_time": datetime.now().isoformat(),
            "chunks_count": result["chunks_created"],
            "file_path": str(storage_path),
            "file_size": file_size
        })
        
        return PDFUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="success",
            chunks_created=result["chunks_created"],
            processing_time_seconds=result["processing_time"],
            message=f"PDF processed successfully. Created {result['chunks_created']} chunks."
        )
    
    except Exception as e:
        # Clean up file on error
        if storage_path.exists():
            storage_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-pdf-batch")
async def upload_pdf_batch(files: List[UploadFile] = File(...)):
    """
    Upload multiple PDF files for batch processing.
    
    Returns list of results for each file.
    """
    if not embedder or not vector_store:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for file in files:
        try:
            # Validate file
            is_valid, error_msg = validate_pdf_file(file)
            if not is_valid:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": error_msg
                })
                continue
            
            # Process single file (reuse existing endpoint logic)
            response = await upload_pdf(BackgroundTasks(), file)
            results.append({
                "filename": file.filename,
                "status": "success",
                "document_id": response.document_id,
                "chunks_created": response.chunks_created
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename if file.filename else "unknown",
                "status": "error",
                "message": str(e)
            })
    
    return {
        "total_files": len(files),
        "processed": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "results": results
    }


@app.get("/documents", response_model=DocumentListResponse)
def list_documents():
    """List all ingested documents with metadata."""
    documents_info = [
        DocumentInfo(
            document_id=doc["document_id"],
            filename=doc["filename"],
            source_type=doc["source_type"],
            upload_time=doc["upload_time"],
            chunks_count=doc["chunks_count"],
            file_size=doc.get("file_size")
        )
        for doc in documents_registry
    ]
    
    return DocumentListResponse(
        total_documents=len(documents_info),
        documents=documents_info
    )


@app.post("/ask-question", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    """Ask a question about the ingested documents."""
    if not embedder or not vector_store or not qa_chain:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed. Please upload documents first."
        )
    
    # Generate query embedding
    query_embedding = embedder.model.encode(
        payload.question,
        normalize_embeddings=True
    )
    
    # Search
    retrieved_docs = vector_store.search(
        query_embedding,
        k=payload.top_k
    )
    
    if not retrieved_docs:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found"
        )
    
    # Generate answer
    answer = qa_chain.generate_answer(
        payload.question,
        retrieved_docs
    )
    
    # Format sources
    sources = []
    for d in retrieved_docs:
        source = Source(
            contract_id=str(d.get("contract_id", d.get("document_id", ""))),
            paragraph_id=int(d.get("paragraph_id", 0)),
            score=float(round(d.get("score", 0.0), 3)),
            source_type=d.get("source", "unknown"),
            filename=d.get("filename", None)
        )
        sources.append(source)
    
    return AnswerResponse(
        answer=answer,
        sources=sources
    )
