# Generative AI–Powered Intelligent Document Analyzer (RAG System)

🔖 Stable Release: v2.0

A Retrieval-Augmented Generation (RAG) system for intelligent contract question answering, built using FastAPI, FAISS, Sentence Transformers, and Large Language Models.

This project ingests legal contracts from the CUAD (Contract Understanding Atticus Dataset) and PDF documents, generates vector embeddings, retrieves relevant contract clauses, and uses an LLM to generate context-aware answers with source attribution.

## 🚀 Key Features

-  Semantic Search with FAISS

-  CUAD JSON contract ingestion

-  **PDF document ingestion** (NEW in v2.0)

-  Incremental indexing for dynamic document addition

-  LLM-powered question answering (RAG)

-  FastAPI backend with OpenAPI docs

-  Source citation (contract ID, paragraph, score, filename)

-  Modular, production-ready architecture

## 🏗️ System Architecture

```bash
User Question
     │
     ▼
FastAPI (/ask-question)
     │
     ▼
Query Embedding (SentenceTransformer)
     │
     ▼
FAISS Vector Search
     │
     ▼
Top-K Relevant Contract Chunks
     │
     ▼
LLM (RAG Prompt)
     │
     ▼
Answer + Source References

```

## 📂 Project Structure

```bash
rag-document-analyzer/
│
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI endpoints
│   │
│   ├── ingestion/
│   │   ├── json_loader.py      # CUAD JSON ingestion
│   │   ├── pdf_loader.py       # PDF document ingestion
│   │   └── document_loader.py  # Unified document loader
│   │
│   ├── embeddings/
│   │   └── embedding_generator.py
│   │
│   ├── retrieval/
│   │   └── vector_store.py     # FAISS index with incremental updates
│   │
│   ├── llm/
│   │   └── qa_chain.py         # RAG QA logic
│   │
│   └── config.py               # Configuration management
│
├── data/                       # (ignored in git)
│   ├── raw_docs/
│   │   ├── CUADv1.json
│   │   ├── pdfs/               # Stored PDF files
│   │   └── uploads/            # Temporary upload storage
│   └── embeddings/
│       ├── unified_embeddings.npy
│       └── unified_metadata.json
│
├── main.py                     # CLI pipeline runner
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Dataset Setup

CUAD v1 (Contract Understanding Atticus Dataset)

Real-world legal contracts with clause-level annotations

Used widely in legal NLP research

📄 **Step-by-step dataset setup guide**:  
👉 [CUAD DataSet](https://github.com/ansh90378/rag-document-analyzer/wiki/CUAD-Dataset-Setup-Guide)

## ⚙️ Setup Instructions

```bash
1️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

```bash
2️⃣ Install Dependencies
pip install -r requirements.txt 
```

## Build Vector Index

### Using CLI Pipeline

Run the ingestion + embedding pipeline with different sources:

**Process CUAD JSON (default):**
```bash
python main.py --source json --query "What are termination conditions?"
```

**Process PDF file:**
```bash
python main.py --source pdf --pdf-path data/raw_docs/pdfs/contract.pdf --query "What is the contract about?"
```

**Process PDF directory:**
```bash
python main.py --source pdf --pdf-path data/raw_docs/pdfs/ --query "What are the terms?"
```

**Process both CUAD and PDFs:**
```bash
python main.py --source both --pdf-path data/raw_docs/pdfs/ --query "What are the terms?"
```

**Command-line options:**
- `--source`: Document source (`json`, `pdf`, or `both`)
- `--pdf-path`: Path to PDF file or directory (required for PDF processing)
- `--query`: Question to answer
- `--top-k`: Number of top results to retrieve (default: 4)

The pipeline will:
- Load documents (CUAD and/or PDFs)
- Chunk documents
- Generate embeddings
- Store vectors in FAISS
- Answer your query with source citations

## 🌐 Run API Server

```bash
uvicorn src.api.app:app --reload
```

Open Swagger UI:

```bash
http://127.0.0.1:8000/docs
```

## API Endpoints

### ✅ Health Check

```
GET /
```

Response:
```json
{
  "status": "RAG API is running 🚀",
  "version": "2.0",
  "index_loaded": true,
  "total_vectors": 1234
}
```

### 📊 Get Statistics

```
GET /stats
```

Returns vector store statistics (total vectors, embedding dimension, etc.)

### 📄 Upload PDF Document

```
POST /upload-pdf
Content-Type: multipart/form-data
```

**Request:** Upload PDF file via form data

**Response:**
```json
{
  "document_id": "1234567890_contract.pdf",
  "filename": "contract.pdf",
  "status": "success",
  "chunks_created": 45,
  "processing_time_seconds": 2.3,
  "message": "PDF processed successfully. Created 45 chunks."
}
```

**Example cURL:**
```bash
curl -X POST "http://127.0.0.1:8000/upload-pdf" \
  -F "file=@/path/to/contract.pdf"
```

### 📦 Upload Multiple PDFs (Batch)

```
POST /upload-pdf-batch
Content-Type: multipart/form-data
```

**Request:** Upload multiple PDF files (max 10 per batch)

**Response:**
```json
{
  "total_files": 3,
  "processed": 3,
  "failed": 0,
  "results": [
    {
      "filename": "contract1.pdf",
      "status": "success",
      "document_id": "1234567890_contract1.pdf",
      "chunks_created": 45
    }
  ]
}
```

### 📋 List Uploaded Documents

```
GET /documents
```

**Response:**
```json
{
  "total_documents": 5,
  "documents": [
    {
      "document_id": "1234567890_contract.pdf",
      "filename": "contract.pdf",
      "source_type": "PDF",
      "upload_time": "2024-01-15T10:30:00",
      "chunks_count": 45,
      "file_size": 1024000
    }
  ]
}
```

### ❓ Ask a Contract Question

```
POST /ask-question
```

**Request Body:**
```json
{
  "question": "What does the contract say about termination?",
  "top_k": 4
}
```

**Sample Response:**
```json
{
  "answer": "The contract allows termination with prior written notice under specific conditions...",
  "sources": [
    {
      "contract_id": "290",
      "paragraph_id": 0,
      "score": 0.749,
      "source_type": "CUAD",
      "filename": null
    },
    {
      "contract_id": "abc123def456",
      "paragraph_id": 0,
      "score": 0.721,
      "source_type": "PDF",
      "filename": "contract.pdf"
    }
  ]
}
```

**Example cURL:**
```bash
curl -X POST "http://127.0.0.1:8000/ask-question" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the termination rights?",
    "top_k": 4
  }'
```

## 🛠️ Tech Stack

- Python 3.10+
- FastAPI
- FAISS (with incremental indexing)
- Sentence Transformers
- Hugging Face Transformers
- PyPDF (PDF text extraction)
- CUAD Dataset
- Uvicorn
- python-dotenv (configuration management)

## Engineering Highlights

Separation of concerns (ingestion, embeddings, retrieval, API)

Vector search optimized for large document sets

Structured API responses with validation

Production-ready FastAPI patterns

Easily extensible for PDFs, multi-tenant search, or cloud deployment

## 📝 PDF Ingestion Details

### Supported Features

- ✅ Single PDF file upload
- ✅ Batch PDF processing
- ✅ Automatic text extraction from all pages
- ✅ Page-level metadata tracking
- ✅ File size validation (default: 50MB max)
- ✅ Error handling for encrypted/corrupted PDFs
- ✅ Incremental indexing (add PDFs without rebuilding entire index)
- ✅ Unified search across CUAD and PDF documents

### PDF Processing Workflow

1. **Upload:** PDF file uploaded via API or placed in `data/raw_docs/pdfs/`
2. **Extraction:** Text extracted from all pages using PyPDF
3. **Chunking:** Text split into chunks (default: 800 chars with 200 char overlap)
4. **Embedding:** Generate vector embeddings using Sentence Transformers
5. **Indexing:** Add to FAISS vector store incrementally
6. **Search:** Query across all documents (CUAD + PDFs)

### Configuration

Create a `.env` file to customize settings:

```env
# PDF Processing
MAX_PDF_SIZE_MB=50
CHUNK_SIZE=800
CHUNK_OVERLAP=200

# CUAD Pre-loading
ENABLE_CUAD_PRELOAD=true

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDINGS_FILE=unified_embeddings.npy
METADATA_FILE=unified_metadata.json

# LLM
LLM_MODEL=google/flan-t5-base
LLM_MAX_TOKENS=256
LLM_MAX_INPUT_TOKENS=512

# API
API_HOST=0.0.0.0
API_PORT=8000
```

## Future Enhancements

- Streaming responses
- Cloud deployment (AWS/GCP)
- Multi-document comparison
- Evaluation metrics (Recall@K, MRR)
- Document deletion/update endpoints

# 👤 Author

Ansh Chauhan

AI / ML Engineer (Aspiring)

Focused on Applied LLMs, RAG systems, and production AI
