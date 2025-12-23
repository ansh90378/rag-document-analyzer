# Generative AI–Powered Intelligent Document Analyzer (RAG System)

🔖 Stable Release: [v1.0](https://github.com/ansh90378/rag-document-analyzer/tree/v1.0)

A Retrieval-Augmented Generation (RAG) system for intelligent contract question answering, built using FastAPI, FAISS, Sentence Transformers, and Large Language Models.

This project ingests legal contracts from the CUAD (Contract Understanding Atticus Dataset), generates vector embeddings, retrieves relevant contract clauses, and uses an LLM to generate context-aware answers with source attribution.

## 🚀 Key Features

-  Semantic Search with FAISS

-  CUAD JSON contract ingestion

-  LLM-powered question answering (RAG)

-  FastAPI backend with OpenAPI docs

-  Source citation (contract ID, paragraph, score)

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
│   │   └── json_loader.py      # CUAD JSON ingestion
│   │
│   ├── embeddings/
│   │   └── embedding_generator.py
│   │
│   ├── retrieval/
│   │   └── vector_store.py     # FAISS index
│   │
│   └── llm/
│       └── qa_chain.py         # RAG QA logic
│
├── data/                       # (ignored in git)
│
├── main.py                     # Pipeline runner
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

Run the ingestion + embedding pipeline:

```bash
python main.py
```

This will:

Load CUAD contracts

Chunk documents

Generate embeddings

Store vectors in FAISS

## 🌐 Run API Server

```bash
uvicorn src.api.app:app --reload
```

Open Swagger UI:

```bash
http://127.0.0.1:8000/docs
```

## API Endpoints
✅ Health Check

```
GET /
```

❓ Ask a Contract Question

```
POST /ask-question
```

Request Body

```
{
  "question": "What does the contract say about termination?",
  "top_k": 4
}
```

Sample Response

```
{
  "answer": "The contract allows termination with prior written notice under specific conditions...",
  "sources": [
    {
      "contract_id": "290",
      "paragraph_id": 0,
      "score": 0.749
    }
  ]
}
```

### Example cURL

```
curl -X POST "http://127.0.0.1:8000/ask-question" \
-H "Content-Type: application/json" \
-d '{
  "question": "What are the termination rights?",
  "top_k": 4
}'
```

## 🛠️ Tech Stack

Python 3.10+

FastAPI

FAISS

Sentence Transformers

Hugging Face Transformers

CUAD Dataset

Uvicorn

## Engineering Highlights

Separation of concerns (ingestion, embeddings, retrieval, API)

Vector search optimized for large document sets

Structured API responses with validation

Production-ready FastAPI patterns

Easily extensible for PDFs, multi-tenant search, or cloud deployment

## Future Enhancements

PDF ingestion

Streaming responses

Cloud deployment (AWS/GCP)

Multi-document comparison

Evaluation metrics (Recall@K, MRR)

# 👤 Author

Ansh Chauhan

AI / ML Engineer (Aspiring)

Focused on Applied LLMs, RAG systems, and production AI
