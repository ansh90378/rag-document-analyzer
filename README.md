# 📄 RAG Document Analyzer (CUAD)

A Retrieval-Augmented Generation (RAG) system for intelligent contract question answering, built using FastAPI, FAISS, Sentence Transformers, and Large Language Models.

This project ingests legal contracts from the CUAD (Contract Understanding Atticus Dataset), generates vector embeddings, retrieves relevant contract clauses, and uses an LLM to generate context-aware answers with source attribution.

## 🚀 Key Features

- 🔍 Semantic Search with FAISS

- 📚 CUAD JSON contract ingestion

- 🧠 LLM-powered question answering (RAG)

- ⚡ FastAPI backend with OpenAPI docs

- 📌 Source citation (contract ID, paragraph, score)

- 🧩 Modular, production-ready architecture

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

## 📊 Dataset

CUAD v1 (Contract Understanding Atticus Dataset)

Real-world legal contracts with clause-level annotations

Used widely in legal NLP research

📌 Dataset is not committed to GitHub (handled via .gitignore).

# Improve - Tell about the dataset and get details who they can get it from. 

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
# (add acelerate into requirements.txt - add this change through git)

