from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from src.ingestion.json_loader import ingest_cuad_documents
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import FAISSVectorStore
from src.llm.qa_chain import RAGQAChain

app = FastAPI(
    title="RAG Document Analyzer API",
    description="Generative AI powered contract question answering system",
    version="1.0"
)

# ---------- Load components once (important for performance) ----------

documents = ingest_cuad_documents()

embedder = EmbeddingGenerator()
embeddings = embedder.generate_embeddings(documents)

metadata = [
    {**doc["metadata"], "text": doc["text"]}
    for doc in documents
]

embedder.save_embeddings(embeddings, metadata)

vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
vector_store.load_embeddings(
    embeddings_path="data/embeddings/cuad_embeddings.npy",
    metadata_path="data/embeddings/cuad_metadata.json"
)

qa_chain = RAGQAChain()

# ---------- Request / Response Models ----------

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 4


class Source(BaseModel):
    contract_id: str
    paragraph_id: int
    score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]

# ---------- Endpoints ----------

@app.get("/")
def health_check():
    return {"status": "RAG API is running 🚀"}


@app.post("/ask-question", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    query_embedding = embedder.model.encode(
        payload.question,
        normalize_embeddings=True
    )

    retrieved_docs = vector_store.search(
        query_embedding,
        k=payload.top_k
    )

    answer = qa_chain.generate_answer(
        payload.question,
        retrieved_docs
    )

    sources = [
        {
            "contract_id": str(d["contract_id"]),   # cast to string (According to Pydantic model)
            "paragraph_id": int(d["paragraph_id"]), # Explicit
            "score": float(round(d["score"], 3))    # Explicit
        }
        for d in retrieved_docs
    ]

    return {
        "answer": answer,
        "sources": sources
    }