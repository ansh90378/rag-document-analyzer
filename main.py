from src.ingestion.json_loader import ingest_cuad_documents
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import FAISSVectorStore
from src.llm.qa_chain import RAGQAChain


def run_rag_pipeline(query: str, top_k: int = 4):
    # 1️⃣ Load documents
    documents = ingest_cuad_documents()

    # 2️⃣ Embeddings
    embedder = EmbeddingGenerator()
    embeddings = embedder.generate_embeddings(documents)

    metadata = [
        {**doc["metadata"], "text": doc["text"]}
        for doc in documents
    ]
    embedder.save_embeddings(embeddings, metadata)

    # 3️⃣ Vector store
    vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
    vector_store.load_embeddings(
        embeddings_path="data/embeddings/cuad_embeddings.npy",
        metadata_path="data/embeddings/cuad_metadata.json"
    )

    # 4️⃣ Query embedding
    query_embedding = embedder.model.encode(
        query,
        normalize_embeddings=True
    )

    # 5️⃣ Retrieve top-k chunks
    retrieved_docs = vector_store.search(query_embedding, k=top_k)

    # 6️⃣ Generate answer
    qa_chain = RAGQAChain()
    answer = qa_chain.generate_answer(query, retrieved_docs)

    return answer, retrieved_docs


if __name__ == "__main__":
    query = "What are the termination conditions for breach of contract?"

    answer, sources = run_rag_pipeline(query)

    print("\n🧠 ANSWER:\n")
    print(answer)

    print("\n📌 SOURCES:\n")
    for src in sources:
        print(
            f"- Contract {src['contract_id']} | "
            f"Paragraph {src['paragraph_id']} | "
            f"Score: {src['score']:.3f}"
        )