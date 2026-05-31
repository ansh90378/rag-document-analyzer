import argparse
from pathlib import Path
from typing import Optional

from src.config import Config
from src.ingestion.json_loader import ingest_cuad_documents
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import FAISSVectorStore
from src.llm.qa_chain import RAGQAChain

# Optional PDF imports
try:
    from src.ingestion.pdf_loader import ingest_pdf_documents
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PDF support not available. Install pypdf: pip install pypdf")


def run_rag_pipeline(
    query: str,
    top_k: int = 4,
    source: str = "json",
    pdf_path: Optional[str] = None
):
    """
    Run RAG pipeline with specified document source.
    
    Args:
        query: Question to answer
        top_k: Number of top results to retrieve
        source: Document source type ("json", "pdf", or "both")
        pdf_path: Path to PDF file or directory (required if source is "pdf")
    """
    print(f"\n[INFO] Running RAG Pipeline (source: {source})")
    print(f"Query: {query}\n")
    
    # 1. Load documents
    print("[1/6] Loading documents...")
    documents = []
    
    if source in ["json", "both"]:
        try:
            cuad_docs = ingest_cuad_documents()
            documents.extend(cuad_docs)
            print(f"   [+] Loaded {len(cuad_docs)} CUAD documents")
        except FileNotFoundError as e:
            print(f"   [!] CUAD file not found: {e}")
            if source == "json":
                raise
        except Exception as e:
            print(f"   [-] Error loading CUAD: {e}")
            if source == "json":
                raise
    
    if source in ["pdf", "both"]:
        if not PDF_AVAILABLE:
            raise ImportError(
                "PDF support not available. Install pypdf: pip install pypdf"
            )
        
        if not pdf_path:
            raise ValueError("pdf_path is required when source is 'pdf' or 'both'")
        
        try:
            pdf_docs = ingest_pdf_documents(
                pdf_path=pdf_path,
                chunk_size=Config.CHUNK_SIZE,
                overlap=Config.CHUNK_OVERLAP
            )
            documents.extend(pdf_docs)
            print(f"   [+] Loaded {len(pdf_docs)} PDF documents")
        except Exception as e:
            print(f"   [-] Error loading PDFs: {e}")
            if source == "pdf":
                raise
    
    if not documents:
        raise ValueError("No documents loaded. Cannot proceed.")
    
    print(f"   Total documents: {len(documents)}\n")
    
    # 2. Generate embeddings
    print("[2/6] Generating embeddings...")
    embedder = EmbeddingGenerator(model_name=Config.EMBEDDING_MODEL)
    embeddings = embedder.generate_embeddings(documents)
    
    metadata = [
        {**doc["metadata"], "text": doc["text"]}
        for doc in documents
    ]
    
    # Save embeddings
    embedder.save_embeddings(
        embeddings,
        metadata,
        output_dir=str(Config.EMBEDDINGS_DIR),
        filename_prefix="unified"
    )
    print(f"   [+] Generated {embeddings.shape[0]} embeddings\n")
    
    # 3. Vector store
    print("[3/6] Initializing vector store...")
    vector_store = FAISSVectorStore(embedding_dim=Config.EMBEDDING_DIM)
    vector_store.load_embeddings(
        embeddings_path=str(Config.get_embeddings_path()),
        metadata_path=str(Config.get_metadata_path())
    )
    print(f"   [+] Loaded {vector_store.index.ntotal} vectors\n")
    
    # 4. Query embedding
    print("[4/6] Processing query...")
    query_embedding = embedder.model.encode(
        query,
        normalize_embeddings=True
    )
    
    # 5. Retrieve top-k chunks
    retrieved_docs = vector_store.search(query_embedding, k=top_k)
    print(f"   [+] Retrieved {len(retrieved_docs)} relevant chunks\n")
    
    # 6. Generate answer
    print("[5/6] Generating answer...")
    qa_chain = RAGQAChain(
        model_name=Config.LLM_MODEL,
        max_tokens=Config.LLM_MAX_TOKENS
    )
    answer = qa_chain.generate_answer(query, retrieved_docs)
    
    return answer, retrieved_docs


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="RAG Document Analyzer - CLI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process CUAD JSON and ask a question
  python main.py --source json --query "What are termination conditions?"

  # Process a PDF file
  python main.py --source pdf --pdf-path data/raw_docs/pdfs/contract.pdf --query "What is the contract about?"

  # Process both CUAD and PDFs
  python main.py --source both --pdf-path data/raw_docs/pdfs/ --query "What are the terms?"
        """
    )
    
    parser.add_argument(
        "--source",
        type=str,
        choices=["json", "pdf", "both"],
        default="json",
        help="Document source type: json (CUAD), pdf, or both (default: json)"
    )
    
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="Path to PDF file or directory (required if source is 'pdf' or 'both')"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default="What are the termination conditions for breach of contract?",
        help="Question to answer (default: example question)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of top results to retrieve (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.source in ["pdf", "both"] and not args.pdf_path:
        parser.error("--pdf-path is required when --source is 'pdf' or 'both'")
    
    if args.pdf_path:
        pdf_path_obj = Path(args.pdf_path)
        if not pdf_path_obj.exists():
            parser.error(f"PDF path does not exist: {args.pdf_path}")
    
    try:
        # Run pipeline
        answer, sources = run_rag_pipeline(
            query=args.query,
            top_k=args.top_k,
            source=args.source,
            pdf_path=args.pdf_path
        )
        
        # Print results
        print("\n" + "="*60)
        print("ANSWER:\n")
        print(answer)
        print("\n" + "="*60)
        print("\nSOURCES:\n")
        
        for src in sources:
            source_type = src.get("source", "unknown")
            doc_id = src.get("contract_id", src.get("document_id", "N/A"))
            para_id = src.get("paragraph_id", 0)
            score = src.get("score", 0.0)
            filename = src.get("filename", "")
            
            if filename:
                print(f"- {source_type}: {filename} | Paragraph {para_id} | Score: {score:.3f}")
            else:
                print(f"- {source_type} | Document {doc_id} | Paragraph {para_id} | Score: {score:.3f}")
        
        print("\n" + "="*60 + "\n")
    
    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        raise


if __name__ == "__main__":
    main()
