"""
Test script for RAG Document Analyzer System
Tests CLI pipeline, PDF ingestion, and API endpoints
"""
import sys
import requests
import json
import time
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_success(msg):
    print(f"{Colors.GREEN}[+] {msg}{Colors.END}")


def print_error(msg):
    print(f"{Colors.RED}[-] {msg}{Colors.END}")


def print_info(msg):
    print(f"{Colors.BLUE}[i] {msg}{Colors.END}")


def print_warning(msg):
    print(f"{Colors.YELLOW}[!] {msg}{Colors.END}")


def test_imports():
    """Test that all required modules can be imported."""
    print_info("Testing imports...")
    try:
        from src.ingestion.json_loader import ingest_cuad_documents
        from src.ingestion.pdf_loader import ingest_pdf_documents
        from src.embeddings.embedding_generator import EmbeddingGenerator
        from src.retrieval.vector_store import FAISSVectorStore
        from src.llm.qa_chain import RAGQAChain
        from src.config import Config
        print_success("All imports successful")
        return True
    except ImportError as e:
        print_error(f"Import failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print_info("Testing configuration...")
    try:
        from src.config import Config
        
        # Check directories exist
        if not Config.DATA_DIR.exists():
            print_error(f"Data directory not found: {Config.DATA_DIR}")
            return False
        
        print_success(f"Data directory: {Config.DATA_DIR}")
        print_success(f"Embedding model: {Config.EMBEDDING_MODEL}")
        print_success(f"Chunk size: {Config.CHUNK_SIZE}")
        return True
    except Exception as e:
        print_error(f"Config test failed: {e}")
        return False


def test_cuad_loading():
    """Test CUAD document loading."""
    print_info("Testing CUAD document loading...")
    try:
        from src.ingestion.json_loader import ingest_cuad_documents
        
        documents = ingest_cuad_documents()
        
        if not documents:
            print_warning("No CUAD documents loaded (file might be empty)")
            return False
        
        print_success(f"Loaded {len(documents)} CUAD document chunks")
        print_info(f"Sample document keys: {list(documents[0].keys())}")
        print_info(f"Sample metadata: {list(documents[0]['metadata'].keys())}")
        return True
    except FileNotFoundError as e:
        print_error(f"CUAD file not found: {e}")
        print_info("Make sure CUADv1.json is in data/raw_docs/")
        return False
    except Exception as e:
        print_error(f"CUAD loading failed: {e}")
        return False


def test_pdf_loader():
    """Test PDF loader functionality."""
    print_info("Testing PDF loader...")
    try:
        from src.ingestion.pdf_loader import load_pdf, ingest_pdf_documents
        
        # Create a simple test PDF if none exists
        pdf_dir = Path("data/raw_docs/pdfs")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        
        test_pdfs = list(pdf_dir.glob("*.pdf"))
        
        if not test_pdfs:
            print_warning("No PDF files found in data/raw_docs/pdfs/")
            print_info("Skipping PDF loader test (create PDF files to test)")
            return True  # Not a failure, just no PDFs to test
        
        # Test loading first PDF
        test_pdf = test_pdfs[0]
        print_info(f"Testing with: {test_pdf.name}")
        
        doc = load_pdf(str(test_pdf))
        print_success(f"Loaded PDF: {doc['metadata']['filename']}")
        print_info(f"  Pages: {doc['metadata']['total_pages']}")
        print_info(f"  Text length: {len(doc['text'])} chars")
        
        # Test ingestion
        documents = ingest_pdf_documents(str(test_pdf))
        print_success(f"Created {len(documents)} chunks from PDF")
        
        return True
    except Exception as e:
        print_error(f"PDF loader test failed: {e}")
        return False


def test_embeddings():
    """Test embedding generation."""
    print_info("Testing embedding generation...")
    try:
        from src.embeddings.embedding_generator import EmbeddingGenerator
        from src.config import Config
        
        embedder = EmbeddingGenerator(model_name=Config.EMBEDDING_MODEL)
        
        # Test with sample text
        test_docs = [
            {"text": "This is a test document about contracts."},
            {"text": "Termination conditions require 30 days notice."}
        ]
        
        embeddings = embedder.generate_embeddings(test_docs)
        
        print_success(f"Generated embeddings: shape {embeddings.shape}")
        print_success(f"Embedding dimension: {embedder.get_embedding_dim()}")
        
        if embeddings.shape[0] != len(test_docs):
            print_error("Embedding count mismatch")
            return False
        
        return True
    except Exception as e:
        print_error(f"Embedding test failed: {e}")
        return False


def test_vector_store():
    """Test vector store operations."""
    print_info("Testing vector store...")
    try:
        import numpy as np
        from src.retrieval.vector_store import FAISSVectorStore
        from src.config import Config
        
        vector_store = FAISSVectorStore(embedding_dim=Config.EMBEDDING_DIM)
        
        # Test with sample embeddings
        test_embeddings = np.random.rand(5, Config.EMBEDDING_DIM).astype('float32')
        test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)  # Normalize
        
        test_metadata = [
            {"document_id": f"test_{i}", "text": f"Test document {i}"}
            for i in range(5)
        ]
        
        vector_store.add_embeddings(test_embeddings, test_metadata)
        
        print_success(f"Added {len(test_metadata)} vectors to index")
        
        # Test search
        query = np.random.rand(Config.EMBEDDING_DIM).astype('float32')
        query = query / np.linalg.norm(query)  # Normalize
        
        results = vector_store.search(query, k=3)
        
        print_success(f"Search returned {len(results)} results")
        
        # Test stats
        stats = vector_store.get_stats()
        print_success(f"Vector store stats: {stats['total_vectors']} vectors")
        
        return True
    except Exception as e:
        print_error(f"Vector store test failed: {e}")
        return False


def test_qa_chain():
    """Test QA chain."""
    print_info("Testing QA chain...")
    try:
        from src.llm.qa_chain import RAGQAChain
        
        qa_chain = RAGQAChain()
        
        test_docs = [
            {
                "text": "The contract allows termination with 30 days written notice.",
                "metadata": {"document_id": "test1"}
            },
            {
                "text": "Termination conditions require prior approval from management.",
                "metadata": {"document_id": "test2"}
            }
        ]
        
        answer = qa_chain.generate_answer("What are the termination conditions?", test_docs)
        
        print_success(f"Generated answer: {answer[:100]}...")
        return True
    except Exception as e:
        print_error(f"QA chain test failed: {e}")
        return False


def test_api_health(base_url="http://127.0.0.1:8000"):
    """Test API health endpoint."""
    print_info("Testing API health endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"API is running: {data.get('status', 'OK')}")
            print_info(f"Total vectors: {data.get('total_vectors', 0)}")
            return True
        else:
            print_error(f"API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_warning("API server is not running")
        print_info("Start the API with: uvicorn src.api.app:app --reload")
        return False
    except Exception as e:
        print_error(f"API health check failed: {e}")
        return False


def test_api_query(base_url="http://127.0.0.1:8000"):
    """Test API query endpoint."""
    print_info("Testing API query endpoint...")
    try:
        payload = {
            "question": "What are termination conditions?",
            "top_k": 3
        }
        
        response = requests.post(
            f"{base_url}/ask-question",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Query successful")
            print_info(f"Answer: {data['answer'][:150]}...")
            print_info(f"Found {len(data['sources'])} sources")
            return True
        else:
            print_error(f"Query failed with status {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print_warning("API server is not running")
        return False
    except Exception as e:
        print_error(f"API query test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RAG SYSTEM TEST SUITE")
    print("="*60 + "\n")
    
    results = {}
    
    # Component tests
    results["imports"] = test_imports()
    print()
    
    results["config"] = test_config()
    print()
    
    results["cuad_loading"] = test_cuad_loading()
    print()
    
    results["pdf_loader"] = test_pdf_loader()
    print()
    
    results["embeddings"] = test_embeddings()
    print()
    
    results["vector_store"] = test_vector_store()
    print()
    
    results["qa_chain"] = test_qa_chain()
    print()
    
    # API tests
    print("="*60)
    print("API TESTS")
    print("="*60 + "\n")
    
    results["api_health"] = test_api_health()
    print()
    
    if results["api_health"]:
        results["api_query"] = test_api_query()
        print()
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60 + "\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status}{Colors.END}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print_success("All tests passed! 🎉")
        return 0
    else:
        print_warning(f"{total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
