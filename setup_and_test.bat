@echo off
REM Setup and Test Script for RAG System (Windows)

echo ================================================
echo RAG System Setup and Test
echo ================================================
echo.

REM Check if venv exists
if exist "venv\Scripts\activate.bat" (
    echo [1/4] Activating virtual environment...
    call venv\Scripts\activate.bat
    echo.
) else (
    echo [!] Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo.
)

echo [2/4] Installing/Updating dependencies...
pip install -r requirements.txt
echo.

echo [3/4] Testing imports...
python -c "from src.ingestion.json_loader import ingest_cuad_documents; print('[+] CUAD loader: OK')" 2>nul
if errorlevel 1 (
    echo [-] CUAD loader: FAILED - Check dependencies
)

python -c "from src.embeddings.embedding_generator import EmbeddingGenerator; print('[+] Embedding generator: OK')" 2>nul
if errorlevel 1 (
    echo [-] Embedding generator: FAILED - Check dependencies
)

python -c "from src.retrieval.vector_store import FAISSVectorStore; print('[+] Vector store: OK')" 2>nul
if errorlevel 1 (
    echo [-] Vector store: FAILED - Check dependencies
)
echo.

echo [4/4] Quick test: CUAD JSON processing...
echo.
python main.py --source json --query "What are termination conditions?" --top-k 3
echo.

echo ================================================
echo Setup complete! 
echo.
echo Next steps:
echo   1. Start API: uvicorn src.api.app:app --reload
echo   2. Test API: curl http://127.0.0.1:8000/
echo   3. Run full test: python test_rag_system.py
echo ================================================
