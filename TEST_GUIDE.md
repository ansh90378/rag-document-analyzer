# RAG System Testing Guide

This guide will help you test the RAG Document Analyzer system step by step.

## Prerequisites

1. **Activate Virtual Environment** (if using one):
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Test 1: Test CLI Pipeline with CUAD (JSON)

This is the simplest test - it uses the existing CUAD data.

```bash
python main.py --source json --query "What are termination conditions?" --top-k 3
```

**Expected Output:**
- Documents loading
- Embeddings generation
- Vector store initialization
- Answer with source citations

**What to check:**
- ✓ Documents load successfully
- ✓ Answer is generated
- ✓ Sources are listed with scores

---

## Test 2: Test API Server

### Step 1: Start the API Server

In one terminal:
```bash
uvicorn src.api.app:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Test Health Endpoint

In another terminal (or use a browser):
```bash
curl http://127.0.0.1:8000/
```

Or open in browser: http://127.0.0.1:8000/

**Expected Response:**
```json
{
  "status": "RAG API is running 🚀",
  "version": "2.0",
  "index_loaded": true,
  "total_vectors": 1234
}
```

### Step 3: Test Query Endpoint

```bash
curl -X POST "http://127.0.0.1:8000/ask-question" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What are termination conditions?\", \"top_k\": 3}"
```

**Or use Python:**
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/ask-question",
    json={"question": "What are termination conditions?", "top_k": 3}
)
print(response.json())
```

**Expected Response:**
```json
{
  "answer": "...",
  "sources": [
    {
      "contract_id": "...",
      "paragraph_id": 0,
      "score": 0.749,
      "source_type": "CUAD",
      "filename": null
    }
  ]
}
```

---

## Test 3: Test PDF Ingestion

### Option A: Via CLI

1. Place a PDF file in `data/raw_docs/pdfs/`:
   ```bash
   # Create directory if it doesn't exist
   mkdir -p data/raw_docs/pdfs
   
   # Copy your PDF file there
   cp your_contract.pdf data/raw_docs/pdfs/
   ```

2. Run the CLI:
   ```bash
   python main.py --source pdf --pdf-path data/raw_docs/pdfs/your_contract.pdf --query "What is this document about?"
   ```

### Option B: Via API

1. Make sure API server is running (see Test 2)

2. Upload a PDF:
   ```bash
   curl -X POST "http://127.0.0.1:8000/upload-pdf" \
     -F "file=@path/to/your/contract.pdf"
   ```

   **Or use Python:**
   ```python
   import requests
   
   with open("path/to/contract.pdf", "rb") as f:
       response = requests.post(
           "http://127.0.0.1:8000/upload-pdf",
           files={"file": f}
       )
   print(response.json())
   ```

3. Query after upload:
   ```bash
   curl -X POST "http://127.0.0.1:8000/ask-question" \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"What does the uploaded contract say?\", \"top_k\": 3}"
   ```

---

## Test 4: Check System Statistics

### Via API:
```bash
curl http://127.0.0.1:8000/stats
```

**Expected Response:**
```json
{
  "total_vectors": 1234,
  "embedding_dim": 384,
  "metadata_count": 1234,
  "index_type": "IndexFlatIP"
}
```

### List Uploaded Documents:
```bash
curl http://127.0.0.1:8000/documents
```

---

## Test 5: Comprehensive Component Test

Run the automated test suite:

```bash
python test_rag_system.py
```

This will test:
- ✓ Module imports
- ✓ Configuration
- ✓ CUAD loading
- ✓ PDF loader
- ✓ Embeddings generation
- ✓ Vector store
- ✓ QA chain
- ✓ API endpoints

---

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:**
1. Make sure virtual environment is activated
2. Install dependencies: `pip install -r requirements.txt`
3. Check Python path: `python --version`

### Issue: CUAD file not found

**Solution:**
- Make sure `data/raw_docs/CUADv1.json` exists
- Check file path in `src/ingestion/json_loader.py`

### Issue: API server won't start

**Solution:**
1. Check if port 8000 is already in use
2. Try different port: `uvicorn src.api.app:app --port 8001`
3. Check for import errors in `src/api/app.py`

### Issue: PDF upload fails

**Solution:**
1. Check file size (max 50MB by default)
2. Verify PDF is not encrypted/password-protected
3. Check `data/raw_docs/uploads/` directory exists
4. Check API logs for detailed error messages

### Issue: No answers generated

**Solution:**
1. Verify documents are indexed: Check `/stats` endpoint
2. Make sure embeddings were generated
3. Check if query is relevant to indexed documents
4. Try different queries

---

## Quick Test Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] CUAD data exists (`data/raw_docs/CUADv1.json`)
- [ ] CLI test with CUAD works
- [ ] API server starts successfully
- [ ] Health endpoint responds
- [ ] Query endpoint returns answers
- [ ] PDF can be uploaded (if testing PDF feature)
- [ ] Statistics endpoint works

---

## Next Steps

Once basic tests pass:
1. Try different types of questions
2. Upload multiple PDFs
3. Test batch PDF upload
4. Compare results between CUAD and PDF sources
5. Experiment with different `top_k` values
