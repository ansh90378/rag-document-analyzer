# Quick Test Guide for RAG System

## Step 1: Setup (One-time)

### Windows:
```batch
setup_and_test.bat
```

### Linux/Mac:
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Test CLI (Simplest Test)

This test uses existing CUAD data and doesn't require API:

```bash
python main.py --source json --query "What are termination conditions?" --top-k 3
```

**What to expect:**
- Documents loading (should show ~44,933 CUAD chunks)
- Embeddings generation (takes a few minutes first time)
- Answer with sources

## Step 3: Test API (If CLI works)

### Terminal 1: Start API Server
```bash
# Windows
venv\Scripts\activate
uvicorn src.api.app:app --reload

# Linux/Mac
source venv/bin/activate
uvicorn src.api.app:app --reload
```

### Terminal 2: Test API

**Health Check:**
```bash
curl http://127.0.0.1:8000/
```

**Ask a Question:**
```bash
curl -X POST "http://127.0.0.1:8000/ask-question" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What are termination conditions?\", \"top_k\": 3}"
```

**Or use Python:**
```python
import requests

# Health check
r = requests.get("http://127.0.0.1:8000/")
print(r.json())

# Ask question
r = requests.post(
    "http://127.0.0.1:8000/ask-question",
    json={"question": "What are termination conditions?", "top_k": 3}
)
print(r.json())
```

## Step 4: Test PDF Upload (Optional)

1. Place a PDF in `data/raw_docs/pdfs/`
2. Upload via API:
```bash
curl -X POST "http://127.0.0.1:8000/upload-pdf" -F "file=@data/raw_docs/pdfs/your_file.pdf"
```

Or via CLI:
```bash
python main.py --source pdf --pdf-path data/raw_docs/pdfs/your_file.pdf --query "What is this about?"
```

## Common Issues

### "Module not found"
- **Fix:** Activate venv and run `pip install -r requirements.txt`

### "CUAD file not found"
- **Fix:** Check `data/raw_docs/CUADv1.json` exists

### API won't start
- **Fix:** Check port 8000 is free, or use `--port 8001`

### Hugging Face download or DNS error
- **Fix:** The first run needs internet access to download the embedding and answer models. Later runs load cached model files first and reuse saved JSON embeddings. If source documents changed, rerun with `--rebuild-embeddings` after restoring internet access.

### No answer generated
- **Fix:** Check embeddings exist in `data/embeddings/`, may need to run CLI first
