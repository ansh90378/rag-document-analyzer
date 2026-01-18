#!/bin/bash
# Test CLI pipeline with different sources

echo "========================================="
echo "Testing RAG CLI Pipeline"
echo "========================================="
echo ""

echo "Test 1: CUAD JSON only"
echo "----------------------------------------"
python main.py --source json --query "What are termination conditions?" --top-k 3
echo ""

echo "Test 2: PDF (if available)"
echo "----------------------------------------"
if [ -d "data/raw_docs/pdfs" ] && [ -n "$(ls -A data/raw_docs/pdfs/*.pdf 2>/dev/null)" ]; then
    PDF_FILE=$(ls data/raw_docs/pdfs/*.pdf | head -1)
    echo "Using PDF: $PDF_FILE"
    python main.py --source pdf --pdf-path "$PDF_FILE" --query "What is this document about?" --top-k 3
else
    echo "No PDF files found in data/raw_docs/pdfs/"
    echo "Skipping PDF test"
fi
echo ""

echo "Test 3: Both JSON and PDF"
echo "----------------------------------------"
if [ -d "data/raw_docs/pdfs" ] && [ -n "$(ls -A data/raw_docs/pdfs/*.pdf 2>/dev/null)" ]; then
    PDF_DIR="data/raw_docs/pdfs"
    python main.py --source both --pdf-path "$PDF_DIR" --query "What are the contract terms?" --top-k 5
else
    echo "No PDF files found. Running JSON only..."
    python main.py --source json --query "What are the contract terms?" --top-k 5
fi
