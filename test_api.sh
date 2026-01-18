#!/bin/bash
# Test API endpoints

API_URL="http://127.0.0.1:8000"

echo "========================================="
echo "Testing RAG API Endpoints"
echo "========================================="
echo ""

echo "Test 1: Health Check"
echo "----------------------------------------"
curl -X GET "$API_URL/" -H "Content-Type: application/json"
echo ""
echo ""

echo "Test 2: Get Statistics"
echo "----------------------------------------"
curl -X GET "$API_URL/stats" -H "Content-Type: application/json"
echo ""
echo ""

echo "Test 3: List Documents"
echo "----------------------------------------"
curl -X GET "$API_URL/documents" -H "Content-Type: application/json"
echo ""
echo ""

echo "Test 4: Ask Question"
echo "----------------------------------------"
curl -X POST "$API_URL/ask-question" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are termination conditions?",
    "top_k": 3
  }'
echo ""
echo ""

echo "Test 5: Upload PDF (if file exists)"
echo "----------------------------------------"
if [ -f "test_contract.pdf" ]; then
    curl -X POST "$API_URL/upload-pdf" \
      -F "file=@test_contract.pdf"
else
    echo "test_contract.pdf not found. Skipping upload test."
    echo "To test upload, create a PDF file or use an existing one."
fi
echo ""
