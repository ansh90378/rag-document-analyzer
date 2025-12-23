import json
from pathlib import Path
from typing import List, Dict


# -------------------------------
# Load CUAD JSON (robust path)
# -------------------------------
def load_cuad_json() -> dict:
    base_dir = Path(__file__).resolve().parents[2]
    json_path = base_dir / "data" / "raw_docs" / "CUADv1.json"

    if not json_path.exists():
        raise FileNotFoundError(f"CUAD file not found at {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------
# Smart chunking function
# -------------------------------
def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 200
) -> List[str]:
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# -------------------------------
# Parse + chunk CUAD documents
# -------------------------------
def ingest_cuad_documents() -> List[Dict]:
    cuad_json = load_cuad_json()
    documents = []

    for contract_id, doc in enumerate(cuad_json.get("data", [])):
        contract_title = doc.get("title", f"Contract-{contract_id}")

        for para_id, para in enumerate(doc.get("paragraphs", [])):
            context = para.get("context", "").strip()
            if not context:
                continue

            chunks = chunk_text(context)

            for chunk_id, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "source": "CUAD",
                        "contract_id": contract_id,
                        "paragraph_id": para_id,
                        "chunk_id": chunk_id,
                        "contract_title": contract_title
                    }
                })

    return documents
