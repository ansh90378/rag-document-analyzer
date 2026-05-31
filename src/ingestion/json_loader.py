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
    """Split text quickly at natural boundaries while retaining useful overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)
    minimum_boundary = max(chunk_size // 2, overlap + 1)
    separators = ("\n\n", "\n", ". ", "; ", ", ", " ")

    while start < length:
        hard_end = min(start + chunk_size, length)
        end = hard_end

        if hard_end < length:
            boundary_floor = min(start + minimum_boundary, hard_end)
            for separator in separators:
                boundary = text.rfind(separator, boundary_floor, hard_end)
                if boundary != -1:
                    end = boundary + len(separator)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        # Keep context between chunks without emitting an overlap-only tail chunk.
        # Rewind to a word boundary so the overlap remains readable.
        next_start = max(end - overlap, start + 1)
        whitespace = text.rfind(" ", start + 1, next_start + 1)
        if whitespace != -1:
            next_start = whitespace + 1
        start = next_start

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
