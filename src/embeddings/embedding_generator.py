from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
from typing import List, Dict


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, documents: List[Dict]) -> np.ndarray:
        """
        Convert text chunks to normalized embeddings
        """
        texts = [doc["text"] for doc in documents]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        output_dir: str = "data/embeddings"
    ):
        """
        Persist embeddings + metadata to disk
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        np.save(output_path / "cuad_embeddings.npy", embeddings)

        with open(output_path / "cuad_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved embeddings to {output_path}")
