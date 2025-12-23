import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict


class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        """
        FAISS index using cosine similarity
        """
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata = []

    def load_embeddings(
        self,
        embeddings_path: str,
        metadata_path: str
    ):
        """
        Load embeddings and metadata from disk
        """
        embeddings = np.load(embeddings_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.index.add(embeddings)

        print(f"Loaded {embeddings.shape[0]} vectors into FAISS index")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 4
    ) -> List[Dict]:
        """
        Perform top-k semantic search
        """
        query_embedding = np.expand_dims(query_embedding, axis=0)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            result = self.metadata[idx].copy()
            result["score"] = float(score)
            results.append(result)

        return results