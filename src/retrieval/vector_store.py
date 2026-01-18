import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        """
        FAISS index using cosine similarity
        
        Args:
            embedding_dim: Dimension of embeddings (e.g., 384 for all-MiniLM-L6-v2)
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata = []
        self.embeddings_cache = None  # Cache embeddings for saving
        self.index_version = None

    def load_embeddings(
        self,
        embeddings_path: str,
        metadata_path: str
    ):
        """
        Load embeddings and metadata from disk
        
        Args:
            embeddings_path: Path to .npy file with embeddings
            metadata_path: Path to .json file with metadata
        """
        embeddings = np.load(embeddings_path)

        # Validate embedding dimensions
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if embeddings.shape[0] != len(self.metadata):
            raise ValueError(
                f"Embedding count mismatch: {embeddings.shape[0]} embeddings "
                f"vs {len(self.metadata)} metadata entries"
            )

        # Store embeddings in cache for later saving
        self.embeddings_cache = embeddings.copy()
        
        self.index.add(embeddings)
        self.index_version = datetime.now().isoformat()

        print(f"Loaded {embeddings.shape[0]} vectors into FAISS index")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ):
        """
        Add new embeddings and metadata to the existing index incrementally.
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            metadata: List of metadata dictionaries, one per embedding
            
        Raises:
            ValueError: If dimensions don't match or counts don't match
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"Count mismatch: {embeddings.shape[0]} embeddings vs "
                f"{len(metadata)} metadata entries"
            )
        
        # Update embeddings cache
        if self.embeddings_cache is None:
            self.embeddings_cache = embeddings.copy()
        else:
            self.embeddings_cache = np.vstack([self.embeddings_cache, embeddings])
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Append metadata
        self.metadata.extend(metadata)
        
        # Update version timestamp
        self.index_version = datetime.now().isoformat()
        
        print(f"Added {embeddings.shape[0]} vectors to FAISS index. "
              f"Total vectors: {self.index.ntotal}")

    def save_index(
        self,
        embeddings_path: str = "data/embeddings/unified_embeddings.npy",
        metadata_path: str = "data/embeddings/unified_metadata.json"
    ):
        """
        Save current index embeddings and metadata to disk.
        
        Args:
            embeddings_path: Path to save embeddings .npy file
            metadata_path: Path to save metadata .json file
        """
        if self.index.ntotal == 0:
            print("Warning: No embeddings to save. Index is empty.")
            return
        
        if self.embeddings_cache is None:
            print("Warning: No embeddings cache available. Cannot save embeddings.")
            return
        
        output_path = Path(embeddings_path).parent
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Validate counts match
        if self.embeddings_cache.shape[0] != len(self.metadata):
            raise ValueError(
                f"Mismatch: {self.embeddings_cache.shape[0]} embeddings vs "
                f"{len(self.metadata)} metadata entries"
            )
        
        # Save embeddings
        np.save(embeddings_path, self.embeddings_cache)
        
        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Saved {self.embeddings_cache.shape[0]} embeddings to {embeddings_path}")
        print(f"Saved {len(self.metadata)} metadata entries to {metadata_path}")

    def get_stats(self) -> Dict:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "metadata_count": len(self.metadata),
            "index_version": self.index_version,
            "index_type": type(self.index).__name__
        }

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 4
    ) -> List[Dict]:
        """
        Perform top-k semantic search
        
        Args:
            query_embedding: Query embedding vector (1D array)
            k: Number of top results to return
            
        Returns:
            List of dictionaries with metadata and similarity scores
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure k doesn't exceed total vectors
        k = min(k, self.index.ntotal)
        
        query_embedding = np.expand_dims(query_embedding, axis=0)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(score)
                results.append(result)

        return results