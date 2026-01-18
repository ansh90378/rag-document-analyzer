from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, documents: List[Dict]) -> np.ndarray:
        """
        Convert text chunks to normalized embeddings
        
        Args:
            documents: List of document dictionaries with 'text' key
            
        Returns:
            numpy array of shape (n_documents, embedding_dim)
        """
        texts = [doc["text"] for doc in documents]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def generate_embeddings_incremental(
        self,
        documents: List[Dict],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for new documents (incremental).
        Same as generate_embeddings but with configurable progress bar.
        
        Args:
            documents: List of document dictionaries with 'text' key
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (n_documents, embedding_dim)
        """
        texts = [doc["text"] for doc in documents]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        output_dir: str = "data/embeddings",
        filename_prefix: str = "cuad",
        append: bool = False
    ):
        """
        Persist embeddings + metadata to disk
        
        Args:
            embeddings: numpy array of embeddings
            metadata: List of metadata dictionaries
            output_dir: Directory to save files
            filename_prefix: Prefix for output files (default: "cuad")
            append: If True, append to existing files; if False, overwrite
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        embeddings_file = output_path / f"{filename_prefix}_embeddings.npy"
        metadata_file = output_path / f"{filename_prefix}_metadata.json"

        if append and embeddings_file.exists() and metadata_file.exists():
            # Load existing data
            existing_embeddings = np.load(embeddings_file)
            with open(metadata_file, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)
            
            # Validate dimensions
            if existing_embeddings.shape[1] != embeddings.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch: existing {existing_embeddings.shape[1]}, "
                    f"new {embeddings.shape[1]}"
                )
            
            # Append new data
            combined_embeddings = np.vstack([existing_embeddings, embeddings])
            combined_metadata = existing_metadata + metadata
            
            np.save(embeddings_file, combined_embeddings)
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(combined_metadata, f, indent=2)
            
            print(f"Appended {embeddings.shape[0]} embeddings to existing file. "
                  f"Total: {combined_embeddings.shape[0]}")
        else:
            # Save new data (overwrite)
            np.save(embeddings_file, embeddings)
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved {embeddings.shape[0]} embeddings to {output_path}")

    def load_embeddings(
        self,
        embeddings_path: str,
        metadata_path: Optional[str] = None
    ) -> tuple:
        """
        Load embeddings and optionally metadata from disk.
        
        Args:
            embeddings_path: Path to .npy file with embeddings
            metadata_path: Optional path to .json file with metadata
            
        Returns:
            Tuple of (embeddings: np.ndarray, metadata: List[Dict] or None)
        """
        embeddings = np.load(embeddings_path)
        
        metadata = None
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        
        return embeddings, metadata

    def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension of the model.
        
        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        return self.embedding_dim
