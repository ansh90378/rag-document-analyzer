import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Support direct Python and pytest invocations launched from the tests directory.
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from unittest.mock import MagicMock, patch

import numpy as np

from main import run_rag_pipeline


def _configure_pipeline_mocks(embedder_loader, vector_store_loader, qa_chain_loader):
    embedder = embedder_loader.return_value
    embedder.model.encode.return_value = np.array([1.0, 0.0])

    vector_store = vector_store_loader.return_value
    vector_store.index.ntotal = 1
    vector_store.search.return_value = [{"text": "Termination requires notice."}]

    qa_chain = qa_chain_loader.return_value
    qa_chain.generate_answer.return_value = "Termination requires notice."

    return embedder, vector_store


@patch("main.RAGQAChain")
@patch("main.FAISSVectorStore")
@patch("main.EmbeddingGenerator")
@patch("main.ingest_cuad_documents")
def test_json_query_reuses_saved_embeddings(
    ingest_cuad_documents,
    embedder_loader,
    vector_store_loader,
    qa_chain_loader,
    tmp_path,
):
    embeddings_path = tmp_path / "unified_embeddings.npy"
    metadata_path = tmp_path / "unified_metadata.json"
    embeddings_path.touch()
    metadata_path.touch()
    embedder, vector_store = _configure_pipeline_mocks(
        embedder_loader,
        vector_store_loader,
        qa_chain_loader,
    )

    with (
        patch("main.Config.get_embeddings_path", return_value=embeddings_path),
        patch("main.Config.get_metadata_path", return_value=metadata_path),
    ):
        answer, sources = run_rag_pipeline("What are termination conditions?")

    ingest_cuad_documents.assert_not_called()
    embedder.generate_embeddings.assert_not_called()
    embedder.save_embeddings.assert_not_called()
    vector_store.load_embeddings.assert_called_once_with(
        embeddings_path=str(embeddings_path),
        metadata_path=str(metadata_path),
    )
    assert answer == "Termination requires notice."
    assert sources == [{"text": "Termination requires notice."}]


@patch("main.RAGQAChain")
@patch("main.FAISSVectorStore")
@patch("main.EmbeddingGenerator")
@patch("main.ingest_cuad_documents")
def test_rebuild_embeddings_refreshes_saved_json_index(
    ingest_cuad_documents,
    embedder_loader,
    vector_store_loader,
    qa_chain_loader,
    tmp_path,
):
    embeddings_path = tmp_path / "unified_embeddings.npy"
    metadata_path = tmp_path / "unified_metadata.json"
    embeddings_path.touch()
    metadata_path.touch()
    embedder, _ = _configure_pipeline_mocks(
        embedder_loader,
        vector_store_loader,
        qa_chain_loader,
    )
    documents = [{"text": "Termination requires notice.", "metadata": {"source": "test"}}]
    ingest_cuad_documents.return_value = documents
    embedder.generate_embeddings.return_value = np.array([[1.0, 0.0]])

    with (
        patch("main.Config.get_embeddings_path", return_value=embeddings_path),
        patch("main.Config.get_metadata_path", return_value=metadata_path),
    ):
        run_rag_pipeline("What are termination conditions?", rebuild_embeddings=True)

    ingest_cuad_documents.assert_called_once_with()
    embedder.generate_embeddings.assert_called_once_with(documents)
    embedder.save_embeddings.assert_called_once()
