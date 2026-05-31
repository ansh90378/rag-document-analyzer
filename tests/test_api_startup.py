from unittest.mock import MagicMock, patch

from src.api import app


def test_cuad_preload_skips_reembedding_when_cuad_is_already_indexed():
    vector_store = MagicMock()
    vector_store.metadata = [{"source": "CUAD", "text": "Existing clause"}]
    embedder = MagicMock()

    with (
        patch.object(app.Config, "ENABLE_CUAD_PRELOAD", True),
        patch.object(app, "vector_store", vector_store),
        patch.object(app, "embedder", embedder),
        patch.object(app, "ingest_cuad_documents") as ingest_documents,
    ):
        app.load_cuad_if_enabled()

    ingest_documents.assert_not_called()
    embedder.generate_embeddings.assert_not_called()
    vector_store.add_embeddings.assert_not_called()


def test_cuad_preload_still_runs_when_loaded_index_contains_only_pdf_chunks():
    vector_store = MagicMock()
    vector_store.metadata = [{"source": "PDF", "text": "Uploaded document"}]
    vector_store.index.ntotal = 1
    embedder = MagicMock()
    embedder.generate_embeddings.return_value = [[0.1, 0.2]]
    documents = [{"text": "CUAD clause", "metadata": {"source": "CUAD"}}]

    with (
        patch.object(app.Config, "ENABLE_CUAD_PRELOAD", True),
        patch.object(app, "vector_store", vector_store),
        patch.object(app, "embedder", embedder),
        patch.object(app, "ingest_cuad_documents", return_value=documents),
    ):
        app.load_cuad_if_enabled()

    embedder.generate_embeddings.assert_called_once_with(documents)
    vector_store.add_embeddings.assert_called_once_with(
        [[0.1, 0.2]],
        [{"source": "CUAD", "text": "CUAD clause"}],
    )
