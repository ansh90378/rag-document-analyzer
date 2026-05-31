from unittest.mock import MagicMock, call, patch

import pytest

from src.embeddings.embedding_generator import EmbeddingGenerator


@patch("src.embeddings.embedding_generator.SentenceTransformer")
def test_embedding_model_uses_local_cache_first(model_loader):
    model = model_loader.return_value
    model.get_embedding_dimension.return_value = 384

    embedder = EmbeddingGenerator(model_name="test-model")

    model_loader.assert_called_once_with("test-model", local_files_only=True)
    model.get_embedding_dimension.assert_called_once_with()
    assert embedder.get_embedding_dim() == 384


@patch("src.embeddings.embedding_generator.SentenceTransformer")
def test_embedding_model_downloads_when_not_cached(model_loader):
    model = MagicMock()
    model.get_embedding_dimension.return_value = 384
    model_loader.side_effect = [OSError("not cached"), model]

    embedder = EmbeddingGenerator(model_name="test-model")

    assert model_loader.call_args_list == [
        call("test-model", local_files_only=True),
        call("test-model"),
    ]
    assert embedder.get_embedding_dim() == 384


@patch("src.embeddings.embedding_generator.SentenceTransformer")
def test_embedding_model_reports_actionable_load_error(model_loader):
    model_loader.side_effect = [OSError("not cached"), RuntimeError("network unavailable")]

    with pytest.raises(RuntimeError, match="Check your internet connection"):
        EmbeddingGenerator(model_name="test-model")
