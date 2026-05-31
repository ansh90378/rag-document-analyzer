from unittest.mock import patch

from src.embeddings.embedding_generator import EmbeddingGenerator


@patch("src.embeddings.embedding_generator.SentenceTransformer")
def test_embedding_dimension_uses_current_sentence_transformers_api(model_loader):
    model = model_loader.return_value
    model.get_embedding_dimension.return_value = 384

    embedder = EmbeddingGenerator(model_name="test-model")

    model_loader.assert_called_once_with("test-model")
    model.get_embedding_dimension.assert_called_once_with()
    assert embedder.get_embedding_dim() == 384
