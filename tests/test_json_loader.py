import pytest

from src.ingestion.json_loader import chunk_text


def test_chunk_text_avoids_overlap_only_tail_chunk():
    text = "a" * 800

    assert chunk_text(text, chunk_size=800, overlap=200) == [text]


def test_chunk_text_prefers_natural_boundary_and_preserves_overlap():
    text = "First clause ends here. Second clause has more contract language. Third clause remains."

    chunks = chunk_text(text, chunk_size=45, overlap=10)

    assert chunks[0] == "First clause ends here."
    assert chunks[1].startswith("ends here.")
    assert all(len(chunk) <= 45 for chunk in chunks)
    assert "".join(chunks).count("Third clause remains.") == 1


def test_chunk_text_rejects_invalid_window_settings():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("text", chunk_size=0)

    with pytest.raises(ValueError, match="overlap"):
        chunk_text("text", chunk_size=10, overlap=10)
