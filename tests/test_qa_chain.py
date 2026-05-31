from unittest.mock import MagicMock, call, patch

from src.llm.qa_chain import RAGQAChain


def _tokens(text, **kwargs):
    return text.split()


@patch("src.llm.qa_chain.AutoModelForSeq2SeqLM.from_pretrained")
@patch("src.llm.qa_chain.AutoTokenizer.from_pretrained")
def test_generate_answer_uses_seq2seq_generate(tokenizer_loader, model_loader):
    tokenizer = tokenizer_loader.return_value
    model = model_loader.return_value
    input_ids = MagicMock()
    attention_mask = MagicMock()
    generated_ids = MagicMock()

    model.device = "cpu"
    tokenizer.encode.side_effect = _tokens
    tokenizer.decode.side_effect = lambda tokens, **kwargs: (
        " ".join(tokens) if isinstance(tokens, list) else "  The notice period is 30 days.  "
    )
    tokenizer.return_value = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    model.generate.return_value = [generated_ids]

    qa_chain = RAGQAChain(model_name="test-model", max_tokens=42)
    answer = qa_chain.generate_answer(
        "What is the notice period?",
        [{"text": "The notice period is 30 days."}],
    )

    tokenizer_loader.assert_called_once_with("test-model", local_files_only=True)
    model_loader.assert_called_once_with("test-model", local_files_only=True)
    prompt = tokenizer.call_args.args[0]
    assert "What is the notice period?" in prompt
    assert "The notice period is 30 days." in prompt
    tokenizer.assert_called_once_with(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids.to.assert_called_once_with("cpu")
    attention_mask.to.assert_called_once_with("cpu")
    model.generate.assert_called_once_with(
        input_ids=input_ids.to.return_value,
        attention_mask=attention_mask.to.return_value,
        max_new_tokens=42,
    )
    assert answer == "The notice period is 30 days."


@patch("src.llm.qa_chain.AutoModelForSeq2SeqLM.from_pretrained")
@patch("src.llm.qa_chain.AutoTokenizer.from_pretrained")
def test_build_prompt_preserves_question_and_fits_multiple_long_contexts(
    tokenizer_loader,
    model_loader,
):
    tokenizer = tokenizer_loader.return_value
    tokenizer.encode.side_effect = _tokens
    tokenizer.decode.side_effect = lambda tokens, **kwargs: " ".join(tokens)
    qa_chain = RAGQAChain(model_name="test-model", max_input_tokens=45)

    prompt = qa_chain.build_prompt(
        "What are termination conditions?",
        ["first " * 100, "second " * 100, "third " * 100],
    )

    assert prompt.index("Question:") < prompt.index("Context:")
    assert "What are termination conditions?" in prompt
    assert "[Source 1]" in prompt
    assert "[Source 2]" in prompt
    assert "[Source 3]" in prompt
    assert prompt.endswith("Answer:")
    assert len(_tokens(prompt)) <= 45


@patch("src.llm.qa_chain.AutoModelForSeq2SeqLM.from_pretrained")
@patch("src.llm.qa_chain.AutoTokenizer.from_pretrained")
def test_generate_answer_returns_not_found_without_retrieved_documents(
    tokenizer_loader,
    model_loader,
):
    qa_chain = RAGQAChain(model_name="test-model")

    assert qa_chain.generate_answer("What is the notice period?", []) == "Not found in document."
    qa_chain.model.generate.assert_not_called()


@patch("src.llm.qa_chain.AutoModelForSeq2SeqLM.from_pretrained")
@patch("src.llm.qa_chain.AutoTokenizer.from_pretrained")
def test_answer_model_downloads_when_not_cached(tokenizer_loader, model_loader):
    tokenizer = MagicMock()
    model = MagicMock()
    tokenizer_loader.side_effect = [OSError("not cached"), tokenizer]
    model_loader.return_value = model

    qa_chain = RAGQAChain(model_name="test-model")

    assert tokenizer_loader.call_args_list == [
        call("test-model", local_files_only=True),
        call("test-model"),
    ]
    model_loader.assert_called_once_with("test-model")
    assert qa_chain.tokenizer is tokenizer
    assert qa_chain.model is model
