from unittest.mock import MagicMock, patch

from src.llm.qa_chain import RAGQAChain


@patch("src.llm.qa_chain.AutoModelForSeq2SeqLM.from_pretrained")
@patch("src.llm.qa_chain.AutoTokenizer.from_pretrained")
def test_generate_answer_uses_seq2seq_generate(tokenizer_loader, model_loader):
    tokenizer = tokenizer_loader.return_value
    model = model_loader.return_value
    input_ids = MagicMock()
    attention_mask = MagicMock()
    generated_ids = MagicMock()

    model.device = "cpu"
    tokenizer.return_value = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    model.generate.return_value = [generated_ids]
    tokenizer.decode.return_value = "  The notice period is 30 days.  "

    qa_chain = RAGQAChain(model_name="test-model", max_tokens=42)
    answer = qa_chain.generate_answer(
        "What is the notice period?",
        [{"text": "The notice period is 30 days."}],
    )

    tokenizer_loader.assert_called_once_with("test-model")
    model_loader.assert_called_once_with("test-model")
    tokenizer.assert_called_once()
    prompt = tokenizer.call_args.args[0]
    assert "What is the notice period?" in prompt
    assert "The notice period is 30 days." in prompt
    tokenizer.assert_called_once_with(prompt, return_tensors="pt", truncation=True)
    input_ids.to.assert_called_once_with("cpu")
    attention_mask.to.assert_called_once_with("cpu")
    model.generate.assert_called_once_with(
        input_ids=input_ids.to.return_value,
        attention_mask=attention_mask.to.return_value,
        max_new_tokens=42,
    )
    tokenizer.decode.assert_called_once_with(
        generated_ids,
        skip_special_tokens=True,
    )
    assert answer == "The notice period is 30 days."
