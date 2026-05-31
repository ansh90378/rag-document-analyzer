from typing import Dict, List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class RAGQAChain:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_tokens: int = 256
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def build_prompt(self, query: str, contexts: List[str]) -> str:
        context_block = "\n\n".join(contexts)

        prompt = f"""
            Answer the question using ONLY the context below.
            If the answer is not present, say exactly: Not found in document.

            Context:
            {context_block}

            Question:
            {query}

            Answer:
            """
        return prompt.strip()

    def generate_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        contexts = [doc["text"] for doc in retrieved_docs]
        prompt = self.build_prompt(query, contexts)
        model_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        model_inputs = {
            name: tensor.to(self.model.device)
            for name, tensor in model_inputs.items()
        }
        output_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens
        )
        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        ).strip()
