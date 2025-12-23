from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import List, Dict


class RAGQAChain:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_tokens: int = 256
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_tokens
        )

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

        response = self.generator(prompt)[0]["generated_text"]
        return response.strip()
