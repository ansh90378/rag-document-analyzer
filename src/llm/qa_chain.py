from typing import Dict, List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class RAGQAChain:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_tokens: int = 256,
        max_input_tokens: int = 512,
        prefer_local_files: bool = True,
    ):
        self.tokenizer, self.model = self._load_model(model_name, prefer_local_files)
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens

    @staticmethod
    def _load_model(model_name: str, prefer_local_files: bool):
        """Load cached model files before contacting Hugging Face when possible."""
        if prefer_local_files:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=True,
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    local_files_only=True,
                )
                return tokenizer, model
            except Exception:
                # Allow the initial run to download files when they are not cached.
                pass

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            return tokenizer, model
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load answer model '{model_name}'. "
                "Check your internet connection for the first download, or ensure "
                "the model is already present in the local Hugging Face cache."
            ) from exc

    def build_prompt(self, query: str, contexts: List[str]) -> str:
        prefix = (
            "Answer the question using ONLY the context below.\n"
            "If the answer is not present, say exactly: Not found in document.\n\n"
            f"Question:\n{query}\n\n"
            "Context:\n"
        )
        suffix = "\n\nAnswer:"
        scaffold_tokens = self.tokenizer.encode(
            prefix + suffix,
            add_special_tokens=True,
        )
        remaining_tokens = max(self.max_input_tokens - len(scaffold_tokens), 0)
        context_excerpts = []

        for index, context in enumerate(contexts):
            if remaining_tokens == 0:
                break

            separator = "\n\n" if context_excerpts else ""
            label = f"[Source {index + 1}]\n"
            header_tokens = self.tokenizer.encode(
                separator + label,
                add_special_tokens=False,
            )
            if len(header_tokens) >= remaining_tokens:
                break

            # Reserve a fair share of the remaining input window for each source
            # so one long chunk cannot crowd all lower-ranked retrieved chunks out.
            sources_left = len(contexts) - index
            excerpt_budget = max(remaining_tokens // sources_left, 1)
            excerpt_budget = max(excerpt_budget - len(header_tokens), 0)
            context_tokens = self.tokenizer.encode(
                context,
                add_special_tokens=False,
            )[:excerpt_budget]
            if not context_tokens:
                continue


        for index, context in enumerate(contexts):
            if remaining_tokens == 0:
                break

            separator = "\n\n" if context_excerpts else ""
            label = f"[Source {index + 1}]\n"
            header_tokens = self.tokenizer.encode(
                separator + label,
                add_special_tokens=False,
            )
            if len(header_tokens) >= remaining_tokens:
                break

            # Reserve a fair share of the remaining input window for each source
            # so one long chunk cannot crowd all lower-ranked retrieved chunks out.
            sources_left = len(contexts) - index
            excerpt_budget = max(remaining_tokens // sources_left, 1)
            excerpt_budget = max(excerpt_budget - len(header_tokens), 0)
            context_tokens = self.tokenizer.encode(
                context,
                add_special_tokens=False,
            )[:excerpt_budget]
            if not context_tokens:
                continue

            excerpt = self.tokenizer.decode(
                context_tokens,
                skip_special_tokens=True,
            ).strip()
            if not excerpt:
                continue


        for index, context in enumerate(contexts):
            if remaining_tokens == 0:
                break

            separator = "\n\n" if context_excerpts else ""
            label = f"[Source {index + 1}]\n"
            header_tokens = self.tokenizer.encode(
                separator + label,
                add_special_tokens=False,
            )
            if len(header_tokens) >= remaining_tokens:
                break

            # Reserve a fair share of the remaining input window for each source
            # so one long chunk cannot crowd all lower-ranked retrieved chunks out.
            sources_left = len(contexts) - index
            excerpt_budget = max(remaining_tokens // sources_left, 1)
            excerpt_budget = max(excerpt_budget - len(header_tokens), 0)
            context_tokens = self.tokenizer.encode(
                context,
                add_special_tokens=False,
            )[:excerpt_budget]
            if not context_tokens:
                continue

            excerpt = self.tokenizer.decode(
                context_tokens,
                skip_special_tokens=True,
            ).strip()
            if not excerpt:
                continue

            context_excerpts.append(separator + label + excerpt)
            remaining_tokens -= len(header_tokens) + len(context_tokens)

        return prefix + "".join(context_excerpts) + suffix

    def generate_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        if not retrieved_docs:
            return "Not found in document."

        contexts = [doc["text"] for doc in retrieved_docs]
        prompt = self.build_prompt(query, contexts)
        model_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
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
