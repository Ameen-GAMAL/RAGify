import os
from openai import OpenAI


class RAGGenerator:
    def __init__(self):
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN environment variable is not set.")

        # Hugging Face OpenAI-compatible endpoint
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
        )

        # Pick a chat/instruct model that HF router supports (example from HF docs style)
        # You can swap this later if you want.
        self.model = "moonshotai/Kimi-K2-Instruct-0905"

    def generate_answer(self, query: str, retrieved_chunks: list) -> str:
        context = "\n\n".join(f"- {c['text']}" for c in retrieved_chunks)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an academic assistant. Use ONLY the provided context. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:",
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )

        return resp.choices[0].message.content.strip()
