"""
Generation module using a local language model (LLM-first mode).

The model is allowed to answer questions using its own knowledge.
Retrieved documents (if any) are used as OPTIONAL supporting context,
not as a strict knowledge boundary.
"""

from __future__ import annotations
from typing import List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------
# Model configuration (offline)
# -----------------------------
MODEL_NAME = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def assemble_prompt(query: str, retrieved_docs: List[str]) -> str:
    """
    Build a prompt that allows the model to answer normally,
    while optionally using retrieved documents as support.
    """

    if retrieved_docs:
        context = "\n\n".join(retrieved_docs)
        prompt = f"""
You are a knowledgeable assistant.

You may use your own general knowledge to answer the question.
The additional context below may help, but it is not mandatory.

ADDITIONAL CONTEXT (optional):
{context}

QUESTION:
{query}

Provide a clear, detailed, and well-explained answer:
"""
    else:
        prompt = f"""
You are a knowledgeable assistant.

Answer the following question clearly and in detail.

QUESTION:
{query}

ANSWER:
"""
    return prompt


def generate_answer(query: str, retrieved_docs: List[str]) -> str:
    """
    Generate a detailed answer using the model's own knowledge.
    Retrieved documents are optional enhancements.
    """

    prompt = assemble_prompt(query, retrieved_docs)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_length=512,          # long answers
        min_length=150,          # avoid short replies
        temperature=0.5,         # more natural language
        repetition_penalty=1.1,
        do_sample=True,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer.strip()
