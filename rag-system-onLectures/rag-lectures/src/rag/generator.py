from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any
from functools import lru_cache

import torch
from transformers import (
    pipeline,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

from src.rag.retriever import RetrievedChunk


@dataclass(frozen=True)
class GenerationConfig:
    task: str = "text2text-generation"
    model: str = "google/flan-t5-base"  # Using Flan-T5 model
    max_context_chars: int = 12000
    max_new_tokens: int = 220  # Answer length limit


def build_context(chunks: List[RetrievedChunk], max_chars: int) -> str:
    parts = []
    used = 0

    for i, ch in enumerate(chunks, start=1):
        block = (
            f"[S{i}] chunk_id={ch.chunk_id} "
            f"lecture={ch.metadata.get('lecture_id')} "
            f"pages={ch.metadata.get('page_start')}-{ch.metadata.get('page_end')}\n"
            f"{ch.text}\n"
        )
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)

    return "\n---\n".join(parts)


def _hf_token() -> str | None:
    # Correct env var for authentication to Hugging Face Hub
    return os.getenv("HF_TOKEN") or None


def _load_tokenizer(model_id: str):
    tok = _hf_token()
    if tok:
        try:
            return T5Tokenizer.from_pretrained(model_id, token=tok)
        except TypeError:
            return T5Tokenizer.from_pretrained(model_id, use_auth_token=tok)
    return T5Tokenizer.from_pretrained(model_id)


def _load_model(task: str, model_id: str):
    tok = _hf_token()
    if task == "text2text-generation":
        if tok:
            try:
                return T5ForConditionalGeneration.from_pretrained(model_id, token=tok)
            except TypeError:
                return T5ForConditionalGeneration.from_pretrained(model_id, use_auth_token=tok)
        return T5ForConditionalGeneration.from_pretrained(model_id)

    # For other model types (optional), you can adjust this to suit your needs
    if tok:
        try:
            return T5ForConditionalGeneration.from_pretrained(model_id, token=tok)
        except TypeError:
            return T5ForConditionalGeneration.from_pretrained(model_id, use_auth_token=tok)
    return T5ForConditionalGeneration.from_pretrained(model_id)


@lru_cache(maxsize=2)
def _get_generator(task: str, model_id: str):
    """
    Cache the pipeline so it doesn't reload on every request.
    IMPORTANT: do NOT pass `use_auth_token` here; auth is handled in from_pretrained.
    """
    tokenizer = _load_tokenizer(model_id)
    model = _load_model(task, model_id)

    device = 0 if torch.cuda.is_available() else -1

    # For some models, pad_token may be missing; safe fallback
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def generate_answer(
    query: str,
    chunks: List[RetrievedChunk],
    cfg: GenerationConfig = GenerationConfig(),
) -> Dict[str, Any]:
    context = build_context(chunks, cfg.max_context_chars)

    instructions = (
        "You are a RAG assistant.\n"
        "Answer the question using ONLY the sources below.\n"
        "If the sources do not contain the answer, say: "
        "\"I don't know based on the provided lectures.\"\n"
        "Cite sources using [S1], [S2], ... that match the source blocks.\n"
    )

    prompt = (
        f"{instructions}\n\n"
        f"Question: {query}\n\n"
        f"Sources:\n{context}\n\n"
        f"Answer (with citations):"
    )

    gen = _get_generator(cfg.task, cfg.model)

    # Use max_new_tokens (preferred) instead of max_length for predictable output length.
    out = gen(
        prompt,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
    )

    # transformers pipelines return a list of dicts
    answer = out[0].get("generated_text", "").strip()

    return {"answer": answer, "citations": []}
