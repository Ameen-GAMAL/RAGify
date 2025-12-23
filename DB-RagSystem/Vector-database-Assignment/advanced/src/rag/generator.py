from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any
from functools import lru_cache

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from src.rag.retriever import RetrievedChunk


@dataclass(frozen=True)
class GenerationConfig:
    # HF pipeline task/model
    task: str = "text2text-generation"
    model: str = "google/flan-t5-large"

    # Context/output sizing
    max_context_chars: int = 2000
    max_new_tokens: int = 600  # 1500 can be slow; raise if you truly need long outputs

    # Decoding controls
    temperature: float = 0.7        # 0 => deterministic; >0 => sampling
    top_p: float = 0.9
    num_beams: int = 4              # beam search for quality
    no_repeat_ngram_size: int = 3   # reduce repetition

    # Reliability control
    hallucination_control: bool = True  # strict grounding when True


def build_context(chunks: List[RetrievedChunk], max_chars: int) -> str:
    """
    Build context from retrieved Q&A pairs.
    """
    parts: List[str] = []
    used = 0

    for i, ch in enumerate(chunks, start=1):
        meta = ch.metadata
        question = meta.get("question", "N/A")
        answer = meta.get("answer", "N/A")

        block = (
            f"[Reference {i}]\n"
            f"Q: {question}\n"
            f"A: {answer}\n"
        )

        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n" + "=" * 60 + "\n" + "\n".join(parts) + "\n"


def _hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or None


def _ensure_sentencepiece():
    """
    Flan-T5 uses SentencePiece. If missing, tokenizer load fails and users silently fall back.
    Fail fast with clear instructions.
    """
    try:
        import sentencepiece  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "T5/Flan-T5 requires the 'sentencepiece' package.\n"
            "Install it in your active venv:\n"
            "  pip install -U sentencepiece protobuf\n"
            "Then restart Streamlit."
        ) from e


def _load_tokenizer(model_id: str):
    _ensure_sentencepiece()
    tok = _hf_token()
    if tok:
        try:
            return AutoTokenizer.from_pretrained(model_id, token=tok)
        except TypeError:
            return AutoTokenizer.from_pretrained(model_id, use_auth_token=tok)
    return AutoTokenizer.from_pretrained(model_id)


def _load_model(model_id: str):
    _ensure_sentencepiece()
    tok = _hf_token()
    if tok:
        try:
            return AutoModelForSeq2SeqLM.from_pretrained(model_id, token=tok)
        except TypeError:
            return AutoModelForSeq2SeqLM.from_pretrained(model_id, use_auth_token=tok)
    return AutoModelForSeq2SeqLM.from_pretrained(model_id)


@lru_cache(maxsize=2)
def _get_generator(task: str, model_id: str):
    """Cache the HF pipeline to avoid reloading."""
    print(f" Loading generator: {model_id}", flush=True)

    tokenizer = _load_tokenizer(model_id)
    model = _load_model(model_id)

    device = 0 if torch.cuda.is_available() else -1
    print(f"   Device: {'GPU' if device == 0 else 'CPU'}", flush=True)

    # Safe pad token handling
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def _make_instructions(hallucination_control: bool, num_refs: int) -> str:
    base_rules = [
        "Provide a helpful, well-structured answer.",
        "Start with a direct answer to the user's question.",
        "Cite sources using [Reference 1], [Reference 2], etc. when using specific information.",
        f"VALID REFERENCES: 1..{num_refs}. Do not cite any other reference numbers.",
        "Never output only a citation. Always include an explanation in words.",
        "If multiple references overlap, synthesize them into one coherent response.",
        "Paraphrase; do not copy sentences verbatim from the references (short phrases are ok).",
        "Use bullet points or numbered lists when listing items or steps.",
    ]

    if hallucination_control:
        base_rules += [
            "HALLUCINATION CONTROL (STRICT): Only use information from the provided references.",
            "If the references do not contain enough information, say what's missing and ask a clarifying question.",
            "Do not guess product specs, compatibility, or policies not stated in the references.",
        ]
    else:
        base_rules += [
            "If references are insufficient, you may add GENERAL GUIDANCE.",
            "Clearly label any such content as: 'General guidance (not in references)'.",
            "Do not cite references for that general guidance section.",
        ]

    return (
        "You are an expert customer support assistant.\n\n"
        "INSTRUCTIONS:\n"
        + "\n".join([f"- {r}" for r in base_rules])
        + "\n\n"
        "RESPONSE STRUCTURE:\n"
        "- Direct answer\n"
        "- Details with citations\n"
        "- Practical advice / next steps\n"
    )




def generate_answer(
    query: str,
    chunks: List[RetrievedChunk],
    cfg: GenerationConfig = GenerationConfig(),
) -> Dict[str, Any]:
    """
    Generate a grounded answer using Flan-T5 based on retrieved Q&A pairs.
    """
    context = build_context(chunks, cfg.max_context_chars)
    instructions = _make_instructions(cfg.hallucination_control, num_refs=len(chunks))

    prompt = (
        f"{instructions}\n"
        f"USER QUESTION:\n{query}\n\n"
        f"REFERENCE INFORMATION:\n{context}\n\n"
        f"FINAL ANSWER:\n"
    )

    gen = _get_generator(cfg.task, cfg.model)

    # # Decoding strategy:
    # # - temperature <= 0 => deterministic (sampling off)
    # # - temperature > 0  => sampling on
    # if cfg.temperature <= 0.0:
    #     gen_kwargs = {
    #         "max_new_tokens": cfg.max_new_tokens,
    #         "do_sample": False,
    #         "num_beams": max(1, int(cfg.num_beams)),
    #         "no_repeat_ngram_size": int(cfg.no_repeat_ngram_size),
    #     }
    # else:
    #     gen_kwargs = {
    #         "max_new_tokens": cfg.max_new_tokens,
    #         "do_sample": True,
    #         "temperature": float(cfg.temperature),
    #         "top_p": float(cfg.top_p),
    #         "num_beams": max(1, int(cfg.num_beams)),
    #         "no_repeat_ngram_size": int(cfg.no_repeat_ngram_size),
    #     }

    out = gen(
    prompt,
    max_new_tokens=cfg.max_new_tokens,
    min_new_tokens=40,          # ✅ prevents "[Reference X]" only
    do_sample=True,
    temperature=float(cfg.temperature),
    top_p=float(cfg.top_p),
    no_repeat_ngram_size=int(cfg.no_repeat_ngram_size),
)
    answer = out[0].get("generated_text", "").strip()

    citations: List[int] = []
    for i in range(1, len(chunks) + 1):
        if f"[Reference {i}]" in answer or f"Reference {i}" in answer:
            citations.append(i)

    return {"answer": answer, "citations": citations}
