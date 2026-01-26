from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any
from functools import lru_cache
import re

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
    max_context_chars: int = 1500  # Reduced to prevent overwhelming the model
    max_new_tokens: int = 150  # Shorter, more focused answers
    min_length: int = 30  # Reduced from 50

    # Decoding controls - adjusted for better paraphrasing
    temperature: float = 0.9  # Higher temperature for more variation
    top_p: float = 0.92
    top_k: int = 40
    num_beams: int = 1
    no_repeat_ngram_size: int = 4  # Increased to prevent copying phrases
    repetition_penalty: float = 1.3  # Stronger penalty
    length_penalty: float = 0.8  # Slightly prefer shorter, focused answers

    # Reliability control
    hallucination_control: bool = True


def build_context(chunks: List[RetrievedChunk], max_chars: int) -> str:
    """
    Build context from retrieved Q&A pairs.
    Extract only key information to reduce copying.
    """
    parts: List[str] = []
    used = 0

    for i, ch in enumerate(chunks, start=1):
        meta = ch.metadata
        question = meta.get("question", "N/A")
        answer = meta.get("answer", "N/A")

        # Shorter, cleaner format
        block = f"{i}. {question} → {answer}\n"

        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n".join(parts) if parts else ""


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
    print(f"🔄 Loading generator: {model_id}", flush=True)

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


def _create_flan_prompt(query: str, context: str, strict_mode: bool) -> str:
    """
    Create a prompt that encourages paraphrasing and synthesis.
    Uses a QA format that Flan-T5 handles better.
    """
    if strict_mode:
        prompt = f"""Based on the following information, answer the question in your own words. Do not copy sentences directly.

Information:
{context}

Question: {query}

Answer (explain in a clear, natural way):"""
    else:
        prompt = f"""Using the information below and your knowledge, provide a helpful answer to the question. Write naturally in your own words.

Information:
{context}

Question: {query}

Answer:"""
    
    return prompt


def _extract_key_facts(chunks: List[RetrievedChunk]) -> List[str]:
    """
    Extract key facts from chunks to create a synthesized answer.
    This is a fallback when LLM copying is detected.
    """
    facts = []
    seen = set()
    
    for chunk in chunks[:3]:  # Top 3 chunks
        answer = chunk.metadata.get("answer", "").strip()
        if not answer or answer == "N/A" or answer in seen:
            continue
        
        seen.add(answer)
        
        # Extract key sentences (first sentence or short answers)
        sentences = answer.split('.')
        if sentences:
            key_fact = sentences[0].strip()
            if key_fact and len(key_fact) > 10:
                facts.append(key_fact)
    
    return facts


def _synthesize_from_facts(query: str, facts: List[str], chunks: List[RetrievedChunk]) -> str:
    """
    Create a synthesized answer from key facts when LLM fails.
    """
    if not facts:
        return "I don't have enough specific information to answer that question accurately."
    
    # Build a natural answer
    if len(facts) == 1:
        answer = f"{facts[0]}. [Reference 1]"
    elif len(facts) == 2:
        answer = f"{facts[0]}. Additionally, {facts[1].lower()}. [References 1-2]"
    else:
        answer = f"{facts[0]}. "
        answer += f"Based on customer experiences, {facts[1].lower()}. "
        if len(facts) > 2:
            answer += f"It's also worth noting that {facts[2].lower()}. "
        answer += "[References 1-3]"
    
    return answer


def _detect_copying(answer: str, chunks: List[RetrievedChunk], threshold: float = 0.6) -> bool:
    """
    Detect if the answer contains substantial copying from sources.
    Returns True if copying is detected.
    """
    if not answer or len(answer) < 20:
        return False
    
    answer_lower = answer.lower()
    
    # Check each chunk for copied phrases
    for chunk in chunks:
        source_text = chunk.metadata.get("answer", "").lower()
        if not source_text or source_text == "n/a":
            continue
        
        # Split into words
        answer_words = answer_lower.split()
        source_words = source_text.split()
        
        # Check for consecutive word matches (5+ words in a row = copying)
        for i in range(len(answer_words) - 4):
            phrase = ' '.join(answer_words[i:i+5])
            if phrase in source_text:
                print(f"⚠️ Detected copying: '{phrase}'", flush=True)
                return True
        
        # Check for high word overlap
        answer_set = set(w for w in answer_words if len(w) > 3)
        source_set = set(w for w in source_words if len(w) > 3)
        
        if answer_set and source_set:
            overlap = len(answer_set & source_set) / len(answer_set)
            if overlap > threshold:
                print(f"⚠️ High word overlap detected: {overlap:.1%}", flush=True)
                return True
    
    return False


def _is_valid_answer(answer: str, min_words: int = 8) -> bool:
    """
    Check if the generated answer is valid and substantive.
    """
    if not answer or len(answer.strip()) < 15:
        return False
    
    # Check if it's only a citation
    citation_only = re.match(r'^\s*\[?Reference\s+\d+\]?\s*$', answer.strip(), re.IGNORECASE)
    if citation_only:
        return False
    
    # Check word count
    words = answer.split()
    if len(words) < min_words:
        return False
    
    # Check for instruction leakage
    bad_markers = [
        "in your own words",
        "do not copy",
        "based on the following",
        "information:",
        "answer (explain",
    ]
    
    for marker in bad_markers:
        if marker.lower() in answer.lower():
            return False
    
    return True


def _clean_answer(answer: str) -> str:
    """
    Clean up the generated answer.
    """
    answer = answer.strip()
    
    # Remove instruction artifacts
    patterns_to_remove = [
        r'^Based on the following.*?Answer.*?:',
        r'^Using the information.*?Answer:',
        r'^\s*Answer\s*:\s*',
    ]
    
    for pattern in patterns_to_remove:
        answer = re.sub(pattern, '', answer, flags=re.IGNORECASE | re.DOTALL)
    
    return answer.strip()


def generate_answer(
    query: str,
    chunks: List[RetrievedChunk],
    cfg: GenerationConfig = GenerationConfig(),
) -> Dict[str, Any]:
    """
    Generate a grounded answer using Flan-T5 based on retrieved Q&A pairs.
    Falls back to manual synthesis if copying is detected.
    """
    # Validate inputs
    if not query or not query.strip():
        return {"answer": "Please provide a question.", "citations": []}
    
    if not chunks:
        return {
            "answer": "I don't have enough information to answer that question. Please try a different question.",
            "citations": []
        }
    
    # Build context from chunks
    context = build_context(chunks, cfg.max_context_chars)
    
    if not context:
        return {
            "answer": "I couldn't retrieve relevant information for your question.",
            "citations": []
        }
    
    # Create optimized prompt
    prompt = _create_flan_prompt(
        query=query,
        context=context,
        strict_mode=cfg.hallucination_control
    )

    # Get generator pipeline
    gen = _get_generator(cfg.task, cfg.model)

    # Generation parameters optimized to reduce copying
    gen_kwargs = {
        "max_new_tokens": cfg.max_new_tokens,
        "min_length": cfg.min_length,
        "do_sample": True,
        "temperature": max(0.8, float(cfg.temperature)),  # Higher temp to encourage variation
        "top_p": float(cfg.top_p),
        "top_k": cfg.top_k,
        "repetition_penalty": float(cfg.repetition_penalty),
        "length_penalty": float(cfg.length_penalty),
        "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
        "early_stopping": False,
    }
    
    # Try LLM generation
    llm_answer = None
    
    try:
        out = gen(prompt, **gen_kwargs)
        raw_answer = out[0].get("generated_text", "").strip()
        cleaned = _clean_answer(raw_answer)
        
        # Check if answer is valid and not copying
        if _is_valid_answer(cleaned):
            # Check for copying
            if not _detect_copying(cleaned, chunks):
                llm_answer = cleaned
                print("✅ LLM generated valid answer without copying", flush=True)
            else:
                print("⚠️ LLM answer contains copying, using fallback synthesis", flush=True)
        else:
            print(f"⚠️ Invalid LLM answer: '{cleaned[:100]}'", flush=True)
            
    except Exception as e:
        print(f"❌ Generation error: {e}", flush=True)
    
    # Fallback: Manual synthesis from key facts
    if llm_answer is None:
        print("🔧 Using manual synthesis from key facts", flush=True)
        facts = _extract_key_facts(chunks)
        answer = _synthesize_from_facts(query, facts, chunks)
    else:
        answer = llm_answer
    
    # Extract citations
    citations: List[int] = []
    for i in range(1, len(chunks) + 1):
        patterns = [
            f"[Reference {i}]",
            f"Reference {i}",
            f"[Ref {i}]",
            f"References {i}",
            f"[References 1-{i}]" if i > 1 else None,
        ]
        
        for pattern in patterns:
            if pattern and pattern.lower() in answer.lower():
                if i not in citations:
                    citations.append(i)
                break
    
    # If no citations were added by synthesis, add them
    if not citations and answer and "Reference" in answer:
        # Extract all reference numbers mentioned
        ref_matches = re.findall(r'Reference[s]?\s*(\d+)', answer, re.IGNORECASE)
        citations = sorted(set(int(m) for m in ref_matches if int(m) <= len(chunks)))

    return {
        "answer": answer,
        "citations": sorted(citations) if citations else [1]  # At least cite the top result
    }