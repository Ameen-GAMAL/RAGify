"""
RAG System - Phase 4: Generation Module
Advanced Information Retrieval Course

Implements:
- Retrieval-Augmented Generation (RAG) response synthesis
- Prompt templates + context formatting
- Optional LLM backends:
    (A) OpenAI Chat Completions (if OPENAI_API_KEY set + openai installed)
    (B) HuggingFace local generation (if transformers installed)
    (C) Deterministic fallback (no LLM) that is still grounded in retrieved products

Expected dataset columns (processed_data.csv):
description,merchantLink,merchantName,price,productDetails,productLink,
productName,reviewsCount,reviewsScore,searchKeyword,withoutDiscountPrice
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path for imports
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Phase 3 retriever
from src.phase3_retrieval import AdvancedRetriever  # uses your existing retrieval pipeline






# -----------------------------
# Data models
# -----------------------------

@dataclass
class ProductRecord:
    doc_id: int
    productName: str = ""
    description: str = ""
    productDetails: str = ""
    price: str = ""
    withoutDiscountPrice: str = ""
    merchantName: str = ""
    merchantLink: str = ""
    productLink: str = ""
    reviewsScore: str = ""
    reviewsCount: str = ""
    searchKeyword: str = ""

    @staticmethod
    def from_row(doc_id: int, row: pd.Series) -> "ProductRecord":
        def _s(x: Any) -> str:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return ""
            return str(x).strip()

        return ProductRecord(
            doc_id=doc_id,
            productName=_s(row.get("productName")),
            description=_s(row.get("description")),
            productDetails=_s(row.get("productDetails")),
            price=_s(row.get("price")),
            withoutDiscountPrice=_s(row.get("withoutDiscountPrice")),
            merchantName=_s(row.get("merchantName")),
            merchantLink=_s(row.get("merchantLink")),
            productLink=_s(row.get("productLink")),
            reviewsScore=_s(row.get("reviewsScore")),
            reviewsCount=_s(row.get("reviewsCount")),
            searchKeyword=_s(row.get("searchKeyword")),
        )


@dataclass
class GenerationConfig:
    top_k: int = 5
    retrieval_method: str = "hybrid"  # semantic | bm25 | hybrid
    rerank: bool = False
    strict_grounding: bool = True     # force answers only from retrieved context
    max_context_chars: int = 6500
    llm_backend: str = "auto"         # auto | openai | hf | fallback
    openai_model: str = "gpt-4.1-mini"
    hf_model: str = "google/flan-t5-base"  # text2text; small-ish; optional
    temperature: float = 0.2


# -----------------------------
# Context building + prompt
# -----------------------------

def _trim(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


class ContextBuilder:
    """
    Creates a compact, citation-friendly context block from retrieved products.
    """

    def __init__(self, max_context_chars: int = 6500):
        self.max_context_chars = max_context_chars

    def build(self, products: List[ProductRecord]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Returns:
          context_text: string with numbered sources [1], [2], ...
          sources: list of dicts with doc_id + key fields
        """
        sources: List[Dict[str, Any]] = []
        blocks: List[str] = []

        for i, p in enumerate(products, start=1):
            # Keep each block concise but useful
            name = _trim(p.productName, 120) or "(missing name)"
            desc = _trim(p.description, 350)
            details = _trim(p.productDetails, 250)

            price_bits = []
            if p.price:
                price_bits.append(f"price={p.price}")
            if p.withoutDiscountPrice and p.withoutDiscountPrice != p.price:
                price_bits.append(f"was={p.withoutDiscountPrice}")
            price_str = ", ".join(price_bits)

            rating_bits = []
            if p.reviewsScore:
                rating_bits.append(f"rating={p.reviewsScore}")
            if p.reviewsCount:
                rating_bits.append(f"reviews={p.reviewsCount}")
            rating_str = ", ".join(rating_bits)

            merchant = p.merchantName or ""
            link = p.productLink or p.merchantLink or ""

            block = (
                f"[{i}] productName: {name}\n"
                f"    description: {desc}\n"
                + (f"    productDetails: {details}\n" if details else "")
                + (f"    merchantName: {merchant}\n" if merchant else "")
                + (f"    pricing: {price_str}\n" if price_str else "")
                + (f"    reviews: {rating_str}\n" if rating_str else "")
                + (f"    link: {link}\n" if link else "")
            )

            blocks.append(block)
            sources.append({
                "source_id": i,
                "doc_id": p.doc_id,
                "productName": p.productName,
                "productLink": p.productLink,
                "merchantName": p.merchantName,
            })

        context_text = "\n".join(blocks)

        # Hard cap context size
        if len(context_text) > self.max_context_chars:
            context_text = context_text[: self.max_context_chars].rstrip() + "\n…"

        return context_text, sources


class PromptBuilder:
    """
    Produces a grounded prompt that forces the model to cite sources [1], [2], ...
    """

    def __init__(self, strict_grounding: bool = True):
        self.strict_grounding = strict_grounding

    def build(self, user_query: str, context: str) -> str:
        grounding_rules = ""
        if self.strict_grounding:
            grounding_rules = (
                "Rules:\n"
                "1) Use ONLY the information in SOURCES.\n"
                "2) If an important detail is missing, say you don't have it.\n"
                "3) When you mention a product or claim, cite it like [1] or [2].\n"
                "4) Do NOT invent prices, ratings, brands, or features.\n"
            )

        prompt = (
            "You are a shopping assistant for a Retrieval-Augmented Generation (RAG) system.\n"
            "Your job is to answer the user using the retrieved product SOURCES.\n\n"
            f"{grounding_rules}\n"
            "SOURCES:\n"
            f"{context}\n"
            "USER QUESTION:\n"
            f"{user_query}\n\n"
            "Write a helpful answer that:\n"
            "- briefly summarizes what the user is looking for\n"
            "- recommends 3-5 relevant products from SOURCES\n"
            "- for each recommended product include: name, why it matches, and any available price/ratings\n"
            "- include citations like [1] after the sentences that use that source\n"
        )
        return prompt


# -----------------------------
# LLM backends
# -----------------------------

class LLMClient:
    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        raise NotImplementedError


class OpenAIChatClient(LLMClient):
    """
    Uses OpenAI Chat Completions (optional dependency).
    Docs: https://platform.openai.com/docs/api-reference/chat  (referenced in citations in write-up)
    """

    def __init__(self, model: str):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "openai package not installed. Install it or switch backend to 'fallback' or 'hf'."
            ) from e

        self.client = OpenAI()
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        # Keep it simple and stable for coursework
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are a careful assistant that cites sources like [1], [2]."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""


class HuggingFaceClient(LLMClient):
    """
    Optional local generation using transformers.
    Works well with text2text models like FLAN-T5 for short grounded synthesis.
    """

    def __init__(self, model_name: str):
        try:
            from transformers import pipeline  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "transformers not installed. Install it or switch backend to 'fallback' or 'openai'."
            ) from e

        # Using text2text-generation keeps outputs more instruction-following than raw LM
        self.pipe = pipeline("text2text-generation", model=model_name)

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=350,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
        )
        return out[0]["generated_text"] if out else ""


class FallbackClient(LLMClient):
    """
    Deterministic non-LLM response: still grounded and useful,
    good for running without any external model/API.
    """

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        # Prompt is not used; this backend will be driven by structured data in GenerationModule.
        return ""


# -----------------------------
# Generation module
# -----------------------------

class GenerationModule:
    def __init__(
        self,
        processed_data_path: str = "data/processed/processed_data.csv",
        vector_db_path: str = "models/vector_db",
        config: Optional[GenerationConfig] = None,
    ):
        self.config = config or GenerationConfig()

        # Load data table once for doc_id -> row lookups
        self.df = pd.read_csv(PROJECT_ROOT / processed_data_path)
        self._validate_columns()

        # Retrieval system from Phase 3
        self.retriever = AdvancedRetriever(
            vector_db_path=vector_db_path,
            processed_data_path=processed_data_path,
        )

        self.context_builder = ContextBuilder(max_context_chars=self.config.max_context_chars)
        self.prompt_builder = PromptBuilder(strict_grounding=self.config.strict_grounding)

        self.llm = self._init_llm_backend()

        # Where we log runs (optional, useful for your report “execution examples”)
        self.log_path = PROJECT_ROOT / "outputs" / "reports" / "phase4_generation_logs.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _validate_columns(self) -> None:
        required = {
            "productName",
            "description",
            "productLink",
            "merchantName",
            "price",
            "reviewsScore",
            "reviewsCount",
            "productDetails",
            "merchantLink",
            "withoutDiscountPrice",
            "searchKeyword",
        }
        missing = sorted(list(required - set(self.df.columns)))
        if missing:
            raise ValueError(f"processed_data.csv is missing required columns: {missing}")

    def _init_llm_backend(self) -> LLMClient:
        backend = (self.config.llm_backend or "auto").lower()

        if backend == "auto":
            # Prefer OpenAI if key exists; else try HF; else fallback.
            if os.getenv("OPENAI_API_KEY"):
                backend = "openai"
            else:
                backend = "hf"

        if backend == "openai":
            return OpenAIChatClient(model=self.config.openai_model)
        if backend == "hf":
            try:
                return HuggingFaceClient(model_name=self.config.hf_model)
            except Exception:
                return FallbackClient()
        if backend == "fallback":
            return FallbackClient()

        raise ValueError(f"Unknown llm_backend: {self.config.llm_backend}")

    def _records_from_results(self, results: List[Dict[str, Any]]) -> List[ProductRecord]:
        records: List[ProductRecord] = []
        for r in results:
            doc_id = int(r.get("doc_id"))
            if doc_id < 0 or doc_id >= len(self.df):
                continue
            row = self.df.iloc[doc_id]
            records.append(ProductRecord.from_row(doc_id=doc_id, row=row))
        return records

    def _fallback_answer(self, query: str, products: List[ProductRecord]) -> str:
        if not products:
            return "I couldn’t find relevant products in the retrieved results for your query."

        lines = []
        lines.append(f"You asked: {query}")
        lines.append("")
        lines.append("Here are some relevant products from the retrieved results:")
        lines.append("")

        for i, p in enumerate(products[: self.config.top_k], start=1):
            name = p.productName or "(missing name)"
            desc = _trim(p.description, 220)
            price_bits = []
            if p.price:
                price_bits.append(f"Price: {p.price}")
            if p.withoutDiscountPrice and p.withoutDiscountPrice != p.price:
                price_bits.append(f"Was: {p.withoutDiscountPrice}")
            rating_bits = []
            if p.reviewsScore:
                rating_bits.append(f"Rating: {p.reviewsScore}")
            if p.reviewsCount:
                rating_bits.append(f"Reviews: {p.reviewsCount}")

            extra = " | ".join([b for b in price_bits + rating_bits if b])
            lines.append(f"{i}) {name} [{i}]")
            if desc:
                lines.append(f"   - {desc} [{i}]")
            if p.merchantName:
                lines.append(f"   - Merchant: {p.merchantName} [{i}]")
            if extra:
                lines.append(f"   - {extra} [{i}]")
            if p.productLink:
                lines.append(f"   - Link: {p.productLink} [{i}]")
            lines.append("")

        return "\n".join(lines).strip()

    def answer(self, user_query: str) -> Dict[str, Any]:
        """
        End-to-end RAG:
          1) retrieve top_k docs
          2) build context
          3) generate response (LLM or fallback)
          4) return response + sources
        """
        start_t = time.time()

        # Retrieve
        results = self.retriever.retrieve(
            user_query,
            method=self.config.retrieval_method,
            top_k=self.config.top_k,
            rerank=self.config.rerank,
            process_query=True,
        )

        products = self._records_from_results(results)
        context, sources = self.context_builder.build(products)
        prompt = self.prompt_builder.build(user_query, context)

        # Generate
        if isinstance(self.llm, FallbackClient):
            response = self._fallback_answer(user_query, products)
        else:
            response = self.llm.generate(prompt, temperature=self.config.temperature).strip()

        elapsed = time.time() - start_t

        out = {
            "query": user_query,
            "retrieval_method": self.config.retrieval_method,
            "rerank": self.config.rerank,
            "top_k": self.config.top_k,
            "backend": self.config.llm_backend,
            "response": response,
            "sources": sources,
            "latency_sec": round(elapsed, 4),
        }

        # Log for reporting/debugging
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

        return out


def main():
    print("=" * 80)
    print("RAG SYSTEM - PHASE 4: GENERATION MODULE")
    print("Advanced Information Retrieval Course")
    print("=" * 80)

    # You can tune these defaults
    config = GenerationConfig(
        top_k=5,
        retrieval_method="hybrid",
        rerank=False,
        strict_grounding=True,
        llm_backend="auto",      # openai if OPENAI_API_KEY exists; else hf; else fallback
        openai_model="gpt-4.1-mini",
        hf_model="google/flan-t5-base",
        temperature=0.2,
    )

    generator = GenerationModule(config=config)

    # Simple interactive CLI
    print("\n✅ Phase 4 ready. Type a query (or 'exit').")
    while True:
        user_query = input("\nQuery> ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            break

        result = generator.answer(user_query)
        print("\n" + "-" * 80)
        print(result["response"])
        print("-" * 80)
        print(f"Latency: {result['latency_sec']}s | Sources: {len(result['sources'])}")
        if result["sources"]:
            print("Source mapping:")
            for s in result["sources"]:
                name = (s.get("productName") or "")[:70]
                print(f"  [{s['source_id']}] doc_id={s['doc_id']}  {name}")

    print("\n✓ Logs saved to:", generator.log_path)


if __name__ == "__main__":
    main()
