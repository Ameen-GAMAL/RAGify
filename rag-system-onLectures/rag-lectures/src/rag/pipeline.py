from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.rag.retriever import Retriever
from src.rag.generator import generate_answer, GenerationConfig
from src.learn.feedback_store import FeedbackPaths


@dataclass(frozen=True)
class RAGConfig:
    top_k: int = 6


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()

    def run(self, query: str, top_k: int = 6) -> Dict[str, Any]:
        # Step 1: Retrieve the top_k relevant documents using the retriever
        retrieved = self.retriever.retrieve(query, top_k=top_k)

        # Step 2: Generate the answer using the Flan-T5-based generator
        gen = generate_answer(
            query=query,
            chunks=retrieved,
            cfg=GenerationConfig(),
        )

        # Step 3: Format the sources from the retrieved documents
        sources = [
            {
                "chunk_id": r.chunk_id,
                "score": r.score,
                "lecture_id": r.metadata.get("lecture_id"),
                "page_start": r.metadata.get("page_start"),
                "page_end": r.metadata.get("page_end"),
                "snippet": (r.text[:240] + "…") if len(r.text) > 240 else r.text,
            }
            for r in retrieved
        ]

        # Return the final result with query, generated answer, and sources
        return {
            "query": query,
            "answer": gen["answer"],
            "sources": sources,
        }
