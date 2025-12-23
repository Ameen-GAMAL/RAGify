"""
End-to-end RAG Pipeline for E-commerce Q&A
Combines retrieval + generation with self-learning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.rag.retriever import Retriever, RetrievedChunk
from src.rag.generator import generate_answer, GenerationConfig


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for RAG pipeline."""
    top_k: int = 6
    use_generation: bool = True

    # Retrieval filtering
    min_similarity: float = 0.5

    # Generation controls (UI should map to these)
    temperature: float = 0.7
    hallucination_control: bool = True


class RAGPipeline:
    """
    End-to-end RAG pipeline for E-commerce Q&A.
    """

    def __init__(self):
        print("\n" + "=" * 70, flush=True)
        print(" Initializing E-commerce Q&A RAG Pipeline", flush=True)
        print("=" * 70, flush=True)

        self.retriever = Retriever()

        print("RAG Pipeline ready!\n", flush=True)

    def synthesize_answer_simple(self, query: str, retrieved: List[RetrievedChunk], cfg: RAGConfig) -> str:
        """
        Simple answer synthesis (fallback method without LLM).
        """
        if not retrieved:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or ask something else."

        relevant = [r for r in retrieved if r.score >= cfg.min_similarity]
        if not relevant:
            return (
                f"I found some results, but they don't seem relevant enough "
                f"(similarity < {cfg.min_similarity:.2f}). Please try rephrasing your question."
            )

        best_match = relevant[0]
        best_question = best_match.metadata.get("question", "")
        best_answer = best_match.metadata.get("answer", "")
        score = best_match.score

        if score > 0.85:
            answer = f"""**Direct Answer:**

{best_answer}

**Confidence:** {score:.1%} match

**Source Question:** "{best_question}"
"""
            if len(relevant) > 1:
                answer += "\n**Additional Related Information:**\n"
                for i, chunk in enumerate(relevant[1:3], 1):
                    related_a = chunk.metadata.get("answer", "")
                    answer += f"\n{i}. {related_a}\n"

        elif score > 0.75:
            answer = f"""**Answer (from similar question):**

{best_answer}

**Your Question:** "{query}"
**Similar Question Found:** "{best_question}"
**Confidence:** {score:.1%}

**Why This Answer Helps:**
This response addresses a similar question and should provide the information you're looking for.
"""
            if len(relevant) > 1:
                answer += "\n**See Also:**\n"
                for chunk in relevant[1:2]:
                    related_q = chunk.metadata.get("question", "")
                    related_a = chunk.metadata.get("answer", "")
                    answer += f"• *{related_q}*\n  {related_a}\n\n"

        else:
            answer = f"""**Multiple Perspectives on Your Question:**

I found several related answers that together address different aspects of your question: *"{query}"*

"""
            for i, chunk in enumerate(relevant[:3], 1):
                q = chunk.metadata.get("question", "")
                a = chunk.metadata.get("answer", "")
                confidence = chunk.score
                answer += f"""**Option {i}** (Relevance: {confidence:.1%})
*Question:* {q}
*Answer:* {a}

"""
            answer += "**Note:** These answers provide different perspectives that may be relevant to your situation.\n"

        return answer

    def run(self, query: str, top_k: int = 6, config: Optional[RAGConfig] = None) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline.
        """
        if config is None:
            config = RAGConfig(top_k=top_k)

        # Step 1: Retrieve similar Q&A pairs
        print(f" Retrieving top-{top_k} similar Q&A pairs for: '{query}'", flush=True)
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        print(f" Retrieved {len(retrieved)} Q&A pairs", flush=True)

        # Step 2: Generate/Synthesize answer
        print(" Generating answer...", flush=True)

        used_llm = False

        if config.use_generation and len(retrieved) > 0:
            try:
                gen_config = GenerationConfig(
                    temperature=float(config.temperature),
                    hallucination_control=bool(config.hallucination_control),
                )
                result = generate_answer(query, retrieved, gen_config)
                answer = result["answer"]
                used_llm = True
                print(" LLM-generated answer ready\n", flush=True)
            except Exception as e:
                # If SentencePiece is missing, generator.py raises a clear RuntimeError.
                # We still fallback, but the user will see the message in terminal.
                print(f" LLM generation failed: {e}", flush=True)
                print(" Falling back to simple synthesis...\n", flush=True)
                answer = self.synthesize_answer_simple(query, retrieved, config)
                used_llm = False
        else:
            answer = self.synthesize_answer_simple(query, retrieved, config)
            print(" Answer synthesized\n", flush=True)

        # Step 3: Format sources
        sources: List[Dict[str, Any]] = []
        for r in retrieved:
            meta = r.metadata
            sources.append(
                {
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "qa_id": meta.get("qa_id"),
                    "question": meta.get("question", ""),
                    "answer": meta.get("answer", ""),
                    "q_type": meta.get("q_type", "other"),
                    "sentiment": meta.get("sentiment", "neutral"),
                }
            )

        # Step 4: Metadata
        metadata = {
            "total_retrieved": len(retrieved),
            "avg_score": round(sum(r.score for r in retrieved) / len(retrieved), 4) if retrieved else 0,
            "top_score": round(retrieved[0].score, 4) if retrieved else 0,
            "min_score": round(retrieved[-1].score, 4) if retrieved else 0,
            "used_llm": used_llm,
            "temperature": float(config.temperature),
            "hallucination_control": bool(config.hallucination_control),
        }

        return {"query": query, "answer": answer, "sources": sources, "metadata": metadata}


def test_pipeline():
    print("\n" + "=" * 70, flush=True)
    print(" Test the RAG pipeline", flush=True)
    print("=" * 70, flush=True)

    pipeline = RAGPipeline()

    test_queries = [
        "Does this case fit iPhone 13?",
        "What is the battery life?",
        "How long does shipping take?",
    ]

    for q in test_queries:
        print(f"\n{'─' * 70}", flush=True)
        print(f"Query: {q}", flush=True)
        print(f"{'─' * 70}", flush=True)

        result = pipeline.run(q, top_k=3)
        print("\n Answer:", flush=True)
        print(result["answer"], flush=True)

        print("\n Metadata:", flush=True)
        for k, v in result["metadata"].items():
            print(f"   {k}: {v}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("✅ Pipeline test complete!", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    test_pipeline()
