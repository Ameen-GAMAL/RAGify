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
    temperature: float = 0.9  # Higher default for better paraphrasing
    hallucination_control: bool = True


class RAGPipeline:
    """
    End-to-end RAG pipeline for E-commerce Q&A.
    """

    def __init__(self):
        print("\n" + "=" * 70, flush=True)
        print("🚀 Initializing E-commerce Q&A RAG Pipeline", flush=True)
        print("=" * 70, flush=True)

        self.retriever = Retriever()

        print("✅ RAG Pipeline ready!\n", flush=True)

    def synthesize_answer_simple(self, query: str, retrieved: List[RetrievedChunk], cfg: RAGConfig) -> str:
        """
        Enhanced answer synthesis without LLM.
        Extracts and combines information naturally.
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

        # High confidence answer
        if score > 0.85:
            answer = f"{best_answer}\n\n"
            
            if len(relevant) > 1:
                # Add supporting information
                additional_info = []
                for chunk in relevant[1:3]:
                    related_a = chunk.metadata.get("answer", "")
                    if related_a and related_a != best_answer:
                        additional_info.append(related_a)
                
                if additional_info:
                    answer += "**Additional information:** "
                    answer += " ".join(additional_info[:2])
            
            answer += f"\n\n*Confidence: {score:.1%}*"

        # Good match
        elif score > 0.70:
            # Synthesize from multiple sources
            answers_seen = {best_answer}
            combined_info = [best_answer]
            
            for chunk in relevant[1:3]:
                ans = chunk.metadata.get("answer", "")
                if ans and ans not in answers_seen and len(combined_info) < 3:
                    answers_seen.add(ans)
                    combined_info.append(ans)
            
            if len(combined_info) == 1:
                answer = f"{combined_info[0]}\n\n*Based on: \"{best_question}\"*"
            else:
                # Natural synthesis
                answer = f"{combined_info[0]}. "
                if len(combined_info) > 1:
                    answer += f"Additionally, {combined_info[1].lower() if combined_info[1][0].isupper() else combined_info[1]}"
                if len(combined_info) > 2:
                    answer += f" Also note that {combined_info[2].lower() if combined_info[2][0].isupper() else combined_info[2]}"
                
                answer += f"\n\n*Confidence: {score:.1%}*"

        # Multiple perspectives
        else:
            answer = "Based on the available information:\n\n"
            
            for i, chunk in enumerate(relevant[:3], 1):
                q = chunk.metadata.get("question", "")
                a = chunk.metadata.get("answer", "")
                confidence = chunk.score
                
                # Extract first sentence or key point
                key_point = a.split('.')[0] if '.' in a else a
                
                answer += f"**{i}.** {key_point}"
                if len(a.split('.')) > 1:
                    answer += "..."
                answer += f" *(Relevance: {confidence:.1%})*\n\n"
            
            answer += "*Note: These answers address different aspects that may be relevant to your question.*"

        return answer

    def run(self, query: str, top_k: int = 6, config: Optional[RAGConfig] = None) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline with improved synthesis.
        """
        if config is None:
            config = RAGConfig(top_k=top_k)

        # Step 1: Retrieve similar Q&A pairs
        print(f"🔍 Retrieving top-{top_k} similar Q&A pairs for: '{query}'", flush=True)
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        print(f"📚 Retrieved {len(retrieved)} Q&A pairs", flush=True)

        # Step 2: Generate/Synthesize answer
        print("💭 Generating answer...", flush=True)

        used_llm = False
        answer = None

        if config.use_generation and len(retrieved) > 0:
            try:
                gen_config = GenerationConfig(
                    temperature=float(config.temperature),
                    hallucination_control=bool(config.hallucination_control),
                )
                result = generate_answer(query, retrieved, gen_config)
                answer = result["answer"]
                
                # Check if the answer is substantive
                if answer and len(answer.split()) > 5:
                    used_llm = True
                    print("✅ Answer generated successfully\n", flush=True)
                else:
                    print("⚠️ Generated answer too short, using fallback\n", flush=True)
                    answer = None
                    
            except Exception as e:
                print(f"❌ LLM generation failed: {e}", flush=True)
                print("🔄 Falling back to simple synthesis...\n", flush=True)
                answer = None
        
        # Fallback to simple synthesis if LLM failed or wasn't used
        if answer is None:
            answer = self.synthesize_answer_simple(query, retrieved, config)
            print("✅ Answer synthesized\n", flush=True)

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
    print("🧪 Test the RAG pipeline", flush=True)
    print("=" * 70, flush=True)

    pipeline = RAGPipeline()

    test_queries = [
        "Does this case fit iPhone 13?",
        "What is the battery life?",
        "How long does shipping take?",
    ]

    for q in test_queries:
        print(f"\n{'─' * 70}", flush=True)
        print(f"❓ Query: {q}", flush=True)
        print(f"{'─' * 70}", flush=True)

        result = pipeline.run(q, top_k=3)
        print("\n💬 Answer:", flush=True)
        print(result["answer"], flush=True)

        print("\n📊 Metadata:", flush=True)
        for k, v in result["metadata"].items():
            print(f"   {k}: {v}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("✅ Pipeline test complete!", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    test_pipeline()