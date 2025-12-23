from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.rag.embeddings import Embedder, EmbeddingConfig
from src.rag.faiss_store import IndexPaths, load_index, search
from src.learn.feedback_store import FeedbackPaths, get_boosts


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class Retriever:
    def __init__(
        self,
        index_paths: IndexPaths = IndexPaths(),
        emb_cfg: EmbeddingConfig = EmbeddingConfig(),
        feedback_paths: FeedbackPaths = FeedbackPaths(),
    ):
        print(" Initializing Retriever...", flush=True)

        # Load FAISS index and metadata
        self.index, self.meta = load_index(index_paths)
        print(f"✓ Loaded FAISS index: {self.index.ntotal} vectors, dim={self.index.d}", flush=True)

        # Initialize embedder
        self.embedder = Embedder(emb_cfg)
        print(f"✓ Loaded embedder: {emb_cfg.model_name} (dim={self.embedder.dim})", flush=True)

        # Verify dimension match
        if self.embedder.dim != self.index.d:
            raise ValueError(
                f"  Dimension mismatch!\n"
                f"   Embedder: {self.embedder.dim}\n"
                f"   FAISS Index: {self.index.d}\n"
                f"   → Rebuild index with: python src/rag/build_index.py"
            )

        self.feedback_paths = feedback_paths
        print("✓ Retriever ready!\n", flush=True)

    def retrieve(self, query: str, top_k: int = 6) -> List[RetrievedChunk]:
        """
        Retrieve top-k most relevant chunks for a query.
        """
        # Embed query
        qv = self.embedder.embed_query(query)

        # Ensure 2D for FAISS
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)

        # Dimension check
        if qv.shape[1] != self.index.d:
            raise ValueError(f"Query embedding dim {qv.shape[1]} != index dim {self.index.d}")

        # Search FAISS
        scores, ids = search(self.index, qv, top_k)

        # Load feedback boosts
        boosts = get_boosts(self.feedback_paths)

        results: List[RetrievedChunk] = []
        chunk_list = self.meta["chunks"]

        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx < 0 or idx >= len(chunk_list):
                continue

            chunk = chunk_list[idx]
            chunk_id = chunk["chunk_id"]

            boost = boosts.get(chunk_id, 0.0)
            boosted_score = float(score + boost)

            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    score=boosted_score,
                    text=chunk["text"],
                    metadata=chunk.get("metadata", {}),
                )
            )

        # Sort by boosted score (descending)
        results.sort(key=lambda r: r.score, reverse=True)
        return results
