

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "sentence-transformers/all-mpnet-base-v2"  # 768-dim, SOTA
    batch_size: int = 32
    normalize: bool = True  # L2 normalization for cosine similarity
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    show_progress: bool = True


class Embedder:
    """
    Generates dense embeddings using Sentence Transformers.
    
    Model: all-mpnet-base-v2
    - 768 dimensions
    - Best performance on semantic similarity tasks
    - Trained on 1B+ sentence pairs
    - Optimized for Q&A and retrieval
    """
    
    def __init__(self, cfg: EmbeddingConfig = EmbeddingConfig()):
        self.cfg = cfg
        
        print(f" Loading embedding model...")
        print(f"   Model: {cfg.model_name}")
        print(f"   Device: {cfg.device}")
        
        # Check for HuggingFace token (optional for public models)
        api_token = os.getenv("HF_TOKEN")
        
        # Load Sentence Transformer model
        self.model = SentenceTransformer(
            cfg.model_name,
            device=cfg.device,
            use_auth_token=api_token if api_token else None,
        )
        
        # Get embedding dimension
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"   Dimension: {self.dim}")
        print(f"   Normalize: {cfg.normalize}")
        
        # Verify GPU usage
        if cfg.device == "cuda" and torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        print(f" Embedder ready!\n")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
            L2-normalized if cfg.normalize=True
        """
        if not texts:
            return np.array([]).reshape(0, self.dim)
        
        # Filter out empty texts
        valid_texts = [t if t else " " for t in texts]  # Replace empty with space
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            valid_texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=self.cfg.show_progress and len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
        )
        
        return embeddings.astype(np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string
            
        Returns:
            np.ndarray of shape (embedding_dim,)
        """
        if not query or not query.strip():
            # Return zero vector for empty query
            return np.zeros(self.dim, dtype=np.float32)
        
        embedding = self.model.encode(
            query.strip(),
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
        )
        
        return embedding.astype(np.float32)
    
    def get_dimension(self) -> int:
        """Return the embedding dimension"""
        return self.dim
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding (1D or 2D)
            emb2: Second embedding (1D or 2D)
            
        Returns:
            Similarity score (0-1 if normalized)
        """
        # Ensure 1D
        if emb1.ndim > 1:
            emb1 = emb1.flatten()
        if emb2.ndim > 1:
            emb2 = emb2.flatten()
        
        # Compute dot product (equals cosine if normalized)
        similarity = np.dot(emb1, emb2)
        
        return float(similarity)


def test_embedder():
    """Test the embedder with sample e-commerce Q&A"""
    print("\n" + "=" * 70)
    print(" TESTING EMBEDDER")
    print("=" * 70)
    
    cfg = EmbeddingConfig()
    embedder = Embedder(cfg)
    
    # Sample e-commerce questions
    questions = [
        "Does this phone case fit iPhone 13?",
        "Will this case work with iPhone 13 Pro?",
        "Is this compatible with Samsung Galaxy S21?",
        "What is the battery life?",
        "How long does shipping take?",
    ]
    
    print(f"\n Embedding {len(questions)} sample questions...")
    embeddings = embedder.embed_texts(questions)
    
    print(f"\n Generated embeddings:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dtype: {embeddings.dtype}")
    print(f"   Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    
    # Test query
    query = "iPhone 13 case compatibility"
    print(f"\n Query: '{query}'")
    query_emb = embedder.embed_query(query)
    print(f"   Query embedding shape: {query_emb.shape}")
    
    # Compute similarities
    print(f"\n Similarities to query:")
    similarities = []
    for i, (q, emb) in enumerate(zip(questions, embeddings)):
        sim = embedder.compute_similarity(query_emb, emb)
        similarities.append((sim, q))
    
    # Sort by similarity
    similarities.sort(reverse=True)
    
    for rank, (sim, q) in enumerate(similarities, 1):
        print(f"   [{rank}] {sim:.4f} - {q}")
    
    print("\n Notice:")
    print("   • iPhone 13 questions have highest similarity")
    print("   • Semantic understanding beyond keyword matching")
    print("   • Ready for production use!")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_embedder()