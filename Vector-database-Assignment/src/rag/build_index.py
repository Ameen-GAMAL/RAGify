"""
Build FAISS index from Q&A chunks using MPNet embeddings
Optimized for 14K+ Q&A pairs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from tqdm import tqdm

from src.rag.embeddings import Embedder, EmbeddingConfig
from src.rag.faiss_store import build_faiss_index, save_index, IndexPaths


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file"""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(
    chunks_path: str = "data/processed/chunks.jsonl",
    out_faiss: str = "data/processed/index.faiss",
    out_meta: str = "data/processed/index_meta.json",
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 64,  # Larger batches for 14K records
) -> None:
    """
    Build FAISS index from Q&A chunks.
    
    Args:
        chunks_path: Path to chunks.jsonl file
        out_faiss: Output path for FAISS index
        out_meta: Output path for metadata JSON
        model_name: Embedding model to use
        batch_size: Batch size for embedding generation
    """
    print("\n" + "=" * 70)
    print("  BUILDING FAISS INDEX FOR Q&A RETRIEVAL")
    print("=" * 70)
    
    chunks_p = Path(chunks_path)
    if not chunks_p.exists():
        raise FileNotFoundError(f" Missing chunks file: {chunks_p.resolve()}")

    # Load chunks
    print(f"\n Loading chunks...")
    chunks = read_jsonl(chunks_p)
    if not chunks:
        raise ValueError(" chunks.jsonl is empty")
    
    print(f" Loaded {len(chunks):,} Q&A chunks")

    # Extract texts for embedding
    texts = [c["text"] for c in chunks]
    
    # Statistics
    avg_len = sum(len(t) for t in texts) / len(texts)
    min_len = min(len(t) for t in texts)
    max_len = max(len(t) for t in texts)
    
    print(f"\n Text Statistics:")
    print(f"   Average length: {avg_len:.0f} chars")
    print(f"   Min length:     {min_len:,} chars")
    print(f"   Max length:     {max_len:,} chars")

    # Initialize embedder with MPNet
    print(f"\n Initializing embedder...")
    cfg = EmbeddingConfig(
        model_name=model_name,
        batch_size=batch_size,
        normalize=True,  # Critical for cosine similarity
    )
    embedder = Embedder(cfg)
    
    # Generate embeddings
    print(f"\n Generating embeddings...")
    print(f"   Model: {model_name}")
    print(f"   Batch size: {batch_size}")
    print(f"   Total chunks: {len(texts):,}")
    
    vectors = embedder.embed_texts(texts)
    
    print(f"\n Embedding generation complete!")
    print(f"   Shape: {vectors.shape}")
    print(f"   Dimension: {vectors.shape[1]}")
    print(f"   Count: {vectors.shape[0]:,}")
    print(f"   Dtype: {vectors.dtype}")
    print(f"   Normalized: {cfg.normalize}")

    # Build the FAISS index
    print(f"\n Building FAISS index...")
    index = build_faiss_index(vectors)
    print(f" FAISS index created!")
    print(f"   Type: IndexFlatIP (Inner Product)")
    print(f"   Vectors: {index.ntotal:,}")
    print(f"   Dimension: {index.d}")

    # Create metadata
    print(f"\n Creating metadata...")
    meta = {
        "model_info": {
            "name": cfg.model_name,
            "dimension": int(vectors.shape[1]),
            "normalize": cfg.normalize,
        },
        "index_info": {
            "type": "IndexFlatIP",
            "total_vectors": int(vectors.shape[0]),
            "dimension": int(index.d),
        },
        "dataset_info": {
            "total_chunks": len(chunks),
            "avg_text_length": round(avg_len, 2),
            "min_text_length": min_len,
            "max_text_length": max_len,
        },
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "metadata": c.get("metadata", {}),
            }
            for c in chunks
        ],
    }

    # Save index and metadata
    print(f"\n Saving index...")
    paths = IndexPaths(Path(out_faiss), Path(out_meta))
    save_index(index, meta, paths)

    # Final summary
    print("\n" + "=" * 70)
    print(" INDEX BUILD COMPLETE!")
    print("=" * 70)
    
    print(f"\n Output Files:")
    print(f"   FAISS Index: {paths.faiss_path.resolve()}")
    print(f"   Metadata:    {paths.meta_path.resolve()}")
    
    print(f"\n Index Summary:")
    print(f"   ├─ Model:        {model_name}")
    print(f"   ├─ Dimension:    {vectors.shape[1]}")
    print(f"   ├─ Vectors:      {index.ntotal:,}")
    print(f"   ├─ Normalized:   {cfg.normalize}")
    print(f"   └─ Index Type:   IndexFlatIP")
    
    print(f"\n Next Steps:")
    print(f"     Test retrieval:")
    print(f"       python -m src.rag.retriever")
    print(f"     Launch API:")
    print(f"       uvicorn src.rag.api:app --reload")
    print(f"     Launch UI:")
    print(f"       streamlit run src/app/ui_streamlit.py")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    # Run with MPNet model
    main(
        model_name="sentence-transformers/all-mpnet-base-v2",  # Best quality
        batch_size=64,  # Optimize for 14K records
    )