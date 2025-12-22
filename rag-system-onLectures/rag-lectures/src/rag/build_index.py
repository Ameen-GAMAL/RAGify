from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from tqdm import tqdm

from src.rag.embeddings import Embedder, EmbeddingConfig
from src.rag.faiss_store import build_faiss_index, save_index, IndexPaths


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
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
) -> None:
    chunks_p = Path(chunks_path)
    if not chunks_p.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_p.resolve()}")

    chunks = read_jsonl(chunks_p)
    if not chunks:
        raise ValueError("chunks.jsonl is empty")

    texts = [c["text"] for c in chunks]

    # Use Flan-T5 for embedding generation
    cfg = EmbeddingConfig(model_name="google/flan-t5-base")  # Ensure the correct Flan-T5 model is used
    embedder = Embedder(cfg)
    
    # Generate embeddings using Flan-T5
    vectors = embedder.embed_texts(texts)

    # Build the FAISS index with the generated embeddings
    index = build_faiss_index(vectors)

    # Save metadata with embedding information
    meta = {
        "embedding_model": cfg.model_name,
        "normalize_embeddings": cfg.normalize,
        "dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "metadata": c.get("metadata", {}),
            }
            for c in chunks
        ],
    }

    # Define paths for saving FAISS index and metadata
    paths = IndexPaths(Path(out_faiss), Path(out_meta))
    save_index(index, meta, paths)

    print(f"Saved FAISS index -> {paths.faiss_path.resolve()}")
    print(f"Saved metadata    -> {paths.meta_path.resolve()}")
    print(f"Indexed chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
