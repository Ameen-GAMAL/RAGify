from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import faiss
import numpy as np


@dataclass(frozen=True)
class IndexPaths:
    faiss_path: Path = Path("data/processed/index.faiss")
    meta_path: Path = Path("data/processed/index_meta.json")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    We use IndexFlatIP (inner product). If embeddings are L2-normalized,
    inner product equals cosine similarity.
    """
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape={vectors.shape}")
    d = vectors.shape[1]  # This is the dimension of the embeddings
    index = faiss.IndexFlatIP(d)  # Use the same dimension for the FAISS index
    index.add(vectors)
    return index


def save_index(index: faiss.Index, meta: Dict[str, Any], paths: IndexPaths) -> None:
    ensure_parent(paths.faiss_path)
    ensure_parent(paths.meta_path)
    faiss.write_index(index, str(paths.faiss_path))
    paths.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(paths: IndexPaths) -> Tuple[faiss.Index, Dict[str, Any]]:
    if not paths.faiss_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {paths.faiss_path}")
    if not paths.meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {paths.meta_path}")
    index = faiss.read_index(str(paths.faiss_path))
    meta = json.loads(paths.meta_path.read_text(encoding="utf-8"))
    return index, meta


def search(index: faiss.Index, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (scores, ids). scores shape (1, k), ids shape (1, k)
    """
    if query_vec.dtype != np.float32:
        query_vec = query_vec.astype(np.float32)
    scores, ids = index.search(query_vec, top_k)
    return scores, ids


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit vectors (L2-normalization).
    This is essential for cosine similarity-based methods in FAISS.
    """
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norm
    return normalized_embeddings
