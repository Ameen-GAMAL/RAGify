"""Document retrieval module for the RAG system.

This module encapsulates the logic for performing semantic search over
the vector database.  It relies on the ``DocumentSet`` structure
defined in ``data_processing.py``.  Given a user query, the module
computes a dense embedding with the same model used for indexing,
performs nearest‑neighbour search using the FAISS index, and returns
the most relevant documents along with their similarity scores.

Inner‑product search is used because the embeddings are normalized to
unit length, making the inner product equivalent to cosine similarity.
Cosine similarity is often preferred for text retrieval because it
focuses on the orientation of the vectors rather than their
magnitude【76339000196768†L161-L165】.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from data_processing import DocumentSet


def retrieve(query: str, doc_set: DocumentSet, top_k: int = 3) -> List[Tuple[str, float]]:
    """Retrieve the top‑``k`` most similar documents for a query.

    Parameters
    ----------
    query : str
        The user’s query.
    doc_set : DocumentSet
        The document set containing the FAISS index and embedding model.
    top_k : int, optional
        The number of documents to return.  Defaults to 3.

    Returns
    -------
    results : list of (str, float)
        A list of tuples where each tuple contains a document text and
        its similarity score.  Scores are sorted in descending order.
    """
    # Compute the embedding for the query using the same model and
    # normalize it.  We use the model stored on the DocumentSet to
    # ensure consistency with indexed embeddings.
    q_emb = doc_set.model.encode([query], convert_to_numpy=True)
    q_norm = np.linalg.norm(q_emb)
    if q_norm == 0:
        q_norm = 1e-9
    q_emb = (q_emb / q_norm).astype('float32')

    # Perform the search.  FAISS returns distances (inner products)
    # and indices.  Higher values indicate greater similarity.
    scores, indices = doc_set.index.search(q_emb, top_k)
    # Flatten the arrays and pair with documents
    result = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(doc_set.docs):
            result.append((doc_set.docs[idx], float(score)))
    # Sort results by score descending just in case
    result.sort(key=lambda x: x[1], reverse=True)
    return result
