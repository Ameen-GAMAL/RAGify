import faiss
import numpy as np
from src.embeddings import EmbeddingModel


class FAISSVectorStore:
    def __init__(self, dimension):
        """
        Initialize FAISS index.
        """
        self.index = faiss.IndexFlatIP(dimension)

    def add_embeddings(self, embeddings):
        """
        Add embeddings to the index.
        """
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=5):
        """
        Search for nearest neighbors.

        Returns:
            distances, indices
        """
        return self.index.search(query_embedding, top_k)
