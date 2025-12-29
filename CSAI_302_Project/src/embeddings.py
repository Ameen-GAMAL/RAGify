from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the embedding model.
        """
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """
        Generate embeddings for a list of texts.

        Args:
            texts (list[str]): List of text chunks

        Returns:
            np.ndarray: Array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
