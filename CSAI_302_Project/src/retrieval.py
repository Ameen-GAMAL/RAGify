import json
from src.vector_store import FAISSVectorStore
from src.embeddings import EmbeddingModel

CHUNKS_FILE = "data/chunks/chunks.json"


class Retriever:
    def __init__(self):
        """
        Initializes the retriever.
        - Loads stored chunks
        - Initializes embedding model
        - Builds FAISS index
        """

        # Initialize embedding model internally
        self.embedder = EmbeddingModel()

        # Load chunks from JSON
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Embed all chunk texts
        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.embedder.embed_texts(texts)

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.vector_store = FAISSVectorStore(dimension)
        self.vector_store.add_embeddings(embeddings)

    def retrieve(self, query: str, top_k: int = 3):
        """
        Retrieve top-k most relevant chunks for a query.
        """
        query_embedding = self.embedder.embed_texts([query])
        distances, indices = self.vector_store.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results
