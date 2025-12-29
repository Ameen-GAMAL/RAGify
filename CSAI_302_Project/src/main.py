from embeddings import EmbeddingModel
from src.retrieval import Retriever
from src.generation import RAGGenerator


def main():
    query = "Explain the ARIES recovery protocol"

    # STEP 3: Retrieval
    embedder = EmbeddingModel()
    retriever = Retriever(embedder)
    retrieved_chunks = retriever.search(query, top_k=3)

    # STEP 4: Generation (API key is read internally)
    generator = RAGGenerator()
    answer = generator.generate_answer(query, retrieved_chunks)

    print("\nFinal Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
