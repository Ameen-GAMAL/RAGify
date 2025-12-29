import streamlit as st

from src.retrieval import Retriever
from src.generation import RAGGenerator


# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Advanced DB RAG Assistant",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Advanced Database Systems – RAG Assistant")
st.markdown(
    "Ask questions based **only** on your course lectures "
    "(retrieval-augmented, no hallucination)."
)

# ---------------------------
# Initialize RAG Components
# ---------------------------
@st.cache_resource
def load_rag():
    retriever = Retriever()
    generator = RAGGenerator()
    return retriever, generator

retriever, generator = load_rag()

# ---------------------------
# User Input
# ---------------------------
query = st.text_input(
    "🔎 Enter your question:",
    placeholder="e.g. Explain the ARIES recovery algorithm"
)

top_k = st.slider(
    "Number of retrieved chunks",
    min_value=1,
    max_value=5,
    value=3
)

# ---------------------------
# Run RAG Pipeline
# ---------------------------
if st.button("Generate Answer") and query:

    with st.spinner("Retrieving relevant lecture content..."):
        retrieved_chunks = retriever.retrieve(query, top_k=top_k)

    with st.spinner("Generating answer..."):
        answer = generator.generate_answer(query, retrieved_chunks)

    # ---------------------------
    # Display Answer
    # ---------------------------
    st.subheader("🧠 Final Answer")
    st.write(answer)

    # ---------------------------
    # Display Retrieved Chunks
    # ---------------------------
    st.subheader("📚 Retrieved Lecture Chunks")

    for i, chunk in enumerate(retrieved_chunks, 1):
        with st.expander(f"Chunk {i} — {chunk['lecture_id']}"):
            st.write(chunk["text"])
