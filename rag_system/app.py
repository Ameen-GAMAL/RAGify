from __future__ import annotations

import os
from typing import List, Tuple

import streamlit as st

from data_processing import (
    load_dataset,
    create_document_set,
    read_pdf,
    add_documents_to_set,
)
from retrieval import retrieve
from generation import generate_answer


# =================================================
# Self-learning helpers (Query + Feedback Memory)
# =================================================
def build_memory_item(
    query: str,
    initial_answer: str,
    final_answer: str,
    retrieved_contexts: List[str],
    user_rating: int | None = None,
    user_feedback: str | None = None,
) -> str:
    """
    Build a memory document that will be embedded and retrievable.
    """

    ctx_preview = (
        "\n".join([f"- {c[:200].strip()}" for c in retrieved_contexts[:3]])
        if retrieved_contexts
        else "No external documents were used."
    )

    memory = f"""
[MEMORY ITEM]

Query:
{query}

Initial LLM Answer:
{initial_answer}

Final Answer (After RAG Refinement):
{final_answer}

Retrieved Context Preview:
{ctx_preview}
"""

    if user_rating is not None or (user_feedback and user_feedback.strip()):
        memory += f"""

[USER FEEDBACK]
Rating: {user_rating}
Improvement / Correction:
{user_feedback}
"""

    return memory.strip()


def save_memory(doc_set, memory_text: str):
    """
    Save memory to:
    1) Vector database (immediate use)
    2) Disk (persistent learning)
    """
    add_documents_to_set(doc_set, [memory_text])

    with open("memory_store.txt", "a", encoding="utf-8") as f:
        f.write(memory_text + "\n\n---\n\n")


@st.cache_resource
def initialize_knowledge_base(csv_path: str):
    df = load_dataset(csv_path)
    return create_document_set(df)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks


# =================================================
# MAIN APP
# =================================================
def main():
    st.set_page_config(page_title="E-commerce FAQ Assistant", layout="centered")
    st.title("🛍️ E-commerce FAQ Assistant")

    st.markdown(
        """
This assistant **first answers using its own knowledge**, then **refines the answer
using retrieved documents**.  
It continuously **learns from past queries and feedback**.
"""
    )

    # -------------------------------------------------
    # Load dataset(s)
    # -------------------------------------------------
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    candidate_paths = [
        os.path.join(base_dir, "Ecommerce_FAQs.csv"),
        os.path.join(base_dir, "multi_questions (Repaired).csv"),
        os.path.join(base_dir, "..", "Ecommerce_FAQs.csv"),
        "/home/oai/share/Ecommerce_FAQs.csv",
    ]

    csv_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            csv_path = p
            break

    if csv_path is None:
        st.error("Dataset not found. Put your CSV in the project root.")
        return

    doc_set = initialize_knowledge_base(csv_path)

    # -------------------------------------------------
    # Load persistent memory (if exists)
    # -------------------------------------------------
    memory_path = "memory_store.txt"
    if os.path.exists(memory_path):
        with open(memory_path, "r", encoding="utf-8") as f:
            raw_memories = f.read().split("\n\n---\n\n")

        memories = [m.strip() for m in raw_memories if m.strip()]
        if memories:
            add_documents_to_set(doc_set, memories)
            st.info(f"Loaded {len(memories)} memory items from previous sessions.")

    # -------------------------------------------------
    # PDF upload (dynamic ingestion)
    # -------------------------------------------------
    st.header("📄 Upload additional documents")
    uploaded = st.file_uploader(
        "Add a PDF to augment the knowledge base", type=["pdf"]
    )

    if uploaded is not None:
        temp_path = "uploaded.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        extracted = read_pdf(temp_path)
        if extracted:
            chunks = chunk_text(extracted)
            add_documents_to_set(doc_set, chunks)
            st.success(f"Added {len(chunks)} passages from uploaded PDF.")
        else:
            st.warning("No text could be extracted from the PDF.")

        os.remove(temp_path)

    # -------------------------------------------------
    # Query UI
    # -------------------------------------------------
    st.header("🔍 Ask a question")
    query = st.text_input("Enter your question here:")
    top_k = st.slider("Number of passages to retrieve", 1, 8, 5)

    # Session state
    if "last_initial_answer" not in st.session_state:
        st.session_state.last_initial_answer = None
    if "last_final_answer" not in st.session_state:
        st.session_state.last_final_answer = None
    if "last_results" not in st.session_state:
        st.session_state.last_results = []
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []

    if st.button("Submit") and query:
        with st.spinner("Generating answer and refining with knowledge base..."):

            # -------------------------------------------------
            # STEP 1: LLM-FIRST (NO DOCUMENTS)
            # -------------------------------------------------
            initial_answer = generate_answer(query, retrieved_docs=[])

            # -------------------------------------------------
            # STEP 2: RETRIEVAL
            # -------------------------------------------------
            results: List[Tuple[str, float]] = retrieve(
                query, doc_set, top_k=top_k
            )
            contexts = [doc for doc, _ in results]

            # -------------------------------------------------
            # STEP 3: RAG REFINEMENT
            # -------------------------------------------------
            refinement_prompt = (
                "Improve and enrich the following answer using the additional context "
                "ONLY if it adds useful or domain-specific information.\n\n"
                f"INITIAL ANSWER:\n{initial_answer}\n\n"
                f"ADDITIONAL CONTEXT:\n{chr(10).join(contexts)}"
            )

            final_answer = generate_answer(refinement_prompt, contexts)

            # Save to session
            st.session_state.last_initial_answer = initial_answer
            st.session_state.last_final_answer = final_answer
            st.session_state.last_results = results
            st.session_state.last_contexts = contexts

            # -------------------------------------------------
            # SAVE QUERY MEMORY
            # -------------------------------------------------
            memory_text = build_memory_item(
                query=query,
                initial_answer=initial_answer,
                final_answer=final_answer,
                retrieved_contexts=contexts,
            )
            save_memory(doc_set, memory_text)

        # -------------------------------------------------
        # Display output
        # -------------------------------------------------
        st.subheader("Final Answer")
        st.write(st.session_state.last_final_answer)

        st.subheader("Retrieved Passages (for refinement & learning)")
        if results:
            for i, (doc, score) in enumerate(results, 1):
                if "[MEMORY ITEM]" in doc:
                    title = f"🧠 Memory Item {i} (similarity {score:.3f})"
                else:
                    title = f"📄 Knowledge Document {i} (similarity {score:.3f})"

                with st.expander(title):
                    st.write(doc)
        else:
            st.info("No passages retrieved.")

    # -------------------------------------------------
    # Feedback UI
    # -------------------------------------------------
    st.subheader("🧠 Self-learning Feedback")

    if st.session_state.last_final_answer is None:
        st.info("Ask a question first, then you can submit feedback.")
        return

    rating = st.slider("How useful was the answer?", 1, 5, 3)
    feedback = st.text_area(
        "Suggest an improved answer / correction / extra context:"
    )

    if st.button("Submit Feedback"):
        with open("user_feedback.log", "a", encoding="utf-8") as f:
            f.write(
                f"QUERY: {query}\nRATING: {rating}\nFEEDBACK: {feedback}\n\n"
            )

        merged_memory = build_memory_item(
            query=query,
            initial_answer=st.session_state.last_initial_answer,
            final_answer=st.session_state.last_final_answer,
            retrieved_contexts=st.session_state.last_contexts,
            user_rating=rating,
            user_feedback=feedback,
        )

        save_memory(doc_set, merged_memory)

        st.success("Feedback saved ✅ and added to memory for future retrieval.")


if __name__ == "__main__":
    main()
