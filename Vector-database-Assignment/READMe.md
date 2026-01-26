# E-commerce Q&A RAG System (Streamlit + FAISS + Flan-T5)

A Retrieval-Augmented Generation (RAG) application that answers e-commerce/product questions by:
- retrieving relevant historical Q&A pairs from a **FAISS** vector index, then
- generating a grounded response using a **Hugging Face Flan-T5** model,
- with a modern **Streamlit** UI and optional self-learning boosts from user feedback.


## Project Overview

This project implements a standard RAG workflow:

1) **Embedding + Retrieval**
- A query is embedded using `sentence-transformers/all-mpnet-base-v2`
- A FAISS index is searched for the most similar Q&A entries

2) **Generation**
- Retrieved Q&A pairs are provided to the generator
- The generator (Flan-T5) produces a final response with citations like: `[Reference 1]`

3) **UI**
- Streamlit provides a search UI + settings (top-k, temperature, hallucination control)
- Retrieved sources can be displayed alongside the final answer

---

## Main File to Run

✅ **Streamlit UI entry point:**
- `pip install -r requirements.txt`
- `streamlit run src/app/ui_streamlit.py`


---


