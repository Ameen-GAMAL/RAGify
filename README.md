# Retrieval-Augmented Generation System with Vector Databases (RAGify)

## 📌 Project Overview

This repository implements a **complete, modular, and end-to-end Retrieval-Augmented Generation (RAG) system** built around **dense vector databases** and **semantic retrieval**.  
The system is designed to retrieve relevant knowledge from a vector index and generate **context-grounded responses** using a Large Language Model (LLM).

The project follows modern RAG system design principles and emphasizes:
- Vector-based semantic search
- Modular pipeline architecture
- Explainable retrieval
- Clean software engineering practices
- Reproducibility and extensibility

This repository represents the **final implementation** of the project.

---

## 🧠 Retrieval-Augmented Generation (RAG)

### Definition

Retrieval-Augmented Generation (RAG) is a hybrid AI architecture that combines:

- **Dense Information Retrieval** using vector similarity search
- **Natural Language Generation** using large language models

Instead of relying solely on a model’s parametric memory, RAG dynamically retrieves external documents and conditions generation on them.

---

### High-Level Pipeline

```text
User Query
   ↓
Query Embedding
   ↓
Vector Similarity Search (FAISS)
   ↓
Top-k Relevant Documents
   ↓
Context Construction
   ↓
Prompt Injection
   ↓
LLM Generation
   ↓
Final Grounded Answer


## 🏗️ System Architecture

### Architecture Diagram (Conceptual)

```text
Lecture Data (JSON Files)
        │
        ▼
Text Chunking (Overlapping)
        │
        ▼
Embedding Model (SBERT MPNet)
        │
        ▼
Vector Database (FAISS)
        │
        ▼
Semantic Search (Top-k Chunks)
        │
        ▼
Retrieved Documents
        │
        ▼
Prompt Assembly + Retrieved Docs
        │
        ▼
LLM Generation (HF Router API)
        │
        ▼
Final Answer + Retrieved Docs


## 📁 Repository Structure

RAGify/
│
├── app.py                     # Optional UI entry point
│
├── ragify/
│   ├── __init__.py
│   ├── embeddings.py          # Embedding model abstraction
│   ├── vector_store.py        # FAISS index management
│   ├── retriever.py           # Semantic retrieval logic
│   ├── generator.py           # LLM interaction layer
│   ├── pipeline.py            # End-to-end RAG pipeline
│   ├── chunking.py            # Text chunking utilities
│   └── loader.py              # Data loading utilities
│
├── data/
│   ├── raw/                   # Raw input documents
│   └── processed/             # Chunked / preprocessed data
│
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```


## 📊 Data Handling
Input Data

Plain text or JSON documents

Each document may contain metadata (IDs, titles, sources)

Chunked Data

Documents are split into overlapping chunks prior to embedding.


| Field       | Description             |
| ----------- | ----------------------- |
| `chunk_id`  | Unique chunk identifier |
| `source_id` | Original document ID    |
| `text`      | Chunk content           |


### ✂️ Chunking Strategy
Motivation

Chunking is necessary because:

Embedding models have input length limits

Long documents dilute semantic relevance

Overlapping preserves boundary semantics

Parameters

| Parameter  | Value          |
| ---------- | -------------- |
| Chunk Size | 400 words      |
| Overlap    | 80 words       |
| Strategy   | Sliding window |


### 🧬 Embedding Layer
Model

Sentence-Transformers (all-mpnet-base-v2)

Properties

| Property         | Value |
| ---------------- | ----- |
| Vector Dimension | 768   |
| Embedding Type   | Dense |
| Normalization    | L2    |



### 🗄️ Vector Database
Engine

FAISS (Facebook AI Similarity Search)

Index Type

IndexFlatIP

Similarity Function

Given normalized embeddings:

cosine similarity
(
𝑥
,
𝑦
)
=
𝑥
⋅
𝑦
cosine similarity(x,y)=x⋅y

Thus inner product search is equivalent to cosine similarity search.






### 🔍 Retrieval Module
Retrieval Steps

Embed user query

Perform FAISS similarity search

Select top-k chunks

Return texts and similarity scores

scores, indices = index.search(query_embedding, k)

Output

Retrieved documents

Similarity scores

Metadata (IDs, sources)




### 🤖 Generation Module
LLM Interface

API-based Large Language Model

Prompt Construction

Retrieved documents are injected into a structured prompt.

Context:
<retrieved documents>

Instruction:
Answer using ONLY the provided context.
If the answer is not contained in the context, say you do not know.


This enforces grounded generation and minimizes hallucination.





## 🔁 End-to-End Pipeline
query → embed → retrieve → assemble context → generate answer


The pipeline is orchestrated in a single modular interface for clarity and extensibility.




## 🚀 Future Improvements

Persistent FAISS index

Multi-stage retrieval (BM25 + dense)

Re-ranking with cross-encoders

Feedback-driven self-learning

Source citation per answer
