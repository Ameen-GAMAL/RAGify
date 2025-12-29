# 📚 Retrieval-Augmented Generation (RAG) System using Vector Databases

## 📌 Project Overview

This project implements a **complete end-to-end Retrieval-Augmented Generation (RAG) system** as part of the **CSAI 302 – Vector Database Assignment**.  
The system is designed to retrieve semantically relevant lecture content using dense vector embeddings and generate grounded, context-aware answers using a Large Language Model (LLM).

The project strictly follows the assignment requirements and includes:
- A **vector database layer**
- A **semantic retrieval mechanism**
- A **generation module grounded in retrieved documents**
- A **user-friendly querying interface (UI)**

---

## 🧠 What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is a hybrid architecture that combines:
- **Information Retrieval (IR)** using vector similarity search
- **Text Generation** using large language models

### High-level RAG Flow:
```text
User Query
   ↓
Query Embedding
   ↓
Vector Similarity Search (FAISS)
   ↓
Top-k Relevant Chunks
   ↓
Context Injection into Prompt
   ↓
LLM Answer Generation
```


## 🏗️ System Architecture
Architecture Diagram (Conceptual)
┌───────────────────┐
│   Lecture Data    │
│  (JSON Files)     │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Text Chunking    │
│ (Overlapping)     │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Embedding Model  │
│ (SBERT MPNet)     │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Vector Database   │
│     (FAISS)       │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Semantic Search  │
│   (Top-k Chunks)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Prompt Assembly   │
│ + Retrieved Docs  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ LLM Generation    │
│ (HF Router API)   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Final Answer    │
│ + Retrieved Docs  │
└───────────────────┘


## 📁 Project Structure


CSAI_302_Project/
│
├── app.py                      # Streamlit UI (main entry point)
│
├── src/
│   ├── embeddings.py           # Embedding model logic
│   ├── vector_store.py         # FAISS vector database
│   ├── retrieval.py            # Semantic retrieval logic
│   ├── generation.py           # LLM generation module
│   ├── chunking.py             # Text chunking with overlap
│   ├── load_data.py            # Lecture data loader
│   └── main.py                 # CLI runner (optional)
│
├── data/
│   ├── lectures/
│   │   ├── lecture_01.json
│   │   ├── lecture_02.json
│   │   └── ...
│   └── chunks/
│       └── chunks.json
│
├── requirements.txt
└── README.md


## 📊 Dataset Description
Dataset: Advaned Database Lectures

Lecture Data

Format: JSON

Fields:

id: Lecture identifier

title: Lecture title

text: Full lecture content

Chunked Data

Stored in: data/chunks/chunks.json

Each chunk contains:

chunk_id

lecture_id

text

📌 Total chunks: 41
📌 Total lectures: 9


### ✂️ Chunking Strategy
Why Chunking?

Large documents cannot be embedded effectively as a single unit. Chunking:

Preserves semantic coherence

Improves retrieval accuracy

Prevents context truncation

Implementation Details

Chunk size: 400 words

Overlap: 80 words

Type: Word-based sliding window

Overlapping chunks ensure that important information near chunk boundaries is preserved.

### 🧬 Embedding Model
Model Used

Sentence-Transformers: all-mpnet-base-v2

Reasons for Selection

High performance on semantic similarity tasks

Produces dense vector representations

Well-documented and widely adopted

Embedding Properties

Output dimension: 768

Embeddings are L2-normalized


### 🗄️ Vector Database
Library

FAISS (Facebook AI Similarity Search)

Index Type

IndexFlatIP (Inner Product)

Similarity Metric

Because embeddings are normalized:

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

Thus, inner product search behaves as cosine similarity search.


### 🔍 Retrieval Mechanism
Steps

Embed all document chunks

Add embeddings to FAISS index

Embed user query

Retrieve top-k most similar chunks

Output

Retrieved chunk texts

(Optional improvement: similarity scores)

distances, indices = index.search(query_embedding, k)





### 🤖 Generation Module
API Used
Hugging Face OpenAI-Compatible Router

Endpoint:

text
Copy code
https://router.huggingface.co/v1
Authentication
Environment variable:

bash
Copy code
export HF_TOKEN=your_huggingface_token
Prompt Design
Injects retrieved chunks as context

Explicit grounding instruction:

“Answer using ONLY the provided context. If the answer is not present, say you do not know.”

This minimizes hallucinations and ensures factual grounding.



### 🖥️ User Interface (Bonus Feature)
Framework

Streamlit

Features

Query input box

Top-k retrieval slider

Displays:

Generated answer

Retrieved document chunks

Run UI:

streamlit run app.py



### 🧪 Demonstration Example
Query
What is backpropagation?

Retrieved Chunks

Chunk from Lecture 03 (Neural Networks)

Chunk from Lecture 04 (Training Algorithms)

Generated Answer

Backpropagation is an algorithm used to train neural networks by computing gradients of the loss function with respect to weights using the chain rule...

Retrieval Explanation

The query embedding was closest to chunks discussing neural network training due to shared semantic concepts such as gradients, loss, and optimization.



### 🚀 Future Improvements

Persist FAISS index to disk

Display similarity scores

Add feedback-based re-ranking

Implement self-learning memory

Add multi-document citation tracking
