

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import numpy as np


@dataclass
class DocumentSet:
    """Simple container for documents and their associated FAISS index.

    Attributes
    ----------
    docs : List[str]
        The list of document texts stored in the vector database.
    embeddings : np.ndarray
        2D array of shape (n_documents, embedding_dim) with the
        normalized dense embeddings for each document.
    index : faiss.Index
        FAISS index for inner‑product search.  The index is populated
        with the embeddings stored in this object.
    model : sentence_transformers.SentenceTransformer
        The embedding model used to generate the embeddings.  Stored
        here so that query embeddings can be computed consistently.
    """

    docs: List[str]
    embeddings: np.ndarray
    index: object
    model: object


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the FAQ dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the FAQ data.  The file is
        expected to have at least two columns named ``prompt`` and
        ``response``.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with the contents of the CSV file.
    """
    df = pd.read_csv(csv_path)
    if 'prompt' not in df.columns or 'response' not in df.columns:
        raise ValueError("Dataset must contain 'prompt' and 'response' columns.")
    return df


def combine_columns(row: pd.Series) -> str:
    """Combine the prompt and response into a single document string.

    Each row in the FAQ dataset typically contains a question and its
    answer.  This function concatenates them with labels to improve
    clarity for the language model.  Splitting into smaller chunks can
    be performed at a later stage if necessary.

    Parameters
    ----------
    row : pandas.Series
        A row from the DataFrame with 'prompt' and 'response' fields.

    Returns
    -------
    combined : str
        Combined string of the form ``Question: …\nAnswer: …``.
    """
    return f"Question: {row['prompt']}\nAnswer: {row['response']}"


def _ensure_model(model_name: str):
    """Load the embedding model lazily.

    The import of sentence_transformers and the actual loading of the
    model are delayed until this function is called.  This avoids
    import errors when the optional dependency is not installed.

    Parameters
    ----------
    model_name : str
        Name of the pre‑trained sentence transformer model.

    Returns
    -------
    model : sentence_transformers.SentenceTransformer
        Instantiated embedding model.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_texts(texts: List[str], model_name: str = 'all-MiniLM-L6-v2', model=None) -> Tuple[np.ndarray, object]:
    """Embed a list of documents into dense vectors.

    Parameters
    ----------
    texts : list of str
        Documents to embed.
    model_name : str, optional
        Name of the sentence transformer model to use.  Defaults to
        ``all-MiniLM-L6-v2`` which is a compact BERT‑based model
        providing fast inference and good quality embeddings【106982305484935†L247-L262】【106982305484935†L289-L295】.
    model : optional
        A pre‑loaded model instance.  If provided, this overrides
        ``model_name`` and reuses the model to avoid repeated loads.

    Returns
    -------
    embeddings : numpy.ndarray
        2D array of shape (n_texts, embedding_dim) containing
        L2‑normalized embeddings.  Normalization ensures that inner
        product search corresponds to cosine similarity.
    model : sentence_transformers.SentenceTransformer
        The model used for embedding.  Returned so that callers can
        reuse it for subsequent queries.
    """
    if model is None:
        model = _ensure_model(model_name)
    # Generate embeddings and normalize them to unit vectors
    embeddings = model.encode(texts, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1e-9
    embeddings = embeddings / norms
    return embeddings.astype('float32'), model


def build_faiss_index(embeddings: np.ndarray):
    """Construct a FAISS index for inner‑product (cosine) search.

    The index uses ``IndexFlatIP`` which stores all vectors in RAM and
    performs exact nearest‑neighbour search based on inner product.
    Because the embeddings are normalized to unit length, the inner
    product corresponds to cosine similarity.

    Parameters
    ----------
    embeddings : numpy.ndarray
        2D array of shape (n_documents, embedding_dim) containing
        normalized vectors.

    Returns
    -------
    index : faiss.Index
        FAISS index populated with the input embeddings.
    """
    import faiss  # Imported here to delay optional dependency
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def create_document_set(df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2') -> DocumentSet:
    """Create a document set with embeddings and FAISS index from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the FAQ dataset.  Must have 'prompt' and
        'response' columns.
    model_name : str, optional
        Name of the embedding model to use.

    Returns
    -------
    DocumentSet
        A structured object containing the documents, embeddings, FAISS
        index and the embedding model.
    """
    docs = [combine_columns(row) for _, row in df.iterrows()]
    embeddings, model = embed_texts(docs, model_name=model_name)
    index = build_faiss_index(embeddings)
    return DocumentSet(docs=docs, embeddings=embeddings, index=index, model=model)


def read_pdf(file_path: str) -> str:
    """Extract text from a PDF file.

    This helper uses `PyPDF2` to extract plain text from each page of
    the provided PDF.  The function concatenates all page texts into a
    single string.  It is intentionally simple; for more advanced
    requirements you may want to implement chunking and metadata
    extraction separately.

    Parameters
    ----------
    file_path : str
        Path to the PDF file to read.

    Returns
    -------
    text : str
        The concatenated text of all pages in the PDF.
    """
    from PyPDF2 import PdfReader  # Imported lazily
    reader = PdfReader(file_path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or '')
        except Exception:
            # Skip pages that cannot be processed
            continue
    return '\n'.join(parts)


def add_documents_to_set(doc_set: DocumentSet, new_docs: List[str]):
    """Augment an existing DocumentSet with additional documents.

    This function computes embeddings for the new documents using the
    same model contained in the ``doc_set``, normalizes them, and adds
    them to both the internal list of documents and the FAISS index.

    Parameters
    ----------
    doc_set : DocumentSet
        Existing document set created by ``create_document_set``.
    new_docs : list of str
        New documents to add.

    Returns
    -------
    None
        The function updates the ``doc_set`` in place.
    """
    # Compute embeddings using the existing model
    new_embs, _ = embed_texts(new_docs, model=doc_set.model)
    # Append documents and embeddings
    doc_set.docs.extend(new_docs)
    doc_set.embeddings = np.vstack([doc_set.embeddings, new_embs])
    # Add to FAISS index
    doc_set.index.add(new_embs)
