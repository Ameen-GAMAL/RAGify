from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "google/flan-t5-base"  # Using Flan-T5 model
    batch_size: int = 1  # Set to 1 for long document embedding
    normalize: bool = True  # cosine similarity via inner product on unit vectors


class Embedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        api_token = os.getenv("HF_TOKEN")
        
        # Load Flan-T5 model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(cfg.model_name, use_auth_token=api_token)
        self.model = T5ForConditionalGeneration.from_pretrained(cfg.model_name, use_auth_token=api_token)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            # Tokenize the text with Flan-T5 tokenizer
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=4096)
            
            # Generate embeddings from Flan-T5
            with torch.no_grad():
                outputs = self.model.encoder(**inputs)
            
            # Get the embeddings from the model's last hidden state
            last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            # Use the mean of the last hidden state's embeddings as the sentence embedding
            sentence_embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(sentence_embedding)

        # Convert the list of embeddings into a numpy array
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        # Embed a single query
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=4096)
        
        with torch.no_grad():
            outputs = self.model.encoder(**inputs)
        
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        query_embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return query_embedding  # shape (1, d)
