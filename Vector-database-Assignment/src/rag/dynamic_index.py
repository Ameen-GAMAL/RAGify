"""
Dynamic In-Memory Index Builder
Allows users to input custom Q&A pairs and build a temporary searchable index
"""

from __future__ import annotations

import json
from typing import List, Dict, Any
from dataclasses import dataclass

import numpy as np
import faiss

from src.rag.embeddings import Embedder, EmbeddingConfig


@dataclass
class CustomQAPair:
    """Structure for user-provided Q&A pairs"""
    question: str
    answer: str
    metadata: Dict[str, Any] = None


class DynamicIndexBuilder:
    """
    Builds a temporary FAISS index from user-provided Q&A pairs.
    No file system operations - everything in memory.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the dynamic index builder
        
        Args:
            model_name: Embedding model to use
        """
        print(f"🔧 Initializing Dynamic Index Builder", flush=True)
        
        # Initialize embedder
        cfg = EmbeddingConfig(
            model_name=model_name,
            batch_size=32,
            normalize=True,
        )
        self.embedder = Embedder(cfg)
        
        # Storage for current session
        self.qa_pairs: List[CustomQAPair] = []
        self.index: faiss.Index = None
        self.chunks: List[Dict[str, Any]] = []
        
        print(f"✅ Dynamic Index Builder ready\n", flush=True)
    
    def add_qa_pairs(self, qa_pairs: List[Dict[str, str]]) -> int:
        """
        Add Q&A pairs to the dynamic index
        
        Args:
            qa_pairs: List of dicts with 'question' and 'answer' keys
            
        Returns:
            Number of pairs added
        """
        added = 0
        
        for pair in qa_pairs:
            question = pair.get('question', '').strip()
            answer = pair.get('answer', '').strip()
            
            if not question or not answer:
                continue
            
            # Create metadata
            metadata = {
                'question': question,
                'answer': answer,
                'q_type': self._detect_question_type(question),
                'char_len': len(question) + len(answer),
                'source': 'user_input',
            }
            
            custom_pair = CustomQAPair(
                question=question,
                answer=answer,
                metadata=metadata
            )
            
            self.qa_pairs.append(custom_pair)
            added += 1
        
        print(f"📝 Added {added} Q&A pairs to dynamic index", flush=True)
        return added
    
    def build_index(self) -> bool:
        """
        Build FAISS index from current Q&A pairs
        
        Returns:
            True if successful, False otherwise
        """
        if not self.qa_pairs:
            print("⚠️ No Q&A pairs to index", flush=True)
            return False
        
        print(f"\n🔨 Building index from {len(self.qa_pairs)} Q&A pairs...", flush=True)
        
        # Prepare texts for embedding
        texts = []
        self.chunks = []
        
        for i, pair in enumerate(self.qa_pairs):
            # Combine question and answer for embedding (same format as main system)
            combined_text = f"Question: {pair.question}\nAnswer: {pair.answer}"
            texts.append(combined_text)
            
            # Create chunk metadata
            chunk = {
                'chunk_id': f'custom_qa_{i+1:04d}',
                'text': combined_text,
                'metadata': pair.metadata or {}
            }
            self.chunks.append(chunk)
        
        # Generate embeddings
        print(f"🔄 Generating embeddings...", flush=True)
        embeddings = self.embedder.embed_texts(texts)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))
        
        print(f"✅ Index built successfully!", flush=True)
        print(f"   Vectors: {self.index.ntotal}", flush=True)
        print(f"   Dimension: {self.index.d}\n", flush=True)
        
        return True
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the dynamic index
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieved Q&A pairs with scores
        """
        if self.index is None or self.index.ntotal == 0:
            print("⚠️ Index is empty. Please add Q&A pairs first.", flush=True)
            return []
        
        # Embed query
        query_vec = self.embedder.embed_query(query)
        
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        
        # Search
        scores, ids = self.index.search(query_vec.astype(np.float32), min(top_k, self.index.ntotal))
        
        # Format results
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            results.append({
                'chunk_id': chunk['chunk_id'],
                'score': float(score),
                'question': chunk['metadata']['question'],
                'answer': chunk['metadata']['answer'],
                'metadata': chunk['metadata'],
            })
        
        return results
    
    def clear(self):
        """Clear all data and reset the index"""
        self.qa_pairs = []
        self.index = None
        self.chunks = []
        print("🗑️ Dynamic index cleared", flush=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        return {
            'total_pairs': len(self.qa_pairs),
            'index_built': self.index is not None,
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.index.d if self.index else 0,
        }
    
    def _detect_question_type(self, question: str) -> str:
        """Simple question type detection"""
        question_lower = question.lower()
        
        if any(q in question_lower for q in ['can', 'will', 'does', 'is', 'are', 'do']):
            return 'yes_no'
        elif any(q in question_lower for q in ['how', 'what', 'why', 'where', 'when', 'who']):
            return 'wh_question'
        else:
            return 'other'
    
    def export_to_json(self, filepath: str = None) -> str:
        """
        Export current Q&A pairs to JSON format
        
        Args:
            filepath: Optional file path to save
            
        Returns:
            JSON string of Q&A pairs
        """
        export_data = {
            'total_pairs': len(self.qa_pairs),
            'qa_pairs': [
                {
                    'question': pair.question,
                    'answer': pair.answer,
                    'metadata': pair.metadata
                }
                for pair in self.qa_pairs
            ]
        }
        
        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"💾 Exported to {filepath}", flush=True)
        
        return json_str


def test_dynamic_index():
    """Test the dynamic index builder"""
    print("\n" + "=" * 70)
    print("🧪 TESTING DYNAMIC INDEX BUILDER")
    print("=" * 70)
    
    # Initialize builder
    builder = DynamicIndexBuilder()
    
    # Sample Q&A pairs
    sample_qa = [
        {
            'question': 'What is the battery life of this phone?',
            'answer': 'The battery life is approximately 10-12 hours with normal use.'
        },
        {
            'question': 'Does this case fit iPhone 13?',
            'answer': 'Yes, this case is specifically designed for iPhone 13 and fits perfectly.'
        },
        {
            'question': 'How long does shipping take?',
            'answer': 'Standard shipping takes 3-5 business days, express shipping takes 1-2 days.'
        },
        {
            'question': 'Is this product waterproof?',
            'answer': 'Yes, it has an IP68 rating, meaning it can withstand submersion up to 1.5 meters for 30 minutes.'
        },
    ]
    
    # Add Q&A pairs
    builder.add_qa_pairs(sample_qa)
    
    # Build index
    builder.build_index()
    
    # Get stats
    stats = builder.get_stats()
    print(f"\n📊 Index Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test search
    test_queries = [
        "battery life",
        "iPhone 13 compatibility",
        "shipping time"
    ]
    
    print(f"\n🔍 Testing Searches:")
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = builder.search(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            print(f"   [{i}] Score: {result['score']:.3f}")
            print(f"       Q: {result['question']}")
            print(f"       A: {result['answer'][:60]}...")
    
    print("\n" + "=" * 70)
    print("✅ Dynamic Index Builder test complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_dynamic_index()