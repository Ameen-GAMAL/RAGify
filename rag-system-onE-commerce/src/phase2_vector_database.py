"""
RAG System - Phase 2: Vector Database Construction
Advanced Information Retrieval Course

This module handles:
- Embedding generation using sentence transformers
- Vector database creation and indexing using FAISS
- Efficient similarity search
- Metadata storage and retrieval
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
from tqdm import tqdm
import time
import json
import warnings
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')


class VectorDatabase:
    """
    Comprehensive Vector Database implementation using FAISS and Sentence Transformers
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        dimension: Optional[int] = None,
        index_type: str = 'flat'
    ):
        """
        Initialize the Vector Database
        
        Args:
            model_name: Name of the sentence transformer model
            dimension: Embedding dimension (auto-detected if None)
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        print("=" * 80)
        print("INITIALIZING VECTOR DATABASE")
        print("=" * 80)
        
        self.model_name = model_name
        self.index_type = index_type
        
        # Load embedding model
        print(f"\n📥 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = dimension or self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded successfully")
        print(f"✓ Embedding dimension: {self.dimension}")
        
        # Initialize FAISS index
        self.index = None
        self.metadata = []  # Store product metadata
        self.id_mapping = {}  # Map FAISS IDs to original IDs
        
        # Statistics
        self.stats = {
            'total_embeddings': 0,
            'index_type': index_type,
            'model_name': model_name,
            'embedding_dimension': self.dimension
        }
    
    def create_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        print("\n" + "=" * 80)
        print("GENERATING EMBEDDINGS")
        print("=" * 80)
        
        print(f"📝 Processing {len(texts):,} texts")
        print(f"📦 Batch size: {batch_size}")
        
        start_time = time.time()
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for better similarity
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Generated {len(embeddings):,} embeddings")
        print(f"✓ Embedding shape: {embeddings.shape}")
        print(f"✓ Time taken: {elapsed_time:.2f} seconds")
        print(f"✓ Speed: {len(texts)/elapsed_time:.2f} texts/second")
        
        return embeddings
    
    def build_index(
        self, 
        embeddings: np.ndarray,
        index_type: Optional[str] = None
    ) -> faiss.Index:
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: numpy array of embeddings
            index_type: Type of index to build ('flat', 'ivf', 'hnsw')
            
        Returns:
            FAISS index
        """
        print("\n" + "=" * 80)
        print("BUILDING FAISS INDEX")
        print("=" * 80)
        
        index_type = index_type or self.index_type
        n_embeddings = embeddings.shape[0]
        
        print(f"📊 Number of embeddings: {n_embeddings:,}")
        print(f"📐 Embedding dimension: {self.dimension}")
        print(f"🔧 Index type: {index_type}")
        
        start_time = time.time()
        
        if index_type == 'flat':
            # Flat index - exact search, good for smaller datasets
            index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
            print("✓ Using Flat Index (exact search)")
            
        elif index_type == 'ivf':
            # IVF index - approximate search, good for larger datasets
            nlist = min(int(np.sqrt(n_embeddings)), 100)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            print(f"✓ Using IVF Index (approximate search)")
            print(f"✓ Training index with {nlist} clusters...")
            
            # Train the index
            index.train(embeddings.astype('float32'))
            print("✓ Index training complete")
            
        elif index_type == 'hnsw':
            # HNSW index - hierarchical navigable small world
            M = 32  # Number of connections per layer
            index = faiss.IndexHNSWFlat(self.dimension, M)
            index.hnsw.efConstruction = 40
            print(f"✓ Using HNSW Index (graph-based search)")
            
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        print(f"📥 Adding {n_embeddings:,} embeddings to index...")
        index.add(embeddings.astype('float32'))
        
        elapsed_time = time.time() - start_time
        
        print(f"✓ Index built successfully")
        print(f"✓ Total embeddings in index: {index.ntotal:,}")
        print(f"✓ Build time: {elapsed_time:.2f} seconds")
        
        self.index = index
        self.stats['total_embeddings'] = index.ntotal
        
        return index
    
    def add_metadata(self, metadata_list: List[Dict]):
        """
        Store metadata for each embedding
        
        Args:
            metadata_list: List of dictionaries containing product info
        """
        print("\n📋 Storing metadata...")
        self.metadata = metadata_list
        
        # Create ID mapping
        for idx, meta in enumerate(metadata_list):
            self.id_mapping[idx] = meta.get('id', idx)
        
        print(f"✓ Stored metadata for {len(metadata_list):,} items")
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Search for similar items given a query
        
        Args:
            query: Search query text
            top_k: Number of results to return
            return_scores: Include similarity scores
            
        Returns:
            List of dictionaries with results and metadata
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search in FAISS index
        scores, indices = self.index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # Prepare results
        results = []
        for idx, (score, doc_idx) in enumerate(zip(scores[0], indices[0])):
            if doc_idx == -1:  # Invalid result
                continue
            
            result = {
                'rank': idx + 1,
                'doc_id': doc_idx,
                'metadata': self.metadata[doc_idx] if doc_idx < len(self.metadata) else {}
            }
            
            if return_scores:
                result['similarity_score'] = float(score)
            
            results.append(result)
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[Dict]]:
        """
        Search for multiple queries in batch
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            
        Returns:
            List of result lists
        """
        # Generate query embeddings
        query_embeddings = self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Batch search
        scores, indices = self.index.search(
            query_embeddings.astype('float32'),
            top_k
        )
        
        # Format results
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for idx, (score, doc_idx) in enumerate(zip(query_scores, query_indices)):
                if doc_idx != -1:
                    results.append({
                        'rank': idx + 1,
                        'doc_id': doc_idx,
                        'similarity_score': float(score),
                        'metadata': self.metadata[doc_idx] if doc_idx < len(self.metadata) else {}
                    })
            all_results.append(results)
        
        return all_results
    
    def save(self, save_dir: str = "models/vector_db"):
        """
        Save the vector database to disk
        
        Args:
            save_dir: Directory to save the database (relative to project root)
        """
        print("\n" + "=" * 80)
        print("SAVING VECTOR DATABASE")
        print("=" * 80)
        
        save_path = PROJECT_ROOT / save_dir
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = save_path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        print(f"✓ FAISS index saved: {index_path}")
        
        # Save metadata
        metadata_path = save_path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"✓ Metadata saved: {metadata_path}")
        
        # Save ID mapping
        id_mapping_path = save_path / "id_mapping.pkl"
        with open(id_mapping_path, 'wb') as f:
            pickle.dump(self.id_mapping, f)
        print(f"✓ ID mapping saved: {id_mapping_path}")
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'stats': self.stats
        }
        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved: {config_path}")
        
        print(f"\n✅ Vector database saved to: {save_path}")
    
    @classmethod
    def load(cls, load_dir: str = "models/vector_db") -> 'VectorDatabase':
        """
        Load a saved vector database
        
        Args:
            load_dir: Directory containing saved database (relative to project root)
            
        Returns:
            VectorDatabase instance
        """
        print("\n" + "=" * 80)
        print("LOADING VECTOR DATABASE")
        print("=" * 80)
        
        load_path = PROJECT_ROOT / load_dir
        
        # Load configuration
        config_path = load_path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"📥 Loading from: {load_path}")
        print(f"📊 Model: {config['model_name']}")
        print(f"📐 Dimension: {config['dimension']}")
        
        # Create instance
        db = cls(
            model_name=config['model_name'],
            dimension=config['dimension'],
            index_type=config['index_type']
        )
        
        # Load FAISS index
        index_path = load_path / "faiss_index.bin"
        db.index = faiss.read_index(str(index_path))
        print(f"✓ FAISS index loaded: {db.index.ntotal:,} embeddings")
        
        # Load metadata
        metadata_path = load_path / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            db.metadata = pickle.load(f)
        print(f"✓ Metadata loaded: {len(db.metadata):,} items")
        
        # Load ID mapping
        id_mapping_path = load_path / "id_mapping.pkl"
        with open(id_mapping_path, 'rb') as f:
            db.id_mapping = pickle.load(f)
        print(f"✓ ID mapping loaded")
        
        db.stats = config['stats']
        
        print(f"\n✅ Vector database loaded successfully")
        
        return db
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        stats = self.stats.copy()
        stats['metadata_count'] = len(self.metadata)
        stats['index_size'] = self.index.ntotal if self.index else 0
        return stats
    
    def print_stats(self):
        """Print database statistics"""
        print("\n" + "=" * 80)
        print("VECTOR DATABASE STATISTICS")
        print("=" * 80)
        
        stats = self.get_stats()
        
        print(f"📊 Model: {stats['model_name']}")
        print(f"📐 Embedding Dimension: {stats['embedding_dimension']}")
        print(f"🔧 Index Type: {stats['index_type']}")
        print(f"📝 Total Embeddings: {stats['total_embeddings']:,}")
        print(f"📋 Metadata Items: {stats['metadata_count']:,}")
        print(f"💾 Index Size: {stats['index_size']:,}")
        
        print("=" * 80)


def build_vector_database_from_csv(
    csv_path: str,
    text_columns: List[str] = None,
    id_column: str = None,
    model_name: str = 'all-MiniLM-L6-v2',
    index_type: str = 'flat',
    batch_size: int = 32,
    save_dir: str = "models/vector_db"
) -> VectorDatabase:
    """
    Build vector database from processed CSV file
    
    Args:
        csv_path: Path to processed CSV file (relative to project root)
        text_columns: Columns to combine for embedding (auto-detect if None)
        id_column: Column to use as ID
        model_name: Sentence transformer model name
        index_type: FAISS index type
        batch_size: Batch size for embedding generation
        save_dir: Directory to save the database (relative to project root)
        
    Returns:
        VectorDatabase instance
    """
    print("=" * 80)
    print("BUILDING VECTOR DATABASE FROM CSV")
    print("=" * 80)
    
    # Load data
    csv_full_path = PROJECT_ROOT / csv_path
    print(f"\n📥 Loading data from: {csv_full_path}")
    df = pd.read_csv(csv_full_path)
    print(f"✓ Loaded {len(df):,} records")
    
    # Auto-detect text columns if not specified
    if text_columns is None:
        text_columns = []
        priority_keywords = ['title', 'name', 'description', 'brand', 'category', 'manufacturer']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in priority_keywords):
                if df[col].dtype == 'object':
                    non_null_pct = df[col].count() / len(df)
                    if non_null_pct > 0.5:
                        text_columns.append(col)
    
    print(f"\n📝 Using text columns: {text_columns}")
    
    # Create combined text
    def combine_text(row):
        parts = []
        for col in text_columns:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                parts.append(str(value).strip())
        return " | ".join(parts) if parts else "No content"
    
    print("\n🔄 Creating combined text for embeddings...")
    texts = df.apply(combine_text, axis=1).tolist()
    print(f"✓ Created {len(texts):,} text entries")
    
    # Prepare metadata
    print("\n📋 Preparing metadata...")
    metadata_list = df.to_dict('records')
    
    # Add sequential ID if no ID column specified
    if id_column and id_column in df.columns:
        for idx, meta in enumerate(metadata_list):
            meta['id'] = meta[id_column]
    else:
        for idx, meta in enumerate(metadata_list):
            meta['id'] = idx
    
    print(f"✓ Prepared metadata for {len(metadata_list):,} items")
    
    # Initialize vector database
    db = VectorDatabase(
        model_name=model_name,
        index_type=index_type
    )
    
    # Generate embeddings
    embeddings = db.create_embeddings(texts, batch_size=batch_size)
    
    # Build index
    db.build_index(embeddings)
    
    # Add metadata
    db.add_metadata(metadata_list)
    
    # Save database
    db.save(save_dir)
    
    # Print statistics
    db.print_stats()
    
    print("\n✅ VECTOR DATABASE BUILD COMPLETE!")
    
    return db


def main():
    """Main execution function for Phase 2"""
    print("=" * 80)
    print("RAG SYSTEM - PHASE 2: VECTOR DATABASE CONSTRUCTION")
    print("Advanced Information Retrieval Course")
    print("=" * 80)
    
    # Configuration
    csv_path = "data/processed/processed_data.csv"
    model_name = 'all-MiniLM-L6-v2'  # Fast and efficient model
    index_type = 'flat'  # Use 'ivf' for larger datasets
    batch_size = 32
    save_dir = "models/vector_db"
    
    try:
        # Build vector database
        db = build_vector_database_from_csv(
            csv_path=csv_path,
            model_name=model_name,
            index_type=index_type,
            batch_size=batch_size,
            save_dir=save_dir
        )
        
        # Test search functionality
        print("\n" + "=" * 80)
        print("TESTING SEARCH FUNCTIONALITY")
        print("=" * 80)
        
        test_queries = [
            "educational toys for kids",
            "action figures",
            "building blocks"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            print("-" * 80)
            
            results = db.search(query, top_k=3)
            
            for result in results:
                print(f"\n   Rank {result['rank']}:")
                print(f"   Similarity: {result['similarity_score']:.4f}")
                
                # Show some metadata
                metadata = result['metadata']
                if 'product_name' in metadata:
                    print(f"   Product: {metadata['product_name'][:80]}")
                if 'manufacturer' in metadata:
                    print(f"   Brand: {metadata.get('manufacturer', 'N/A')}")
        
        print("\n" + "=" * 80)
        print("✅ PHASE 2 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nNext Steps:")
        print("1. Vector database saved in 'models/vector_db/' directory")
        print("2. Test search with different queries")
        print("3. Proceed to Phase 3: Retrieval Mechanism")
        
    except Exception as e:
        print(f"\n❌ Error in Phase 2: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()