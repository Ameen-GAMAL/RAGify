"""
RAG System - Phase 3: Retrieval Mechanism
Advanced Information Retrieval Course

Fix: Display productName + description instead of Rank N/A
by injecting productName/description into vector DB metadata.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import sys
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import json
import time
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2_vector_database import VectorDatabase


class QueryProcessor:
    """Processes and enhances user queries"""

    def __init__(self):
        self.stop_words = set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
            'that', 'the', 'to', 'was', 'will', 'with'
        ])

    def clean_query(self, query: str) -> str:
        query = query.lower().strip()
        query = re.sub(r'[^a-z0-9\s-]', '', query)  # keep hyphens
        query = ' '.join(query.split())
        return query

    def expand_query(self, query: str, synonyms: Dict[str, List[str]] = None) -> str:
        if not synonyms:
            synonyms = {
                'toy': ['plaything', 'game'],
                'kid': ['child', 'children', 'toddler'],
                'educational': ['learning', 'educational'],
                'doll': ['figurine', 'action figure'],
                'car': ['vehicle', 'automobile'],
            }

        words = query.split()
        expanded_words = set(words)
        for word in words:
            if word in synonyms:
                expanded_words.update(synonyms[word])
        return ' '.join(expanded_words)

    def process(self, query: str, expand: bool = False) -> str:
        query = self.clean_query(query)
        if expand:
            query = self.expand_query(query)
        return query


class BM25Retriever:
    """BM25 keyword-based retrieval"""

    def __init__(self, documents: List[str], metadata: List[Dict] = None):
        print("=" * 80)
        print("INITIALIZING BM25 RETRIEVER")
        print("=" * 80)

        self.documents = documents
        self.metadata = metadata or [{} for _ in documents]

        print(f"\n📝 Tokenizing {len(documents):,} documents...")
        self.tokenized_docs = [doc.lower().split() for doc in documents]

        print("🔧 Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_docs)

        print(f"✓ BM25 index built with {len(documents):,} documents")
        print("=" * 80)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] > 0:
                results.append({
                    'rank': rank,
                    'doc_id': int(idx),
                    'bm25_score': float(scores[idx]),
                    'metadata': self.metadata[idx]
                })

        return results


class HybridRetriever:
    """Combines semantic and keyword-based retrieval"""

    def __init__(
        self,
        vector_db: VectorDatabase,
        bm25_retriever: BM25Retriever,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5
    ):
        print("=" * 80)
        print("INITIALIZING HYBRID RETRIEVER")
        print("=" * 80)

        self.vector_db = vector_db
        self.bm25_retriever = bm25_retriever
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        print(f"✓ Semantic weight: {semantic_weight}")
        print(f"✓ BM25 weight: {bm25_weight}")
        print("=" * 80)

    @staticmethod
    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()
        if max_score - min_score == 0:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def search(self, query: str, top_k: int = 10, retrieve_k: int = 50) -> List[Dict]:
        semantic_results = self.vector_db.search(query, top_k=retrieve_k)
        bm25_results = self.bm25_retriever.search(query, top_k=retrieve_k)

        semantic_scores = {r['doc_id']: r['similarity_score'] for r in semantic_results}
        bm25_scores = {r['doc_id']: r['bm25_score'] for r in bm25_results}

        all_doc_ids = set(semantic_scores.keys()) | set(bm25_scores.keys())

        if semantic_scores:
            arr = np.array(list(semantic_scores.values()))
            norm = self.normalize_scores(arr)
            semantic_scores = {doc_id: float(norm[i]) for i, doc_id in enumerate(semantic_scores.keys())}

        if bm25_scores:
            arr = np.array(list(bm25_scores.values()))
            norm = self.normalize_scores(arr)
            bm25_scores = {doc_id: float(norm[i]) for i, doc_id in enumerate(bm25_scores.keys())}

        combined_scores = {}
        for doc_id in all_doc_ids:
            sem = semantic_scores.get(doc_id, 0.0)
            bm = bm25_scores.get(doc_id, 0.0)
            combined_scores[doc_id] = {
                'combined_score': self.semantic_weight * sem + self.bm25_weight * bm,
                'semantic_score': sem,
                'bm25_score': bm,
            }

        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)[:top_k]

        results = []
        for rank, (doc_id, scores) in enumerate(sorted_docs, 1):
            results.append({
                'rank': rank,
                'doc_id': doc_id,
                'combined_score': scores['combined_score'],
                'semantic_score': scores['semantic_score'],
                'bm25_score': scores['bm25_score'],
                'metadata': self.vector_db.metadata[doc_id] if doc_id < len(self.vector_db.metadata) else {}
            })

        return results


class ReRanker:
    """Re-ranks retrieved results using cross-encoder"""

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print("=" * 80)
        print("INITIALIZING RE-RANKER")
        print("=" * 80)

        print(f"\n📥 Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        print("✓ Cross-encoder loaded")
        print("=" * 80)

    @staticmethod
    def _get_text_for_rerank(meta: Dict) -> str:
        """Use richer text for reranking if available."""
        if not isinstance(meta, dict):
            return ""
        # prefer product details/description; fallback to name
        for k in ["productDetails", "description", "product_name", "productName"]:
            v = meta.get(k, "")
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
        return ""

    def rerank(self, query: str, results: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        if not results:
            return results

        pairs = []
        for r in results:
            meta = r.get("metadata", {}) or {}
            doc_text = self._get_text_for_rerank(meta)
            pairs.append([query, doc_text])

        scores = self.model.predict(pairs)

        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        for rank, r in enumerate(reranked, 1):
            r["rank"] = rank

        if top_k:
            reranked = reranked[:top_k]

        return reranked


class AdvancedRetriever:
    """Complete retrieval system combining all strategies"""

    def __init__(self, vector_db_path: str = "models/vector_db", processed_data_path: str = "data/processed/processed_data.csv"):
        print("=" * 80)
        print("INITIALIZING ADVANCED RETRIEVAL SYSTEM")
        print("Advanced Information Retrieval Course - Phase 3")
        print("=" * 80)

        print("\n📥 Loading vector database...")
        self.vector_db = VectorDatabase.load(PROJECT_ROOT / vector_db_path)
        print(f"✓ Loaded {self.vector_db.index.ntotal:,} embeddings")

        print("\n📥 Loading processed data...")
        df = pd.read_csv(PROJECT_ROOT / processed_data_path)
        print(f"✓ Loaded {len(df):,} records")

        # ✅ FIX: Attach productName + description into metadata for printing and reranking
        self._attach_product_fields_to_metadata(df)

        print("\n🔄 Creating documents for BM25...")
        documents = self._create_documents(df)
        print(f"✓ Created {len(documents):,} documents")

        self.query_processor = QueryProcessor()
        self.bm25_retriever = BM25Retriever(documents, self.vector_db.metadata)
        self.hybrid_retriever = HybridRetriever(
            self.vector_db,
            self.bm25_retriever,
            semantic_weight=0.5,
            bm25_weight=0.5
        )

        try:
            self.reranker = ReRanker()
        except Exception as e:
            print(f"⚠️  Re-ranker not available: {e}")
            self.reranker = None

        print("\n✅ ADVANCED RETRIEVAL SYSTEM READY!")
        print("=" * 80)

    def _attach_product_fields_to_metadata(self, df: pd.DataFrame) -> None:
        """
        Your real columns include productName + description.
        This function injects them into vector_db.metadata as:
          - product_name  (from df['productName'])
          - description   (from df['description'])
        Assumes row order alignment between df rows and metadata entries.
        """
        if "productName" not in df.columns or "description" not in df.columns:
            raise ValueError("Expected columns 'productName' and 'description' were not found in processed_data.csv")

        if not hasattr(self.vector_db, "metadata") or not isinstance(self.vector_db.metadata, list):
            raise ValueError("Vector DB metadata missing/invalid; cannot attach product fields.")

        n = min(len(df), len(self.vector_db.metadata))
        filled_name = 0
        filled_desc = 0

        for i in range(n):
            meta = self.vector_db.metadata[i]
            if not isinstance(meta, dict):
                meta = {}
                self.vector_db.metadata[i] = meta

            # product_name
            if not str(meta.get("product_name", "")).strip():
                v = df.iloc[i]["productName"]
                if pd.notna(v) and str(v).strip():
                    meta["product_name"] = str(v).strip()
                    filled_name += 1

            # description
            if not str(meta.get("description", "")).strip():
                v = df.iloc[i]["description"]
                if pd.notna(v) and str(v).strip():
                    meta["description"] = str(v).strip()
                    filled_desc += 1

            # keep original column names too (optional but helpful)
            if "productName" not in meta and pd.notna(df.iloc[i]["productName"]):
                meta["productName"] = str(df.iloc[i]["productName"]).strip()
            if "description" not in meta and pd.notna(df.iloc[i]["description"]):
                meta["description"] = str(df.iloc[i]["description"]).strip()

        print(f"✓ Attached product_name for {filled_name:,} rows; description for {filled_desc:,} rows")

    @staticmethod
    def _create_documents(df: pd.DataFrame) -> List[str]:
        """
        Build BM25 documents from the most informative text fields.
        With your schema, these are usually:
          productName, description, productDetails, merchantName, searchKeyword
        """
        preferred_cols = ["productName", "description", "productDetails", "merchantName", "searchKeyword"]
        cols = [c for c in preferred_cols if c in df.columns]

        # fallback to all object columns if needed
        if not cols:
            cols = list(df.select_dtypes(include=["object"]).columns)

        documents = []
        for _, row in df.iterrows():
            parts = []
            for c in cols:
                v = row[c]
                if pd.notna(v) and str(v).strip():
                    parts.append(str(v).strip())
            documents.append(" ".join(parts))
        return documents

    def retrieve(self, query: str, method: str = 'hybrid', top_k: int = 10, rerank: bool = False, process_query: bool = True) -> List[Dict]:
        if process_query:
            original_query = query
            query = self.query_processor.process(query)
            print(f"📝 Query: '{original_query}' → '{query}'")

        if method == 'semantic':
            results = self.vector_db.search(query, top_k=top_k if not rerank else top_k * 2)
        elif method == 'bm25':
            results = self.bm25_retriever.search(query, top_k=top_k if not rerank else top_k * 2)
        elif method == 'hybrid':
            results = self.hybrid_retriever.search(query, top_k=top_k if not rerank else top_k * 2)
        else:
            raise ValueError(f"Unknown method: {method}")

        if rerank and self.reranker and results:
            print("🔄 Applying re-ranking...")
            results = self.reranker.rerank(query, results, top_k=top_k)

        return results

    def compare_methods(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        print("\n" + "=" * 80)
        print("COMPARING RETRIEVAL METHODS")
        print(f"Query: '{query}'")
        print("=" * 80)

        methods = ['semantic', 'bm25', 'hybrid']
        comparison = {}

        for method in methods:
            print(f"\n🔍 Method: {method.upper()}")
            start = time.time()
            results = self.retrieve(query, method=method, top_k=top_k, process_query=False)
            elapsed = time.time() - start

            comparison[method] = results
            print(f"   Retrieved {len(results)} results in {elapsed:.4f}s")

        return comparison

    def save_config(self, output_path: str = "models/retrieval_config.json"):
        config = {
            'semantic_weight': self.hybrid_retriever.semantic_weight,
            'bm25_weight': self.hybrid_retriever.bm25_weight,
            'vector_db_stats': self.vector_db.get_stats(),
            'num_documents': len(self.bm25_retriever.documents)
        }

        output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Configuration saved to: {output_path}")


def _safe_trim(text: str, limit: int = 220) -> str:
    text = "" if text is None else str(text)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def main():
    print("=" * 80)
    print("RAG SYSTEM - PHASE 3: RETRIEVAL MECHANISM")
    print("Advanced Information Retrieval Course")
    print("=" * 80)

    try:
        retriever = AdvancedRetriever()

        test_queries = [
            "educational toys for toddlers",
            "action figures superheroes",
            "building blocks construction set"
        ]

        print("\n" + "=" * 80)
        print("TESTING RETRIEVAL METHODS")
        print("=" * 80)

        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"Query: '{query}'")
            print('='*80)

            comparison = retriever.compare_methods(query, top_k=3)

            for method, results in comparison.items():
                print(f"\n📊 {method.upper()} Results:")
                print("-" * 80)

                for r in results[:3]:
                    meta = r.get("metadata", {}) or {}

                    # ✅ Display product name + description
                    name = meta.get("product_name") or meta.get("productName") or "N/A"
                    desc = meta.get("description") or ""

                    print(f"\nRank {r['rank']}: {_safe_trim(name, 80)}")
                    if desc:
                        print(f"  Description: {_safe_trim(desc, 240)}")

                    if 'combined_score' in r:
                        print(f"  Combined: {r['combined_score']:.4f} (Sem: {r['semantic_score']:.4f}, BM25: {r['bm25_score']:.4f})")
                    elif 'similarity_score' in r:
                        print(f"  Similarity: {r['similarity_score']:.4f}")
                    elif 'bm25_score' in r:
                        print(f"  BM25: {r['bm25_score']:.4f}")

        retriever.save_config()

        print("\n" + "=" * 80)
        print("✅ PHASE 3 COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error in Phase 3: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
