"""
RAG System - Phase 3: Usage Examples
Advanced retrieval strategies and comparisons
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3_retrieval import AdvancedRetriever


# ============================================================================
# Example 1: Basic Retrieval with Different Methods
# ============================================================================

def example_basic_retrieval():
    """Compare different retrieval methods"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Retrieval with Different Methods")
    print("=" * 80)
    
    retriever = AdvancedRetriever()
    query = "educational toys for toddlers"
    
    methods = ['semantic', 'bm25', 'hybrid']
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Method: {method.upper()}")
        print('='*80)
        
        results = retriever.retrieve(query, method=method, top_k=5)
        
        for r in results:
            print(f"\n{r['rank']}. {r['metadata'].get('product_name', 'N/A')[:70]}")
            
            if 'combined_score' in r:
                print(f"   Combined: {r['combined_score']:.4f}")
                print(f"   Semantic: {r['semantic_score']:.4f}")
                print(f"   BM25: {r['bm25_score']:.4f}")
            elif 'similarity_score' in r:
                print(f"   Similarity: {r['similarity_score']:.4f}")
            elif 'bm25_score' in r:
                print(f"   BM25: {r['bm25_score']:.4f}")


# ============================================================================
# Example 2: Query Processing Demonstration
# ============================================================================

def example_query_processing():
    """Show how query processing works"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Query Processing Demonstration")
    print("=" * 80)
    
    from src.phase3_retrieval import QueryProcessor
    
    processor = QueryProcessor()
    
    test_queries = [
        "What are the BEST toys for KIDS???",
        "I need some educational games",
        "Looking for LEGO Star Wars sets!!!"
    ]
    
    for original in test_queries:
        print(f"\n{'='*80}")
        print(f"Original: '{original}'")
        print("-" * 80)
        
        # Clean
        cleaned = processor.clean_query(original)
        print(f"Cleaned:  '{cleaned}'")
        
        # Remove stop words
        filtered = processor.remove_stop_words(cleaned)
        print(f"Filtered: '{filtered}'")
        
        # Expand
        expanded = processor.expand_query(filtered)
        print(f"Expanded: '{expanded}'")


# ============================================================================
# Example 3: Brand-Specific Searches
# ============================================================================

def example_brand_search():
    """Search for specific brands - BM25 excels here"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Brand-Specific Searches")
    print("=" * 80)
    
    retriever = AdvancedRetriever()
    
    brand_queries = [
        "LEGO building sets",
        "Mattel toys",
        "Hasbro action figures"
    ]
    
    for query in brand_queries:
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print('='*80)
        
        # Compare BM25 vs Hybrid
        bm25_results = retriever.retrieve(query, method='bm25', top_k=3)
        hybrid_results = retriever.retrieve(query, method='hybrid', top_k=3)
        
        print("\n📊 BM25 Results:")
        for r in bm25_results:
            print(f"   {r['rank']}. {r['metadata'].get('product_name', 'N/A')[:60]}")
        
        print("\n📊 Hybrid Results:")
        for r in hybrid_results:
            print(f"   {r['rank']}. {r['metadata'].get('product_name', 'N/A')[:60]}")


# ============================================================================
# Example 4: Conceptual Searches
# ============================================================================

def example_conceptual_search():
    """Semantic search excels at conceptual queries"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Conceptual Searches")
    print("=" * 80)
    
    retriever = AdvancedRetriever()
    
    conceptual_queries = [
        "toys that encourage creativity",
        "educational games for learning",
        "outdoor play equipment"
    ]
    
    for query in conceptual_queries:
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print('='*80)
        
        # Semantic search is best for concepts
        results = retriever.retrieve(query, method='semantic', top_k=5)
        
        for r in results:
            print(f"\n{r['rank']}. {r['metadata'].get('product_name', 'N/A')[:70]}")
            print(f"   Similarity: {r.get('similarity_score', 0):.4f}")


# ============================================================================
# Example 5: Adjusting Hybrid Weights
# ============================================================================

def example_weight_tuning():
    """Experiment with different weight combinations"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Adjusting Hybrid Weights")
    print("=" * 80)
    
    retriever = AdvancedRetriever()
    query = "LEGO educational building blocks"
    
    weight_configs = [
        (0.7, 0.3, "High Semantic"),
        (0.5, 0.5, "Balanced"),
        (0.3, 0.7, "High BM25")
    ]
    
    for sem_weight, bm25_weight, label in weight_configs:
        print(f"\n{'='*80}")
        print(f"Configuration: {label}")
        print(f"Semantic: {sem_weight}, BM25: {bm25_weight}")
        print('='*80)
        
        # Set weights
        retriever.hybrid_retriever.semantic_weight = sem_weight
        retriever.hybrid_retriever.bm25_weight = bm25_weight
        
        # Search
        results = retriever.retrieve(query, method='hybrid', top_k=3)
        
        for r in results:
            print(f"\n{r['rank']}. {r['metadata'].get('product_name', 'N/A')[:70]}")
            print(f"   Combined: {r['combined_score']:.4f}")
            print(f"   Semantic: {r['semantic_score']:.4f}")
            print(f"   BM25: {r['bm25_score']:.4f}")
    
    # Reset to default
    retriever.hybrid_retriever.semantic_weight = 0.5
    retriever.hybrid_retriever.bm25_weight = 0.5


# ============================================================================
# Example 6: Re-Ranking Demonstration
# ============================================================================

def example_reranking():
    """Show the effect of re-ranking"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Re-Ranking Demonstration")
    print("=" * 80)
    
    retriever = AdvancedRetriever()
    query = "action figures for boys age 8"
    
    if not retriever.reranker:
        print("⚠️  Re-ranker not available")
        return
    
    print(f"Query: '{query}'")
    
    # Without re-ranking
    print("\n" + "="*80)
    print("WITHOUT RE-RANKING")
    print("="*80)
    
    results_no_rerank = retriever.retrieve(
        query,
        method='hybrid',
        top_k=5,
        rerank=False
    )
    
    for r in results_no_rerank:
        print(f"\n{r['rank']}. {r['metadata'].get('product_name', 'N/A')[:70]}")
        print(f"   Combined: {r['combined_score']:.4f}")
    
    # With re-ranking
    print("\n" + "="*80)
    print("WITH RE-RANKING")
    print("="*80)
    
    results_rerank = retriever.retrieve(
        query,
        method='hybrid',
        top_k=5,
        rerank=True
    )
    
    for r in results_rerank:
        print(f"\n{r['rank']}. {r['metadata'].get('product_name', 'N/A')[:70]}")
        print(f"   Rerank: {r.get('rerank_score', 0):.4f}")
        print(f"   Combined: {r.get('combined_score', 0):.4f}")


# ============================================================================
# Example 7: Batch Query Processing
# ============================================================================

def example_batch_queries():
    """Process multiple queries at once"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Batch Query Processing")
    print("=" * 80)
    
    retriever = AdvancedRetriever()
    
    queries = [
        "dolls for girls",
        "remote control cars",
        "board games family",
        "educational science kits",
        "outdoor sports equipment"
    ]
    
    print(f"Processing {len(queries)} queries...\n")
    
    for query in queries:
        print(f"{'='*80}")
        print(f"Query: '{query}'")
        print("-"*80)
        
        results = retriever.retrieve(query, method='hybrid', top_k=3)
        
        for r in results:
            print(f"{r['rank']}. {r['metadata'].get('product_name', 'N/A')[:60]}")
        
        print()


# ============================================================================
# Example 8: Full Comparison Analysis
# ============================================================================

def example_full_comparison():
    """Comprehensive comparison of all methods"""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Full Comparison Analysis")
    print("=" * 80)
    
    retriever = AdvancedRetriever()
    
    # Test different types of queries
    test_cases = [
        {
            'query': 'LEGO Star Wars',
            'type': 'Brand + Theme',
            'expected_best': 'bm25 or hybrid'
        },
        {
            'query': 'creative art supplies',
            'type': 'Conceptual',
            'expected_best': 'semantic'
        },
        {
            'query': 'learning toys for preschool',
            'type': 'Mixed',
            'expected_best': 'hybrid'
        }
    ]
    
    for test in test_cases:
        print(f"\n{'='*80}")
        print(f"Query: '{test['query']}'")
        print(f"Type: {test['type']}")
        print(f"Expected Best: {test['expected_best']}")
        print('='*80)
        
        comparison = retriever.compare_methods(test['query'], top_k=3)
        
        # Display all methods
        for method, results in comparison.items():
            print(f"\n📊 {method.upper()}:")
            
            if not results:
                print("   No results")
                continue
            
            for r in results:
                print(f"   {r['rank']}. {r['metadata'].get('product_name', 'N/A')[:60]}")


# ============================================================================
# Example 9: Export Results to CSV
# ============================================================================

def example_export_results():
    """Export search results for analysis"""
    print("\n" + "=" * 80)
    print("EXAMPLE 9: Export Results to CSV")
    print("=" * 80)
    
    import pandas as pd
    
    retriever = AdvancedRetriever()
    
    queries = [
        "educational toys",
        "action figures",
        "board games"
    ]
    
    all_results = []
    
    for query in queries:
        results = retriever.retrieve(query, method='hybrid', top_k=10)
        
        for r in results:
            all_results.append({
                'query': query,
                'rank': r['rank'],
                'product_name': r['metadata'].get('product_name', 'N/A'),
                'manufacturer': r['metadata'].get('manufacturer', 'N/A'),
                'combined_score': r.get('combined_score', 0),
                'semantic_score': r.get('semantic_score', 0),
                'bm25_score': r.get('bm25_score', 0)
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    output_path = PROJECT_ROOT / "outputs/reports/retrieval_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Exported {len(all_results)} results to: {output_path}")
    print("\nSample results:")
    print(df.head(10).to_string())


# ============================================================================
# Main: Run Examples
# ============================================================================

def main():
    """Run all examples"""
    print("RAG System - Phase 3 Examples")
    print("=" * 80)
    print("\nAvailable examples:")
    print("1. Basic retrieval with different methods")
    print("2. Query processing demonstration")
    print("3. Brand-specific searches")
    print("4. Conceptual searches")
    print("5. Adjusting hybrid weights")
    print("6. Re-ranking demonstration")
    print("7. Batch query processing")
    print("8. Full comparison analysis")
    print("9. Export results to CSV")
    
    choice = input("\nRun example (1-9) or 'all' for all: ").strip()
    
    examples = {
        '1': example_basic_retrieval,
        '2': example_query_processing,
        '3': example_brand_search,
        '4': example_conceptual_search,
        '5': example_weight_tuning,
        '6': example_reranking,
        '7': example_batch_queries,
        '8': example_full_comparison,
        '9': example_export_results
    }
    
    if choice == 'all':
        for name, func in examples.items():
            print(f"\n\n{'='*80}")
            print(f"Running Example {name}")
            print('='*80)
            try:
                func()
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    elif choice in examples:
        examples[choice]()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()