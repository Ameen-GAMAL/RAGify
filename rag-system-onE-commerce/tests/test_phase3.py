"""
Test Script for Phase 3: Retrieval Mechanism
Run this to validate your Phase 3 implementation
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_dependencies():
    """Test if all required packages are installed"""
    print("=" * 80)
    print("TEST 1: Checking Dependencies")
    print("=" * 80)
    
    required_packages = {
        'rank_bm25': 'rank-bm25',
        'nltk': 'nltk',
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu or faiss-gpu',
    }
    
    missing = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package:25} - Installed")
        except ImportError:
            print(f"✗ {package:25} - NOT FOUND")
            missing.append(pip_name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(set(missing)))
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def test_phase2_output():
    """Check if Phase 2 output exists"""
    print("\n" + "=" * 80)
    print("TEST 2: Checking Phase 2 Output")
    print("=" * 80)
    
    required_files = [
        PROJECT_ROOT / "models/vector_db/faiss_index.bin",
        PROJECT_ROOT / "models/vector_db/metadata.pkl",
        PROJECT_ROOT / "data/processed/processed_data.csv"
    ]
    
    all_exist = True
    for file_path in required_files:
        if file_path.exists():
            print(f"✓ Found: {file_path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"✗ Missing: {file_path.relative_to(PROJECT_ROOT)}")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Please complete Phase 2 first!")
        return False
    
    print("\n✅ Phase 2 output is ready!")
    return True


def test_query_processor():
    """Test query processing"""
    print("\n" + "=" * 80)
    print("TEST 3: Testing Query Processor")
    print("=" * 80)
    
    try:
        from src.phase3_retrieval import QueryProcessor
        
        processor = QueryProcessor()
        print("✓ QueryProcessor initialized")
        
        # Test cleaning
        test_query = "What are the BEST toys for kids???"
        cleaned = processor.clean_query(test_query)
        print(f"\n🧪 Query cleaning:")
        print(f"   Input:  '{test_query}'")
        print(f"   Output: '{cleaned}'")
        
        # Test stop word removal
        filtered = processor.remove_stop_words(cleaned)
        print(f"\n🧪 Stop word removal:")
        print(f"   Input:  '{cleaned}'")
        print(f"   Output: '{filtered}'")
        
        # Test expansion
        expanded = processor.expand_query("toy car")
        print(f"\n🧪 Query expansion:")
        print(f"   Input:  'toy car'")
        print(f"   Output: '{expanded}'")
        
        print("\n✅ Query Processor works correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing QueryProcessor: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bm25_retriever():
    """Test BM25 retriever"""
    print("\n" + "=" * 80)
    print("TEST 4: Testing BM25 Retriever")
    print("=" * 80)
    
    try:
        from src.phase3_retrieval import BM25Retriever
        
        # Create sample documents
        documents = [
            "educational toy for learning",
            "action figure superhero",
            "building blocks construction set",
            "doll princess dress up",
            "remote control car racing"
        ]
        
        metadata = [{'id': i, 'name': doc} for i, doc in enumerate(documents)]
        
        print("🧪 Creating BM25 index with sample documents...")
        retriever = BM25Retriever(documents, metadata)
        print("✓ BM25 index created")
        
        # Test search
        query = "educational learning toy"
        print(f"\n🧪 Searching for: '{query}'")
        results = retriever.search(query, top_k=3)
        
        print(f"✓ Retrieved {len(results)} results:")
        for r in results:
            print(f"   {r['rank']}. {r['metadata']['name']}")
            print(f"      BM25 Score: {r['bm25_score']:.4f}")
        
        print("\n✅ BM25 Retriever works correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing BM25Retriever: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_retriever():
    """Test hybrid retriever"""
    print("\n" + "=" * 80)
    print("TEST 5: Testing Hybrid Retriever")
    print("=" * 80)
    
    try:
        from src.phase3_retrieval import AdvancedRetriever
        
        print("🧪 Initializing Advanced Retriever...")
        print("   (This may take a moment on first run)")
        
        retriever = AdvancedRetriever()
        print("✓ Advanced Retriever initialized")
        
        # Test semantic search
        query = "educational toys"
        print(f"\n🧪 Testing semantic search: '{query}'")
        results = retriever.retrieve(query, method='semantic', top_k=3)
        print(f"✓ Retrieved {len(results)} semantic results")
        
        # Test BM25 search
        print(f"\n🧪 Testing BM25 search: '{query}'")
        results = retriever.retrieve(query, method='bm25', top_k=3)
        print(f"✓ Retrieved {len(results)} BM25 results")
        
        # Test hybrid search
        print(f"\n🧪 Testing hybrid search: '{query}'")
        results = retriever.retrieve(query, method='hybrid', top_k=3)
        print(f"✓ Retrieved {len(results)} hybrid results")
        
        if results:
            print("\n📊 Sample hybrid result:")
            r = results[0]
            print(f"   Product: {r['metadata'].get('product_name', 'N/A')[:60]}")
            print(f"   Combined Score: {r.get('combined_score', 0):.4f}")
            print(f"   Semantic Score: {r.get('semantic_score', 0):.4f}")
            print(f"   BM25 Score: {r.get('bm25_score', 0):.4f}")
        
        print("\n✅ Hybrid Retriever works correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing Hybrid Retriever: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_method_comparison():
    """Test method comparison"""
    print("\n" + "=" * 80)
    print("TEST 6: Testing Method Comparison")
    print("=" * 80)
    
    try:
        from src.phase3_retrieval import AdvancedRetriever
        
        retriever = AdvancedRetriever()
        
        query = "toy for kids"
        print(f"🧪 Comparing methods for: '{query}'")
        
        comparison = retriever.compare_methods(query, top_k=2)
        
        print("\n✓ Comparison results:")
        for method, results in comparison.items():
            print(f"\n   {method.upper()}: {len(results)} results")
            if results:
                r = results[0]
                print(f"      Top result: {r['metadata'].get('product_name', 'N/A')[:50]}")
        
        print("\n✅ Method comparison works correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing method comparison: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("PHASE 3: RETRIEVAL MECHANISM - COMPREHENSIVE TESTING")
    print("=" * 80)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Phase 2 Output", test_phase2_output),
        ("Query Processor", test_query_processor),
        ("BM25 Retriever", test_bm25_retriever),
        ("Hybrid Retriever", test_hybrid_retriever),
        ("Method Comparison", test_method_comparison),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
        
        if not results[test_name]:
            print(f"\n⚠️  Test '{test_name}' failed. Fix this before proceeding.\n")
            # Don't break - show all results
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\nYour Phase 3 setup is complete and working!")
        print("\nTo run the full retrieval system:")
        print("   python src/phase3_retrieval.py")
        print("\nOr use interactively:")
        print("   from src.phase3_retrieval import AdvancedRetriever")
        print("   retriever = AdvancedRetriever()")
        print("   results = retriever.retrieve('toys for kids', method='hybrid', top_k=5)")
        return True
    else:
        print("\n" + "=" * 80)
        print("⚠️  SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease fix the failed tests before proceeding.")
        failed_tests = [name for name, passed in results.items() if not passed]
        print(f"\nFailed tests: {', '.join(failed_tests)}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)