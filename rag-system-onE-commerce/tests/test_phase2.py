"""
Test Script for Phase 2: Vector Database Construction
Run this to validate your Phase 2 implementation
"""

import sys
from pathlib import Path
import time

def test_dependencies():
    """Test if all required packages are installed"""
    print("=" * 80)
    print("TEST 1: Checking Dependencies")
    print("=" * 80)
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu or faiss-gpu',
        'torch': 'torch',
        'tqdm': 'tqdm'
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
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def test_phase1_output():
    """Check if Phase 1 output exists"""
    print("\n" + "=" * 80)
    print("TEST 2: Checking Phase 1 Output")
    print("=" * 80)
    
    required_file = Path("data/processed_data.csv")
    
    if not required_file.exists():
        print(f"✗ Phase 1 output not found: {required_file}")
        print("\n⚠️  Please complete Phase 1 first!")
        print("Run: python data_processing.py")
        return False
    
    print(f"✓ Found: {required_file}")
    print(f"  Size: {required_file.stat().st_size / (1024*1024):.2f} MB")
    
    # Check if it's readable
    try:
        import pandas as pd
        df = pd.read_csv(required_file, nrows=5)
        print(f"  Rows: Can read (sample of 5 loaded)")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Sample columns: {', '.join(df.columns[:3])}...")
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False
    
    print("\n✅ Phase 1 output is ready!")
    return True


def test_embedding_model():
    """Test loading the embedding model"""
    print("\n" + "=" * 80)
    print("TEST 3: Testing Embedding Model")
    print("=" * 80)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("📥 Loading model: all-MiniLM-L6-v2")
        print("   (First time will download ~90MB)")
        
        start = time.time()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        load_time = time.time() - start
        
        print(f"✓ Model loaded in {load_time:.2f} seconds")
        print(f"✓ Embedding dimension: {model.get_sentence_embedding_dimension()}")
        
        # Test encoding
        print("\n🧪 Testing encoding...")
        test_text = ["This is a test sentence", "Another test"]
        embeddings = model.encode(test_text)
        
        print(f"✓ Generated embeddings shape: {embeddings.shape}")
        print(f"✓ Embedding values look good: [{embeddings[0][0]:.4f}, {embeddings[0][1]:.4f}, ...]")
        
        print("\n✅ Embedding model works correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error with embedding model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_faiss():
    """Test FAISS installation"""
    print("\n" + "=" * 80)
    print("TEST 4: Testing FAISS")
    print("=" * 80)
    
    try:
        import faiss
        import numpy as np
        
        print(f"✓ FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")
        
        # Test creating a simple index
        dimension = 128
        n_vectors = 100
        
        print(f"\n🧪 Creating test index...")
        print(f"   Dimension: {dimension}")
        print(f"   Vectors: {n_vectors}")
        
        # Create random vectors
        vectors = np.random.random((n_vectors, dimension)).astype('float32')
        
        # Create and populate index
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        
        print(f"✓ Index created with {index.ntotal} vectors")
        
        # Test search
        query = np.random.random((1, dimension)).astype('float32')
        distances, indices = index.search(query, k=5)
        
        print(f"✓ Search works - found {len(indices[0])} results")
        print(f"✓ Sample similarity scores: {distances[0][:3]}")
        
        print("\n✅ FAISS works correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error with FAISS: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_database_class():
    """Test the VectorDatabase class"""
    print("\n" + "=" * 80)
    print("TEST 5: Testing VectorDatabase Class")
    print("=" * 80)
    
    try:
        from vector_database import VectorDatabase
        
        print("✓ VectorDatabase class imported successfully")
        
        # Test initialization
        print("\n🧪 Testing initialization...")
        db = VectorDatabase(model_name='all-MiniLM-L6-v2', index_type='flat')
        print(f"✓ Database initialized")
        print(f"  Model: {db.model_name}")
        print(f"  Dimension: {db.dimension}")
        print(f"  Index type: {db.index_type}")
        
        # Test embedding generation
        print("\n🧪 Testing embedding generation...")
        test_texts = [
            "Educational toy for kids",
            "Action figure superhero",
            "Building blocks set"
        ]
        embeddings = db.create_embeddings(test_texts, show_progress=False)
        print(f"✓ Generated embeddings: {embeddings.shape}")
        
        # Test index building
        print("\n🧪 Testing index building...")
        db.build_index(embeddings)
        print(f"✓ Index built with {db.index.ntotal} vectors")
        
        # Test metadata
        print("\n🧪 Testing metadata storage...")
        metadata = [
            {'id': 0, 'name': 'Product 1', 'price': 19.99},
            {'id': 1, 'name': 'Product 2', 'price': 29.99},
            {'id': 2, 'name': 'Product 3', 'price': 39.99}
        ]
        db.add_metadata(metadata)
        print(f"✓ Metadata stored for {len(db.metadata)} items")
        
        # Test search
        print("\n🧪 Testing search...")
        results = db.search("toy for children", top_k=2)
        print(f"✓ Search returned {len(results)} results")
        for r in results:
            print(f"  Rank {r['rank']}: {r['metadata']['name']} (score: {r['similarity_score']:.4f})")
        
        print("\n✅ VectorDatabase class works correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing VectorDatabase: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load():
    """Test save/load functionality"""
    print("\n" + "=" * 80)
    print("TEST 6: Testing Save/Load Functionality")
    print("=" * 80)
    
    try:
        from vector_database import VectorDatabase
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Create and populate database
        print("\n🧪 Creating database...")
        db = VectorDatabase(model_name='all-MiniLM-L6-v2')
        
        test_texts = ["Product A", "Product B", "Product C"]
        embeddings = db.create_embeddings(test_texts, show_progress=False)
        db.build_index(embeddings)
        db.add_metadata([{'id': i, 'name': text} for i, text in enumerate(test_texts)])
        
        print(f"✓ Database created with {len(test_texts)} items")
        
        # Save
        print("\n🧪 Testing save...")
        db.save(str(temp_dir))
        print("✓ Database saved")
        
        # Check files
        expected_files = ['faiss_index.bin', 'metadata.pkl', 'id_mapping.pkl', 'config.json']
        for filename in expected_files:
            filepath = temp_dir / filename
            if filepath.exists():
                print(f"  ✓ {filename} ({filepath.stat().st_size} bytes)")
            else:
                print(f"  ✗ {filename} NOT FOUND")
                return False
        
        # Load
        print("\n🧪 Testing load...")
        loaded_db = VectorDatabase.load(str(temp_dir))
        print("✓ Database loaded")
        print(f"  Embeddings: {loaded_db.index.ntotal}")
        print(f"  Metadata: {len(loaded_db.metadata)} items")
        
        # Test search on loaded database
        print("\n🧪 Testing search on loaded database...")
        results = loaded_db.search("Product", top_k=2)
        print(f"✓ Search works - returned {len(results)} results")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n🧹 Cleaned up temporary files")
        
        print("\n✅ Save/Load works correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing save/load: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("PHASE 2: VECTOR DATABASE - COMPREHENSIVE TESTING")
    print("=" * 80)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Phase 1 Output", test_phase1_output),
        ("Embedding Model", test_embedding_model),
        ("FAISS", test_faiss),
        ("VectorDatabase Class", test_vector_database_class),
        ("Save/Load", test_save_load)
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
            break
    
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
        print("\nYour Phase 2 setup is complete and working!")
        print("\nTo build your vector database:")
        print("   python vector_database.py")
        print("\nOr use interactively:")
        print("   from vector_database import build_vector_database_from_csv")
        print("   db = build_vector_database_from_csv('data/processed_data.csv')")
        return True
    else:
        print("\n" + "=" * 80)
        print("⚠️  SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease fix the failed tests before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)