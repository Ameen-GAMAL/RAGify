"""
Quick Test Script for Phase 1
Run this to validate your Phase 1 implementation
"""

import sys
from pathlib import Path

def test_phase1():
    """Test Phase 1 setup and execution"""
    
    print("=" * 80)
    print("TESTING PHASE 1 SETUP")
    print("=" * 80)
    
    # Test 1: Check file structure
    print("\n✓ Test 1: Checking directory structure...")
    required_dirs = ['data', 'outputs']
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"   ✓ {dir_name}/ exists")
        else:
            print(f"   ✗ {dir_name}/ missing - creating it...")
            Path(dir_name).mkdir(exist_ok=True)
    
    # Test 2: Check dataset file
    print("\n✓ Test 2: Checking for dataset file...")
    data_dir = Path('data')
    csv_files = list(data_dir.glob('*.csv'))
    
    if csv_files:
        print(f"   ✓ Found {len(csv_files)} CSV file(s):")
        for f in csv_files:
            print(f"      - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
        dataset_path = csv_files[0]
    else:
        print("   ✗ No CSV files found in data/ directory")
        print("   → Please download the dataset from Kaggle and place it in data/")
        return False
    
    # Test 3: Check dependencies
    print("\n✓ Test 3: Checking dependencies...")
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✓ {package} installed")
        except ImportError:
            print(f"   ✗ {package} NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   → Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    # Test 4: Try importing the DataProcessor
    print("\n✓ Test 4: Testing DataProcessor import...")
    try:
        from src.phase1_data_processing import DataProcessor
        print("   ✓ DataProcessor imported successfully")
    except Exception as e:
        print(f"   ✗ Error importing DataProcessor: {e}")
        return False
    
    # Test 5: Quick data load test
    print("\n✓ Test 5: Testing data loading...")
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path, nrows=100)  # Just load 100 rows for testing
        print(f"   ✓ Successfully loaded sample data")
        print(f"   ✓ Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   ✓ Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return False
    
    # All tests passed
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nYour Phase 1 setup is ready!")
    print("\nTo run the full pipeline:")
    print("   python data_processing.py")
    print("\nOr use it interactively:")
    print("   from data_processing import DataProcessor")
    print(f"   processor = DataProcessor('{dataset_path}')")
    print("   processor.load_data()")
    
    return True


def quick_run():
    """Quick test run of the data processor"""
    print("\n" + "=" * 80)
    print("QUICK TEST RUN")
    print("=" * 80)
    
    try:
        from src.phase1_data_processing import DataProcessor
        
        # Find dataset
        data_dir = Path('data')
        csv_files = list(data_dir.glob('*.csv'))
        
        if not csv_files:
            print("No dataset found. Please run test_phase1() first.")
            return
        
        dataset_path = csv_files[0]
        
        # Initialize processor
        processor = DataProcessor(str(dataset_path))
        
        # Quick test
        print("\nLoading data...")
        df = processor.load_data()
        
        print("\nRunning quick exploration...")
        print(f"✓ Loaded {len(df):,} rows")
        print(f"✓ Columns: {list(df.columns[:5])} ...")
        
        print("\n✅ Quick test successful!")
        print("\nTo run the full pipeline, use: python data_processing.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests
    if test_phase1():
        # If user wants, run quick test
        print("\n" + "=" * 80)
        response = input("\nRun a quick test? (y/n): ").lower()
        if response == 'y':
            quick_run()
    else:
        print("\n⚠️ Please fix the issues above before proceeding.")
        sys.exit(1)