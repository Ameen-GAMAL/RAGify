"""
RAG System - Phase 1: Data Processing & Exploration
Advanced Information Retrieval Course
Dataset: Google Shopping (Toy Products)

This module handles data loading, cleaning, preprocessing, and exploration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import re
from typing import Dict, List, Tuple
import json
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Comprehensive data processor for Google Shopping dataset
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the DataProcessor
        
        Args:
            data_path: Path to the CSV dataset file (relative to project root)
        """
        self.data_path = PROJECT_ROOT / data_path
        self.df = None
        self.df_clean = None
        self.stats = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load the dataset from CSV file"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.data_path, encoding=encoding)
                    print(f"✓ Data loaded successfully with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df is None:
                raise Exception("Could not load data with any encoding")
            
            print(f"✓ Dataset shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            print(f"✓ File size: {self.data_path.stat().st_size / (1024*1024):.2f} MB")
            
            return self.df
            
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            raise
    
    def explore_data(self) -> Dict:
        """Perform comprehensive data exploration"""
        print("\n" + "=" * 80)
        print("DATA EXPLORATION")
        print("=" * 80)
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Basic info
        print("\n📊 Dataset Overview:")
        print(f"   - Total Records: {len(self.df):,}")
        print(f"   - Total Columns: {len(self.df.columns)}")
        print(f"   - Memory Usage: {self.df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        
        # Column information
        print("\n📋 Column Information:")
        print("-" * 80)
        for col in self.df.columns:
            dtype = self.df[col].dtype
            non_null = self.df[col].count()
            null_pct = (len(self.df) - non_null) / len(self.df) * 100
            unique = self.df[col].nunique()
            
            print(f"   {col:30} | Type: {str(dtype):10} | Non-Null: {non_null:,} ({100-null_pct:.1f}%) | Unique: {unique:,}")
        
        # Missing values analysis
        print("\n🔍 Missing Values Analysis:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            missing_pct = (missing / len(self.df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Missing Count': missing[missing > 0],
                'Percentage': missing_pct[missing > 0]
            }).sort_values('Percentage', ascending=False)
            print(missing_df)
        else:
            print("   ✓ No missing values found!")
        
        # Display sample data
        print("\n📝 Sample Data (First 5 rows):")
        print("-" * 80)
        print(self.df.head())
        
        # Store statistics
        self.stats['total_records'] = len(self.df)
        self.stats['total_columns'] = len(self.df.columns)
        self.stats['missing_values'] = missing.to_dict()
        self.stats['data_types'] = self.df.dtypes.to_dict()
        
        return self.stats
    
    def analyze_text_fields(self) -> Dict:
        """Analyze text fields for RAG system preparation"""
        print("\n" + "=" * 80)
        print("TEXT FIELDS ANALYSIS (For RAG)")
        print("=" * 80)
        
        text_stats = {}
        text_columns = self.df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            # Skip if mostly empty
            if self.df[col].isnull().sum() / len(self.df) > 0.9:
                continue
            
            print(f"\n📝 Column: '{col}'")
            print("-" * 80)
            
            # Get non-null values
            non_null_values = self.df[col].dropna()
            
            if len(non_null_values) == 0:
                print("   ⚠ No valid data")
                continue
            
            # Calculate text statistics
            lengths = non_null_values.astype(str).str.len()
            word_counts = non_null_values.astype(str).str.split().str.len()
            
            stats = {
                'count': len(non_null_values),
                'avg_length': lengths.mean(),
                'max_length': lengths.max(),
                'min_length': lengths.min(),
                'avg_words': word_counts.mean(),
                'max_words': word_counts.max()
            }
            
            print(f"   - Valid entries: {stats['count']:,}")
            print(f"   - Avg character length: {stats['avg_length']:.1f}")
            print(f"   - Length range: {stats['min_length']:.0f} - {stats['max_length']:.0f}")
            print(f"   - Avg word count: {stats['avg_words']:.1f}")
            print(f"   - Max words: {stats['max_words']:.0f}")
            
            # Show samples
            print(f"\n   Sample values:")
            for i, sample in enumerate(non_null_values.head(3), 1):
                sample_str = str(sample)[:100] + "..." if len(str(sample)) > 100 else str(sample)
                print(f"      {i}. {sample_str}")
            
            text_stats[col] = stats
        
        return text_stats
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess the data"""
        print("\n" + "=" * 80)
        print("DATA CLEANING & PREPROCESSING")
        print("=" * 80)
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.df_clean = self.df.copy()
        cleaning_steps = []
        
        # 1. Remove duplicates
        initial_rows = len(self.df_clean)
        self.df_clean = self.df_clean.drop_duplicates()
        duplicates_removed = initial_rows - len(self.df_clean)
        if duplicates_removed > 0:
            print(f"✓ Removed {duplicates_removed} duplicate rows")
            cleaning_steps.append(f"Removed {duplicates_removed} duplicates")
        else:
            print("✓ No duplicate rows found")
        
        # 2. Clean text columns
        text_columns = self.df_clean.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            # Remove leading/trailing whitespace
            self.df_clean[col] = self.df_clean[col].astype(str).str.strip()
            
            # Replace 'nan' string with actual NaN
            self.df_clean[col] = self.df_clean[col].replace(['nan', 'NaN', 'None', ''], np.nan)
        
        print(f"✓ Cleaned {len(text_columns)} text columns")
        cleaning_steps.append(f"Cleaned {len(text_columns)} text columns")
        
        # 3. Handle missing values in numeric columns
        numeric_columns = self.df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            for col in numeric_columns:
                if self.df_clean[col].isnull().sum() > 0:
                    median_val = self.df_clean[col].median()
                    self.df_clean[col].fillna(median_val, inplace=True)
            print(f"✓ Filled missing values in {len(numeric_columns)} numeric columns")
            cleaning_steps.append(f"Filled numeric missing values")
        
        # 4. Remove rows with ALL missing values
        initial_rows = len(self.df_clean)
        self.df_clean = self.df_clean.dropna(how='all')
        all_missing_removed = initial_rows - len(self.df_clean)
        if all_missing_removed > 0:
            print(f"✓ Removed {all_missing_removed} rows with all missing values")
            cleaning_steps.append(f"Removed {all_missing_removed} completely empty rows")
        
        # Summary
        print("\n" + "=" * 80)
        print(f"✓ CLEANING COMPLETE")
        print(f"   Original rows: {len(self.df):,}")
        print(f"   Clean rows: {len(self.df_clean):,}")
        print(f"   Rows removed: {len(self.df) - len(self.df_clean):,}")
        print("=" * 80)
        
        self.stats['cleaning_steps'] = cleaning_steps
        self.stats['original_rows'] = len(self.df)
        self.stats['clean_rows'] = len(self.df_clean)
        
        return self.df_clean
    
    def create_combined_text(self, columns_to_combine: List[str] = None) -> pd.Series:
        """
        Create combined text for embedding generation
        
        Args:
            columns_to_combine: List of column names to combine. If None, auto-detect
            
        Returns:
            Series with combined text for each product
        """
        print("\n" + "=" * 80)
        print("CREATING COMBINED TEXT FOR EMBEDDINGS")
        print("=" * 80)
        
        if self.df_clean is None:
            raise ValueError("Data not cleaned. Call clean_data() first.")
        
        # Auto-detect text columns if not specified
        if columns_to_combine is None:
            # Prioritize columns likely to contain rich information
            priority_keywords = ['title', 'name', 'description', 'brand', 'category', 
                                'manufacturer', 'features', 'specifications']
            
            columns_to_combine = []
            for col in self.df_clean.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in priority_keywords):
                    # Check if column has substantial data
                    non_null_pct = self.df_clean[col].count() / len(self.df_clean)
                    if non_null_pct > 0.5:  # At least 50% non-null
                        columns_to_combine.append(col)
        
        print(f"Combining columns: {columns_to_combine}")
        
        # Create combined text
        def combine_row(row):
            parts = []
            for col in columns_to_combine:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    parts.append(f"{col}: {str(value).strip()}")
            return " | ".join(parts)
        
        combined_text = self.df_clean.apply(combine_row, axis=1)
        
        # Statistics
        avg_length = combined_text.str.len().mean()
        print(f"\n✓ Combined text created for {len(combined_text):,} products")
        print(f"✓ Average combined text length: {avg_length:.0f} characters")
        
        # Show sample
        print("\n📝 Sample combined text:")
        print("-" * 80)
        print(combined_text.iloc[0][:500] + "..." if len(combined_text.iloc[0]) > 500 else combined_text.iloc[0])
        
        return combined_text
    
    def visualize_data_quality(self, save_dir: str = "outputs/visualizations"):
        """Create visualizations for data quality analysis"""
        print("\n" + "=" * 80)
        print("GENERATING DATA QUALITY VISUALIZATIONS")
        print("=" * 80)
        
        save_path = PROJECT_ROOT / save_dir
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Missing values heatmap
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            missing_pct = (missing / len(self.df) * 100)
            missing_pct[missing_pct > 0].sort_values(ascending=True).plot(
                kind='barh', ax=axes[0, 0], color='coral'
            )
            axes[0, 0].set_title('Missing Values by Column (%)', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Percentage Missing')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values!', 
                           ha='center', va='center', fontsize=14, fontweight='bold')
            axes[0, 0].set_title('Missing Values Analysis')
        
        # Data types distribution
        dtype_counts = self.df.dtypes.value_counts()
        axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%',
                       startangle=90, colors=sns.color_palette("Set2"))
        axes[0, 1].set_title('Data Types Distribution', fontsize=12, fontweight='bold')
        
        # Duplicate rows
        duplicate_count = self.df.duplicated().sum()
        unique_count = len(self.df) - duplicate_count
        axes[1, 0].bar(['Unique', 'Duplicates'], [unique_count, duplicate_count],
                       color=['skyblue', 'coral'])
        axes[1, 0].set_title('Duplicate Rows Analysis', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Count')
        
        # Records before/after cleaning
        if self.df_clean is not None:
            axes[1, 1].bar(['Original', 'After Cleaning'], 
                          [len(self.df), len(self.df_clean)],
                          color=['lightgray', 'lightgreen'])
            axes[1, 1].set_title('Data Cleaning Impact', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Number of Records')
        
        plt.tight_layout()
        viz_path = save_path / 'data_quality_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization: {viz_path}")
        plt.close()
    
    def save_processed_data(self, output_path: str = "data/processed/processed_data.csv"):
        """Save the cleaned and processed data"""
        if self.df_clean is None:
            raise ValueError("No cleaned data to save. Run clean_data() first.")
        
        output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df_clean.to_csv(output_path, index=False)
        print(f"\n✓ Processed data saved to: {output_path}")
        print(f"✓ File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    def generate_report(self, output_path: str = "outputs/reports/phase1_report.txt"):
        """Generate a comprehensive report of Phase 1"""
        output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RAG SYSTEM - PHASE 1 REPORT\n")
            f.write("Data Processing & Exploration\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Records: {self.stats.get('original_rows', 'N/A'):,}\n")
            f.write(f"Total Columns: {self.stats.get('total_columns', 'N/A')}\n")
            f.write(f"Records after cleaning: {self.stats.get('clean_rows', 'N/A'):,}\n\n")
            
            f.write("CLEANING STEPS\n")
            f.write("-" * 80 + "\n")
            for step in self.stats.get('cleaning_steps', []):
                f.write(f"- {step}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"\n✓ Phase 1 report saved to: {output_path}")


def main():
    """Main execution function for Phase 1"""
    print("=" * 80)
    print("RAG SYSTEM - PHASE 1: DATA PROCESSING & EXPLORATION")
    print("Advanced Information Retrieval Course")
    print("=" * 80)
    
    # Initialize processor
    # UPDATE THIS PATH WITH YOUR ACTUAL DATA FILE NAME
    data_path = "data/raw/google_shopping_dataset.csv"
    
    processor = DataProcessor(data_path)
    
    # Execute pipeline
    try:
        # 1. Load data
        processor.load_data()
        
        # 2. Explore data
        processor.explore_data()
        
        # 3. Analyze text fields
        processor.analyze_text_fields()
        
        # 4. Clean data
        processor.clean_data()
        
        # 5. Create combined text for embeddings
        combined_text = processor.create_combined_text()
        
        # 6. Visualize data quality
        processor.visualize_data_quality()
        
        # 7. Save processed data
        processor.save_processed_data()
        
        # 8. Generate report
        processor.generate_report()
        
        print("\n" + "=" * 80)
        print("✅ PHASE 1 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nNext Steps:")
        print("1. Review the data quality visualization in 'outputs/visualizations/data_quality_analysis.png'")
        print("2. Check the processed data in 'data/processed/processed_data.csv'")
        print("3. Read the full report in 'outputs/reports/phase1_report.txt'")
        print("4. When ready, proceed to Phase 2: Vector Database Construction")
        
    except Exception as e:
        print(f"\n❌ Error in Phase 1: {str(e)}")
        raise


if __name__ == "__main__":
    main()