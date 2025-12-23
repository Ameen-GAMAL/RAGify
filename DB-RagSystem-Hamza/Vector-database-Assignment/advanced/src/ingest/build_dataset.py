"""
Complete E-commerce Q&A Dataset Building Pipeline
Orchestrates: Extract → Clean → Chunk → Generate Manifest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

# Import pipeline stages
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingest.extract_text import main as extract_main
from src.ingest.clean_text import main as clean_main
from src.ingest.chunk_text import main as chunk_main


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON file with pretty formatting"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def compute_statistics(
    qa_pairs: List[Dict],
    qa_clean: List[Dict],
    chunks: List[Dict]
) -> Dict[str, Any]:
    """Compute comprehensive dataset statistics"""
    
    stats = {}
    
    # Basic counts
    stats['total_qa_pairs'] = len(qa_pairs)
    stats['total_clean_pairs'] = len(qa_clean)
    stats['total_chunks'] = len(chunks)
    stats['empty_removed'] = len(qa_pairs) - len(qa_clean)
    
    # Question statistics
    if qa_clean:
        questions = [r['question_clean'] for r in qa_clean if r.get('question_clean')]
        stats['avg_question_length'] = sum(len(q) for q in questions) / max(1, len(questions))
        stats['questions_present'] = len(questions)
    
    # Answer statistics
    if qa_clean:
        answers = [r['answer_clean'] for r in qa_clean if r.get('answer_clean')]
        stats['avg_answer_length'] = sum(len(a) for a in answers) / max(1, len(answers))
        stats['answers_present'] = len(answers)
    
    # Chunk statistics
    if chunks:
        chunk_lengths = [c['metadata']['char_len'] for c in chunks]
        stats['avg_chunk_length'] = sum(chunk_lengths) / len(chunk_lengths)
        stats['min_chunk_length'] = min(chunk_lengths)
        stats['max_chunk_length'] = max(chunk_lengths)
        
        # Question type distribution
        q_types = {}
        for c in chunks:
            q_type = c['metadata'].get('q_type', 'unknown')
            q_types[q_type] = q_types.get(q_type, 0) + 1
        stats['question_types'] = q_types
        
        # Sentiment distribution
        sentiments = {}
        for c in chunks:
            sent = c['metadata'].get('sentiment', 'neutral')
            sentiments[sent] = sentiments.get(sent, 0) + 1
        stats['sentiment_distribution'] = sentiments
    
    # Round floats
    for key, val in stats.items():
        if isinstance(val, float):
            stats[key] = round(val, 2)
    
    return stats


def main() -> None:
    """
    Run the complete Q&A ingestion pipeline
    """
    print("\n" + "=" * 70)
    print(" E-COMMERCE Q&A DATASET BUILDER")
    print("=" * 70)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define paths
    paths = {
        'raw_csv': Path("data/raw_csv"),
        'qa_pairs': Path("data/processed/qa_pairs.jsonl"),
        'qa_clean': Path("data/processed/qa_pairs_clean.jsonl"),
        'chunks': Path("data/processed/chunks.jsonl"),
        'manifest': Path("data/processed/manifest.json"),
    }
    
    # Check if raw CSV directory exists
    if not paths['raw_csv'].exists():
        print(f"\n Warning: {paths['raw_csv']} doesn't exist!")
        print(f"   Creating directory...")
        paths['raw_csv'].mkdir(parents=True, exist_ok=True)
        print(f"   Please place your CSV file(s) in: {paths['raw_csv'].resolve()}")
        print(f"   Expected columns: 'Question' and 'Answer'")
        return
    
    try:
        # Stage 1: Extract Q&A pairs from CSV
        print("\n" + "─" * 70)
        print(" STAGE 1/3: Extracting Q&A Pairs from CSV")
        print("─" * 70)
        extract_main(
            raw_dir=str(paths['raw_csv']),
            out_path=str(paths['qa_pairs'])
        )
        
        # Stage 2: Clean and normalize text
        print("\n" + "─" * 70)
        print(" STAGE 2/3: Cleaning and Normalizing Text")
        print("─" * 70)
        clean_main(
            in_path=str(paths['qa_pairs']),
            out_path=str(paths['qa_clean'])
        )
        
        # Stage 3: Create chunks (1 Q&A = 1 chunk)
        print("\n" + "─" * 70)
        print(" STAGE 3/3: Creating Searchable Chunks")
        print("─" * 70)
        chunk_main(
            in_path=str(paths['qa_clean']),
            out_path=str(paths['chunks'])
        )
        
        # Load processed data for statistics
        qa_pairs = read_jsonl(paths['qa_pairs'])
        qa_clean = read_jsonl(paths['qa_clean'])
        chunks = read_jsonl(paths['chunks'])
        
        # Compute statistics
        stats = compute_statistics(qa_pairs, qa_clean, chunks)
        
        # Sample Q&A pairs
        sample_pairs = []
        for qa in qa_clean[:5]:
            sample_pairs.append({
                'qa_id': qa['qa_id'],
                'question': qa['question_clean'][:100],
                'answer': qa['answer_clean'][:100],
            })
        
        # Create manifest
        manifest = {
            'dataset_info': {
                'name': 'E-commerce Q&A Dataset',
                'type': 'question_answer',
                'created': datetime.now().isoformat(),
                'description': 'Product questions and answers for RAG system',
            },
            'paths': {
                'raw_csv_dir': str(paths['raw_csv'].resolve()),
                'processed_dir': str(paths['qa_pairs'].parent.resolve()),
                'qa_pairs': str(paths['qa_pairs'].resolve()),
                'qa_pairs_clean': str(paths['qa_clean'].resolve()),
                'chunks': str(paths['chunks'].resolve()),
            },
            'statistics': stats,
            'sample_pairs': sample_pairs,
            'next_steps': {
                '1': 'Build FAISS index: python -m src.rag.build_index',
                '2': 'Test retrieval: python -m src.rag.retriever',
                '3': 'Launch UI: streamlit run src/app/ui_streamlit.py',
            },
            'notes': {
                'chunking_strategy': '1 Q&A pair = 1 chunk (optimal for Q&A retrieval)',
                'model_recommendation': 'sentence-transformers/all-mpnet-base-v2',
                'feedback_learning': 'Enable with feedback_store.py for continuous improvement',
            }
        }
        
        # Write manifest
        write_json(paths['manifest'], manifest)
        
        # Final summary
        print("\n" + "=" * 70)
        print(" PIPELINE COMPLETE!")
        print("=" * 70)
        
        print(f"\n FINAL STATISTICS:")
        print(f"   ├─ Total Q&A pairs:      {stats['total_qa_pairs']:,}")
        print(f"   ├─ Clean pairs:          {stats['total_clean_pairs']:,}")
        print(f"   ├─ Empty removed:        {stats['empty_removed']:,}")
        print(f"   └─ Total chunks:         {stats['total_chunks']:,}")
        
        print(f"\n AVERAGE LENGTHS:")
        print(f"   ├─ Question:             {stats.get('avg_question_length', 0):.0f} chars")
        print(f"   ├─ Answer:               {stats.get('avg_answer_length', 0):.0f} chars")
        print(f"   └─ Chunk:                {stats.get('avg_chunk_length', 0):.0f} chars")
        
        print(f"\n QUESTION TYPES:")
        for q_type, count in stats.get('question_types', {}).items():
            pct = (count / stats['total_chunks'] * 100) if stats['total_chunks'] > 0 else 0
            print(f"   ├─ {q_type:15s}: {count:,} ({pct:.1f}%)")
        
        print(f"\n OUTPUT FILES:")
        print(f"   ├─ {paths['qa_pairs'].name}")
        print(f"   ├─ {paths['qa_clean'].name}")
        print(f"   ├─ {paths['chunks'].name}")
        print(f"   └─ {paths['manifest'].name}")
        
        print(f"\n MANIFEST:")
        print(f"   {paths['manifest'].resolve()}")
        
        print(f"\n NEXT STEPS:")
        print(f"     Build FAISS index:")
        print(f"       python -m src.rag.build_index")
        print(f"     Test retrieval:")
        print(f"       python -m src.rag.retriever")
        print(f"     Launch UI:")
        print(f"       streamlit run src/app/ui_streamlit.py")
        
        print(f"\n Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()