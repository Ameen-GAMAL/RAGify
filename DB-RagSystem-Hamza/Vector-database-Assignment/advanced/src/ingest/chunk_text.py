"""
Chunking strategy for Q&A pairs
Strategy: 1 Q&A pair = 1 chunk (no splitting needed)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class ChunkRecord:
    chunk_id: str              # qa_0001_c0001
    text: str                  # Combined clean text for embedding
    metadata: Dict[str, Any]   # Includes qa_id, question, answer, etc.


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write JSONL file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(
    in_path: str = "data/processed/qa_pairs_clean.jsonl",
    out_path: str = "data/processed/chunks.jsonl",
) -> None:
    """
    Create chunks from Q&A pairs
    
    For Q&A systems: 1 Q&A pair = 1 chunk
    This is optimal for semantic search because:
    - Each question-answer is a complete, self-contained unit
    - No need to split or merge
    - Preserves full context
    - Enables precise matching
    """
    in_p = Path(in_path)
    out_p = Path(out_path)
    
    print("=" * 70)
    print(" Creating Chunks from Q&A Pairs")
    print("=" * 70)
    
    if not in_p.exists():
        raise FileNotFoundError(f" Input file not found: {in_p.resolve()}")
    
    # Load cleaned Q&A pairs
    rows = read_jsonl(in_p)
    print(f"\n Loaded {len(rows):,} cleaned Q&A pairs")
    
    # Create chunks (1 Q&A = 1 chunk)
    chunks = []
    
    print(f"\n Creating chunks...")
    for idx, row in enumerate(rows, start=1):
        # Each Q&A pair becomes one chunk
        chunk_id = f"{row['qa_id']}_c0001"  # Always c0001 since no splitting
        
        # Use combined clean text for embedding
        text = row['text_combined_clean']
        
        # Build comprehensive metadata
        metadata = {
            # Original identifiers
            'qa_id': row['qa_id'],
            'source_file': row['source_file'],
            'row_num': row['row_num'],
            
            # Original texts (for display)
            'question': row['question_clean'],
            'answer': row['answer_clean'],
            
            # Statistics from cleaning
            'char_len': len(text),
            'word_count': len(text.split()),
            'q_char_len': row['metadata']['q_char_len'],
            'a_char_len': row['metadata']['a_char_len'],
            
            # Semantic info
            'q_type': row['metadata']['q_type'],
            'products_mentioned': row['metadata']['products_mentioned'],
            'sentiment': row['metadata']['sentiment'],
            
            # For retrieval filtering
            'chunk_index': 1,  # Always 1 (no splitting)
            'total_chunks': 1,  # Always 1 (no splitting)
        }
        
        chunk = ChunkRecord(
            chunk_id=chunk_id,
            text=text,
            metadata=metadata,
        )
        
        chunks.append(asdict(chunk))
    
    # Write chunks
    write_jsonl(out_p, chunks)
    
    # Statistics
    total_chunks = len(chunks)
    avg_len = sum(c['metadata']['char_len'] for c in chunks) / max(1, total_chunks)
    avg_words = sum(c['metadata']['word_count'] for c in chunks) / max(1, total_chunks)
    
    # Length distribution
    lengths = [c['metadata']['char_len'] for c in chunks]
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    
    print("\n" + "=" * 70)
    print(" Chunking Complete!")
    print("=" * 70)
    print(f"\n Statistics:")
    print(f"   Total chunks:        {total_chunks:,}")
    print(f"   Chunks per Q&A:      1 (no splitting)")
    print(f"\n Chunk Sizes:")
    print(f"   Average length:      {avg_len:.0f} chars ({avg_words:.0f} words)")
    print(f"   Min length:          {min_len:,} chars")
    print(f"   Max length:          {max_len:,} chars")
    print(f"\n Strategy:")
    print(f"   • 1 Q&A pair = 1 chunk (optimal for Q&A retrieval)")
    print(f"   • Preserves complete context")
    print(f"   • No information loss from splitting")
    print(f"\n Output:")
    print(f"   {out_p.resolve()}")
    
    # Sample chunks
    print(f"\n🔍 Sample Chunks (first 3):")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n   [{i}] {chunk['chunk_id']}")
        print(f"       Q: {chunk['metadata']['question'][:60]}...")
        print(f"       A: {chunk['metadata']['answer'][:60]}...")
        print(f"       Length: {chunk['metadata']['char_len']} chars")


if __name__ == "__main__":
    main()