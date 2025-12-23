"""
Extract Q&A pairs from E-commerce CSV
Handles 14K+ records with robust encoding detection
"""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Dict, Any, List

from tqdm import tqdm


@dataclass
class QARecord:
    qa_id: str              # Unique ID: qa_0001, qa_0002, etc.
    source_file: str        # Source CSV filename
    row_num: int            # Row number in CSV
    question: str           # Original question
    answer: str             # Original answer
    text_combined: str      # Combined Q+A for embedding
    char_count: int         # Character count
    word_count: int         # Word count
    is_empty: bool          # Flag for empty records


def iter_csv_paths(raw_dir: Path) -> List[Path]:
    """Find all CSV files in the raw directory"""
    return sorted([p for p in raw_dir.glob("*.csv") if p.is_file()])


def qa_id_from_row(filename: str, row_num: int) -> str:
    """Generate unique QA ID: ecommerce_qa0001"""
    stem = Path(filename).stem
    return f"{stem}_qa{row_num:04d}"


def combine_qa_text(question: str, answer: str) -> str:
    """
    Combine question and answer for embedding.
    
    Strategy:
    - Include BOTH Q and A for semantic search
    - Allows matching on question similarity OR answer content
    - Better for paraphrased/similar questions
    """
    parts = []
    
    if question and question.strip():
        parts.append(f"Question: {question.strip()}")
    
    if answer and answer.strip():
        parts.append(f"Answer: {answer.strip()}")
    
    return "\n".join(parts) if parts else ""


def clean_field(text: str) -> str:
    """Basic cleaning for CSV fields"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove null bytes (common in corrupted CSVs)
    text = text.replace('\x00', '')
    
    return text.strip()


def extract_qa_pairs(csv_path: Path) -> Iterable[QARecord]:
    """
    Extract Q&A pairs from CSV file with robust error handling
    Supports multiple encodings and CSV formats
    """
    
    # Try different encodings (common in web data)
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with csv_path.open("r", encoding=encoding, errors='replace') as f:
                # Detect CSV dialect
                sample = f.read(8192)
                f.seek(0)
                
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    has_header = csv.Sniffer().has_header(sample)
                except csv.Error:
                    # Fallback to default
                    dialect = csv.excel
                    has_header = True
                
                # Read CSV
                if has_header:
                    reader = csv.DictReader(f, dialect=dialect)
                else:
                    # Assume: Column A = Question, Column B = Answer
                    reader = csv.DictReader(f, fieldnames=['Question', 'Answer'], dialect=dialect)
                
                row_count = 0
                for i, row in enumerate(reader, start=1):
                    # Handle different column name variations
                    question = (
                        row.get('Question') or 
                        row.get('question') or 
                        row.get('Q') or 
                        row.get('q') or 
                        ""
                    )
                    
                    answer = (
                        row.get('Answer') or 
                        row.get('answer') or 
                        row.get('A') or 
                        row.get('a') or 
                        ""
                    )
                    
                    # Clean fields
                    question = clean_field(question)
                    answer = clean_field(answer)
                    
                    # Skip completely empty rows
                    if not question and not answer:
                        continue
                    
                    # Combine for embedding
                    text_combined = combine_qa_text(question, answer)
                    char_count = len(text_combined)
                    word_count = len(text_combined.split()) if text_combined else 0
                    
                    row_count += 1
                    
                    yield QARecord(
                        qa_id=qa_id_from_row(csv_path.name, row_count),
                        source_file=csv_path.name,
                        row_num=row_count,
                        question=question,
                        answer=answer,
                        text_combined=text_combined,
                        char_count=char_count,
                        word_count=word_count,
                        is_empty=(char_count == 0),
                    )
                
                # If we successfully read rows, break
                if row_count > 0:
                    break
                
        except UnicodeDecodeError:
            continue  # Try next encoding
        except Exception as e:
            print(f"  Warning: Error reading {csv_path.name} with {encoding}: {e}")
            continue
    else:
        # None of the encodings worked
        raise ValueError(f" Could not read {csv_path} with any encoding")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write records to JSONL format"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(
    raw_dir: str = "data/raw_csv",
    out_path: str = "data/processed/qa_pairs.jsonl",
) -> None:
    """
    Main extraction pipeline for E-commerce Q&A CSV
    Handles 14K+ records with progress tracking
    """
    raw_dir_p = Path(raw_dir)
    out_path_p = Path(out_path)

    print("=" * 70)
    print(" Extracting E-commerce Q&A Pairs")
    print("=" * 70)

    # Find CSV files
    csvs = iter_csv_paths(raw_dir_p)
    if not csvs:
        raise FileNotFoundError(f" No CSV files found in: {raw_dir_p.resolve()}")

    print(f"\n Found {len(csvs)} CSV file(s)")
    for csv_file in csvs:
        print(f"   - {csv_file.name}")

    all_rows: List[Dict[str, Any]] = []
    
    print(f"\n Processing CSV files...")
    
    for csv_file in tqdm(csvs, desc="Extracting Q&A pairs"):
        count = 0
        try:
            for rec in extract_qa_pairs(csv_file):
                all_rows.append(asdict(rec))
                count += 1
            print(f"    {csv_file.name}: {count:,} Q&A pairs")
        except Exception as e:
            print(f"    {csv_file.name}: Failed - {e}")
            continue

    # Write to JSONL
    write_jsonl(out_path_p, all_rows)
    
    # Statistics
    total = len(all_rows)
    non_empty = sum(1 for r in all_rows if not r['is_empty'])
    avg_chars = sum(r['char_count'] for r in all_rows) / max(1, total)
    avg_words = sum(r['word_count'] for r in all_rows) / max(1, total)
    
    # Question statistics
    questions_present = sum(1 for r in all_rows if r['question'])
    avg_q_len = sum(len(r['question']) for r in all_rows if r['question']) / max(1, questions_present)
    
    # Answer statistics
    answers_present = sum(1 for r in all_rows if r['answer'])
    avg_a_len = sum(len(r['answer']) for r in all_rows if r['answer']) / max(1, answers_present)
    
    print("\n" + "=" * 70)
    print(" Extraction Complete!")
    print("=" * 70)
    print(f"\n Statistics:")
    print(f"   Total Q&A pairs:     {total:,}")
    print(f"   Non-empty pairs:     {non_empty:,}")
    print(f"   Empty pairs:         {total - non_empty:,}")
    print(f"\n Length Stats:")
    print(f"   Avg combined length: {avg_chars:.0f} chars ({avg_words:.0f} words)")
    print(f"   Avg question length: {avg_q_len:.0f} chars")
    print(f"   Avg answer length:   {avg_a_len:.0f} chars")
    print(f"\n Output:")
    print(f"   {out_path_p.resolve()}")
    
    # Sample preview
    print(f"\n Sample Q&A Pairs (first 3):")
    for i, r in enumerate(all_rows[:3], 1):
        print(f"\n   [{i}] {r['qa_id']}")
        print(f"       Q: {r['question'][:80]}...")
        print(f"       A: {r['answer'][:80]}...")


if __name__ == "__main__":
    main()