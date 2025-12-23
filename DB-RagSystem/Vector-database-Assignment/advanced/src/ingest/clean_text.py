"""
Enhanced text cleaning for E-commerce Q&A
Handles noisy user-generated content with smart normalization
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter


@dataclass
class CleanQARecord:
    qa_id: str
    source_file: str
    row_num: int
    question: str              # Original
    answer: str                # Original
    question_clean: str        # Cleaned version
    answer_clean: str          # Cleaned version
    text_combined_clean: str   # Combined clean text for embedding
    metadata: Dict[str, Any]   # Additional stats


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


class EcommerceTextCleaner:
    """Advanced text cleaning for e-commerce Q&A"""
    
    def __init__(self):
        # Product name patterns to standardize
        self.product_patterns = {
            # iPhone variations
            r'\biphone\s*(\d+)\s*(plus|pro|max|mini)?\b': self._normalize_iphone,
            # Samsung Galaxy
            r'\bgalaxy\s*s(\d+)\b': lambda m: f'Galaxy S{m.group(1)}',
            # Generic model numbers
            r'\bmodel\s*[:#]?\s*([a-z0-9-]+)\b': lambda m: f'Model {m.group(1).upper()}',
        }
        
        # Common e-commerce abbreviations
        self.abbreviations = {
            r'\bgb\b': 'GB',
            r'\bmb\b': 'MB',
            r'\bhp\b': 'HP',
            r'\bupc\b': 'UPC',
            r'\bsku\b': 'SKU',
            r'\bqty\b': 'quantity',
            r'\bpls\b': 'please',
            r'\bthx\b': 'thanks',
            r'\bu\b': 'you',
            r'\br\b': 'are',
        }
    
    def _normalize_iphone(self, match):
        """Standardize iPhone naming"""
        num = match.group(1)
        suffix = match.group(2) or ''
        if suffix:
            return f'iPhone {num} {suffix.title()}'
        return f'iPhone {num}'
    
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            aggressive: Apply more aggressive cleaning
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        original = text
        
        # 1. Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # 2. Fix common encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        text = text.replace('Â', ' ')
        
        # 3. Normalize whitespace
        text = ' '.join(text.split())
        
        # 4. Fix punctuation spacing
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
        
        # 5. Expand common abbreviations
        for pattern, replacement in self.abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 6. Standardize product names
        for pattern, normalizer in self.product_patterns.items():
            text = re.sub(pattern, normalizer, text, flags=re.IGNORECASE)
        
        # 7. Remove excessive punctuation
        text = re.sub(r'([!?]){2,}', r'\1', text)
        text = re.sub(r'\.{4,}', '...', text)
        
        # 8. Normalize currency
        text = re.sub(r'\$\s+(\d+)', r'$\1', text)
        
        # 9. Aggressive cleaning (optional)
        if aggressive:
            # Remove URLs
            text = re.sub(r'http\S+|www\.\S+', '', text)
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            # Remove special characters (keep basic punctuation)
            text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # 10. Final cleanup
        text = ' '.join(text.split())
        text = text.strip()
        
        return text
    
    def extract_metadata(self, question: str, answer: str) -> Dict[str, Any]:
        """Extract useful metadata from Q&A pair"""
        metadata = {}
        
        # Length statistics
        metadata['q_char_len'] = len(question)
        metadata['a_char_len'] = len(answer)
        metadata['q_word_count'] = len(question.split()) if question else 0
        metadata['a_word_count'] = len(answer.split()) if answer else 0
        
        # Question type detection
        question_lower = question.lower()
        if any(q in question_lower for q in ['can i', 'will', 'does', 'is', 'are']):
            metadata['q_type'] = 'yes_no'
        elif any(q in question_lower for q in ['how', 'what', 'why', 'where', 'when']):
            metadata['q_type'] = 'wh_question'
        else:
            metadata['q_type'] = 'other'
        
        # Product mentions
        product_keywords = ['iphone', 'galaxy', 'case', 'screen', 'battery', 'charger']
        mentioned = [kw for kw in product_keywords if kw in question_lower or kw in answer.lower()]
        metadata['products_mentioned'] = mentioned
        
        # Sentiment hints (very basic)
        positive_words = ['good', 'great', 'perfect', 'love', 'excellent', 'yes']
        negative_words = ['no', 'not', 'bad', 'poor', 'terrible', 'wrong']
        
        answer_lower = answer.lower()
        pos_count = sum(1 for w in positive_words if w in answer_lower)
        neg_count = sum(1 for w in negative_words if w in answer_lower)
        
        if pos_count > neg_count:
            metadata['sentiment'] = 'positive'
        elif neg_count > pos_count:
            metadata['sentiment'] = 'negative'
        else:
            metadata['sentiment'] = 'neutral'
        
        return metadata


def main(
    in_path: str = "data/processed/qa_pairs.jsonl",
    out_path: str = "data/processed/qa_pairs_clean.jsonl",
) -> None:
    """
    Clean Q&A pairs with enhanced preprocessing
    """
    in_p = Path(in_path)
    out_p = Path(out_path)
    
    print("=" * 70)
    print("🧹 Cleaning E-commerce Q&A Text")
    print("=" * 70)
    
    if not in_p.exists():
        raise FileNotFoundError(f" Input file not found: {in_p.resolve()}")
    
    # Load Q&A pairs
    rows = read_jsonl(in_p)
    print(f"\n Loaded {len(rows):,} Q&A pairs")
    
    # Initialize cleaner
    cleaner = EcommerceTextCleaner()
    
    # Clean all records
    clean_rows = []
    empty_count = 0
    
    print(f"\n Cleaning text...")
    for row in rows:
        question = row.get('question', '')
        answer = row.get('answer', '')
        
        # Clean texts
        q_clean = cleaner.clean_text(question, aggressive=False)
        a_clean = cleaner.clean_text(answer, aggressive=False)
        
        # Skip if both are empty after cleaning
        if not q_clean and not a_clean:
            empty_count += 1
            continue
        
        # Combine for embedding
        combined_clean = f"Question: {q_clean}\nAnswer: {a_clean}" if q_clean and a_clean else (q_clean or a_clean)
        
        # Extract metadata
        metadata = cleaner.extract_metadata(q_clean, a_clean)
        
        # Create clean record
        rec = CleanQARecord(
            qa_id=row['qa_id'],
            source_file=row['source_file'],
            row_num=row['row_num'],
            question=question,
            answer=answer,
            question_clean=q_clean,
            answer_clean=a_clean,
            text_combined_clean=combined_clean,
            metadata=metadata,
        )
        
        clean_rows.append(asdict(rec))
    
    # Write cleaned data
    write_jsonl(out_p, clean_rows)
    
    # Statistics
    total = len(clean_rows)
    avg_q_len = sum(r['metadata']['q_char_len'] for r in clean_rows) / max(1, total)
    avg_a_len = sum(r['metadata']['a_char_len'] for r in clean_rows) / max(1, total)
    
    # Question types distribution
    q_types = Counter(r['metadata']['q_type'] for r in clean_rows)
    
    print("\n" + "=" * 70)
    print(" Cleaning Complete!")
    print("=" * 70)
    print(f"\n Statistics:")
    print(f"   Input records:       {len(rows):,}")
    print(f"   Cleaned records:     {total:,}")
    print(f"   Removed (empty):     {empty_count:,}")
    print(f"\nAverage Lengths:")
    print(f"   Question:            {avg_q_len:.0f} chars")
    print(f"   Answer:              {avg_a_len:.0f} chars")
    print(f"\n Question Types:")
    for q_type, count in q_types.most_common():
        print(f"   {q_type:15s}: {count:,} ({count/total*100:.1f}%)")
    print(f"\n Output:")
    print(f"   {out_p.resolve()}")


if __name__ == "__main__":
    main()