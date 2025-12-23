"""
Self-Learning Feedback System for RAG
Implements pairwise learning from user feedback
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


@dataclass(frozen=True)
class FeedbackPaths:
    """Paths for feedback storage"""
    feedback_json: Path = Path("data/processed/feedback.json")
    boosts_json: Path = Path("data/processed/boosts.json")


@dataclass
class FeedbackRecord:
    """Single feedback entry"""
    timestamp: str
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    rating: int  # +1 (helpful), -1 (not helpful)
    comment: str


def make_feedback_record(
    query: str,
    answer: str,
    sources: List[Dict[str, Any]],
    rating: int,
    comment: str = "",
) -> FeedbackRecord:
    """Create a new feedback record"""
    return FeedbackRecord(
        timestamp=datetime.now().isoformat(),
        query=query,
        answer=answer,
        sources=sources,
        rating=rating,
        comment=comment,
    )


def read_json(path: Path) -> Dict[str, Any]:
    """Read JSON file, return empty dict if not exists"""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON file with pretty formatting"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def append_feedback(paths: FeedbackPaths, record: FeedbackRecord) -> None:
    """
    Append feedback to storage
    
    Args:
        paths: Feedback storage paths
        record: Feedback record to append
    """
    # Load existing feedback
    data = read_json(paths.feedback_json)
    
    # Initialize if empty
    if not data:
        data = {"feedback": []}
    
    # Append new record
    data["feedback"].append(asdict(record))
    
    # Write back
    write_json(paths.feedback_json, data)
    
    print(f" Feedback saved: {record.rating:+d} ({record.query[:50]}...)")


def update_boosts(
    paths: FeedbackPaths,
    used_chunk_ids: List[str],
    rating: int,
) -> None:
    """
    Update boost scores for retrieved chunks based on feedback.
    
    Pairwise Learning Strategy:
    - Positive feedback (+1): Increase boost for retrieved chunks
    - Negative feedback (-1): Decrease boost for retrieved chunks
    
    Args:
        paths: Feedback storage paths
        used_chunk_ids: List of chunk IDs that were retrieved
        rating: User rating (+1 or -1)
    """
    # Load existing boosts
    boosts = read_json(paths.boosts_json)
    
    # Initialize if empty
    if not boosts:
        boosts = {"chunk_boosts": {}}
    
    chunk_boosts = boosts.get("chunk_boosts", {})
    
    # Learning rate
    BOOST_DELTA = 0.05  # Small incremental changes
    
    # Update boosts for each chunk
    for chunk_id in used_chunk_ids:
        current_boost = chunk_boosts.get(chunk_id, 0.0)
        
        # Apply rating
        if rating > 0:
            # Positive feedback: increase boost
            new_boost = current_boost + BOOST_DELTA
        else:
            # Negative feedback: decrease boost
            new_boost = current_boost - BOOST_DELTA
        
        # Clip to reasonable range [-1.0, 1.0]
        new_boost = max(-1.0, min(1.0, new_boost))
        
        chunk_boosts[chunk_id] = round(new_boost, 4)
    
    # Save updated boosts
    boosts["chunk_boosts"] = chunk_boosts
    boosts["last_updated"] = datetime.now().isoformat()
    boosts["total_chunks_with_boosts"] = len(chunk_boosts)
    
    write_json(paths.boosts_json, boosts)
    
    print(f" Updated boosts for {len(used_chunk_ids)} chunks (rating: {rating:+d})")


def get_boosts(paths: FeedbackPaths) -> Dict[str, float]:
    """
    Get current boost scores for all chunks
    
    Returns:
        Dictionary mapping chunk_id -> boost_score
    """
    boosts = read_json(paths.boosts_json)
    return boosts.get("chunk_boosts", {})


def get_feedback_stats(paths: FeedbackPaths) -> Dict[str, Any]:
    """
    Get statistics about feedback collected
    
    Returns:
        Dictionary with feedback statistics
    """
    feedback_data = read_json(paths.feedback_json)
    feedback_list = feedback_data.get("feedback", [])
    
    if not feedback_list:
        return {
            "total_feedback": 0,
            "positive": 0,
            "negative": 0,
            "positive_rate": 0.0,
        }
    
    total = len(feedback_list)
    positive = sum(1 for f in feedback_list if f.get("rating", 0) > 0)
    negative = sum(1 for f in feedback_list if f.get("rating", 0) < 0)
    
    return {
        "total_feedback": total,
        "positive": positive,
        "negative": negative,
        "positive_rate": round(positive / total * 100, 2) if total > 0 else 0.0,
    }


def get_top_performing_chunks(paths: FeedbackPaths, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Get top performing chunks based on boost scores
    
    Args:
        paths: Feedback storage paths
        top_k: Number of top chunks to return
        
    Returns:
        List of (chunk_id, boost_score) tuples
    """
    boosts = get_boosts(paths)
    
    # Sort by boost score
    sorted_boosts = sorted(
        boosts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    return [
        {"chunk_id": chunk_id, "boost_score": score}
        for chunk_id, score in sorted_boosts
    ]


def analyze_feedback_patterns(paths: FeedbackPaths) -> Dict[str, Any]:
    """
    Analyze patterns in user feedback
    
    Returns:
        Analysis results including:
        - Most common queries
        - Average ratings by query type
        - Temporal patterns
    """
    feedback_data = read_json(paths.feedback_json)
    feedback_list = feedback_data.get("feedback", [])
    
    if not feedback_list:
        return {"message": "No feedback data available"}
    
    # Query frequency
    query_counts = defaultdict(int)
    query_ratings = defaultdict(list)
    
    for f in feedback_list:
        query = f.get("query", "").lower()
        rating = f.get("rating", 0)
        
        query_counts[query] += 1
        query_ratings[query].append(rating)
    
    # Most common queries
    common_queries = sorted(
        query_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Average ratings by query
    avg_ratings = {
        query: round(sum(ratings) / len(ratings), 2)
        for query, ratings in query_ratings.items()
    }
    
    # Temporal analysis (by day)
    feedback_by_day = defaultdict(int)
    for f in feedback_list:
        timestamp = f.get("timestamp", "")
        day = timestamp[:10]  # Extract YYYY-MM-DD
        feedback_by_day[day] += 1
    
    return {
        "total_feedback": len(feedback_list),
        "unique_queries": len(query_counts),
        "common_queries": [
            {"query": q, "count": c} for q, c in common_queries
        ],
        "feedback_by_day": dict(feedback_by_day),
    }


def reset_feedback(paths: FeedbackPaths) -> None:
    """Reset all feedback and boosts (use with caution!)"""
    write_json(paths.feedback_json, {"feedback": []})
    write_json(paths.boosts_json, {"chunk_boosts": {}})
    print("  All feedback and boosts have been reset")


# Test and demo functions
def test_feedback_system():
    """Test the feedback system"""
    print("\n" + "=" * 70)
    print(" TESTING FEEDBACK SYSTEM")
    print("=" * 70)
    
    paths = FeedbackPaths()
    
    # Simulate positive feedback
    print("\n Simulating positive feedback...")
    record1 = make_feedback_record(
        query="iPhone 13 case compatibility",
        answer="Yes, this case fits iPhone 13",
        sources=[{"chunk_id": "qa_0001_c0001"}],
        rating=1,
        comment="Very helpful answer!"
    )
    append_feedback(paths, record1)
    update_boosts(paths, ["qa_0001_c0001"], rating=1)
    
    # Simulate negative feedback
    print("\n Simulating negative feedback...")
    record2 = make_feedback_record(
        query="Shipping time",
        answer="Ships in 3-5 days",
        sources=[{"chunk_id": "qa_0002_c0001"}],
        rating=-1,
        comment="Answer was not accurate"
    )
    append_feedback(paths, record2)
    update_boosts(paths, ["qa_0002_c0001"], rating=-1)
    
    # Get statistics
    print("\n Feedback Statistics:")
    stats = get_feedback_stats(paths)
    print(f"   Total feedback: {stats['total_feedback']}")
    print(f"   Positive: {stats['positive']}")
    print(f"   Negative: {stats['negative']}")
    print(f"   Positive rate: {stats['positive_rate']}%")
    
    # Get boosts
    print("\n Current Boosts:")
    boosts = get_boosts(paths)
    for chunk_id, boost in boosts.items():
        print(f"   {chunk_id}: {boost:+.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_feedback_system()