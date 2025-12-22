from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


@dataclass(frozen=True)
class FeedbackPaths:
    feedback_jsonl: Path = Path("data/processed/feedback.jsonl")
    boosts_json: Path = Path("data/processed/boosts.json")


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_feedback(
    paths: FeedbackPaths,
    record: Dict[str, Any],
) -> None:
    paths.feedback_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with paths.feedback_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_boosts(
    paths: FeedbackPaths,
    used_chunk_ids: List[str],
    rating: int,
) -> Dict[str, float]:
    """
    Simple self-learning: maintain per-chunk boost weights.
    rating: +1 for helpful, -1 for not helpful
    """
    boosts: Dict[str, float] = _read_json(paths.boosts_json, default={})
    for cid in used_chunk_ids:
        boosts[cid] = float(boosts.get(cid, 0.0) + rating * 0.1)  # small increments
        # keep boosts bounded
        boosts[cid] = max(-1.0, min(1.0, boosts[cid]))
    _write_json(paths.boosts_json, boosts)
    return boosts


def get_boosts(paths: FeedbackPaths) -> Dict[str, float]:
    return _read_json(paths.boosts_json, default={})


def make_feedback_record(
    query: str,
    answer: str,
    sources: List[Dict[str, Any]],
    rating: int,
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": answer,
        "sources": sources,
        "rating": rating,
        "comment": comment or "",
    }
