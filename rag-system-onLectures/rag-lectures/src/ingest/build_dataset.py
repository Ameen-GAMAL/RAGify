from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List

from src.ingest.extract_text import main as extract_main
from src.ingest.clean_text import main as clean_main
from src.ingest.chunk_text import main as chunk_main


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    # Run the full pipeline
    extract_main()
    clean_main()
    chunk_main()

    pages = read_jsonl(Path("data/processed/pages.jsonl"))
    chunks = read_jsonl(Path("data/processed/chunks.jsonl"))

    stats = defaultdict(lambda: {"pages": 0, "empty_pages": 0, "chars": 0, "chunks": 0})

    for p in pages:
        s = stats[p["lecture_id"]]
        s["pages"] += 1
        s["empty_pages"] += 1 if p.get("is_empty") else 0
        s["chars"] += int(p.get("char_count", 0))

    for c in chunks:
        lecture_id = c["metadata"]["lecture_id"]
        stats[lecture_id]["chunks"] += 1

    manifest = {
        "raw_pdfs_dir": str(Path("data/raw_pdfs").resolve()),
        "processed_dir": str(Path("data/processed").resolve()),
        "outputs": {
            "pages": str(Path("data/processed/pages.jsonl").resolve()),
            "pages_clean": str(Path("data/processed/pages_clean.jsonl").resolve()),
            "chunks": str(Path("data/processed/chunks.jsonl").resolve()),
        },
        "per_lecture": dict(stats),
        "qa_hints": {
            "check_headers_removed": "Open pages_clean.jsonl and verify repeated course/title lines are removed.",
            "check_bullets_normalized": "Look for '-' bullets instead of PDF glyphs like ''.",
            "check_chunk_sizes": "Most chunks should be > ~500 chars; adjust chunk_size if too small/large.",
        },
    }

    write_json(Path("data/processed/manifest.json"), manifest)
    print(f"Wrote manifest → {Path('data/processed/manifest.json').resolve()}")


if __name__ == "__main__":
    main()
