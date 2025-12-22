from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Iterable, Tuple, Optional
from collections import defaultdict

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_lecture_text_and_offsets(
    pages: List[Dict[str, Any]]
) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Concatenate pages into one lecture_text and return mapping:
    [(page_num, start_char, end_char), ...]
    """
    parts: List[str] = []
    offsets: List[Tuple[int, int, int]] = []
    cursor = 0

    for p in pages:
        t = (p.get("text_clean") or "").strip()
        if not t:
            continue

        start = cursor
        parts.append(t)
        cursor += len(t)
        end = cursor
        offsets.append((p["page_num"], start, end))

        # Add a separator between pages to preserve boundaries
        parts.append("\n\n")
        cursor += 2

    lecture_text = "".join(parts).strip()
    return lecture_text, offsets


def pages_spanned(
    offsets: List[Tuple[int, int, int]],
    chunk_start: int,
    chunk_end: int
) -> Tuple[Optional[int], Optional[int]]:
    touched = [
        page_num
        for (page_num, s, e) in offsets
        if not (e < chunk_start or s > chunk_end)
    ]
    if not touched:
        return None, None
    return min(touched), max(touched)


def main(
    in_path: str = "data/processed/pages_clean.jsonl",
    out_path: str = "data/processed/chunks.jsonl",
    chunk_size: int = 2200,
    chunk_overlap: int = 300,
) -> None:
    in_p = Path(in_path)
    out_p = Path(out_path)

    rows = read_jsonl(in_p)
    by_lecture = defaultdict(list)
    for r in rows:
        by_lecture[r["lecture_id"]].append(r)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    out_rows: List[Dict[str, Any]] = []

    for lecture_id, pages in by_lecture.items():
        pages = sorted(pages, key=lambda x: x["page_num"])
        lecture_text, offsets = build_lecture_text_and_offsets(pages)
        if not lecture_text:
            continue

        chunks = splitter.split_text(lecture_text)

        # Locate chunks in the concatenated text to estimate page spans.
        # Use progressive searching to reduce mismatch issues.
        search_cursor = 0
        for idx, ch in enumerate(chunks, start=1):
            ch = ch.strip()
            if not ch:
                continue

            pos = lecture_text.find(ch, search_cursor)
            if pos == -1:
                # fallback: just mark unknown span
                chunk_start, chunk_end = 0, 0
            else:
                chunk_start = pos
                chunk_end = pos + len(ch)
                # advance cursor allowing overlap
                search_cursor = max(pos, pos + len(ch) - chunk_overlap)

            p_start, p_end = pages_spanned(offsets, chunk_start, chunk_end)

            chunk_id = f"{lecture_id}_c{idx:04d}"
            rec = ChunkRecord(
                chunk_id=chunk_id,
                text=ch,
                metadata={
                    "lecture_id": lecture_id,
                    "source_file": pages[0]["source_file"],
                    "page_start": p_start,
                    "page_end": p_end,
                    "char_len": len(ch),
                },
            )
            out_rows.append(asdict(rec))

    write_jsonl(out_p, out_rows)
    print(f"Wrote {len(out_rows)} chunks → {out_p.resolve()}")


if __name__ == "__main__":
    main()
