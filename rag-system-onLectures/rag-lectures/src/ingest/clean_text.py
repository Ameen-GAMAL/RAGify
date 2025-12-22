from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Iterable, Any, Tuple, Set


@dataclass
class CleanPageRecord:
    lecture_id: str
    source_file: str
    page_num: int
    text_clean: str
    removed_header_lines: List[str]
    removed_footer_lines: List[str]


# Common bullet glyphs seen in PDFs (includes your sample "")
BULLET_CHARS = [
    "\uf0bf",  # private-use bullet (common in converted fonts)
    "",
    "•",
    "◦",
    "▪",
    "●",
    "‣",
    "–",
    "—",
]


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


def canonical_line(line: str) -> str:
    # Normalize for counting equality across pages (spacing/case differences)
    s = line.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.casefold()
    return s


def is_noise_line(line: str) -> bool:
    # Filters for lines we shouldn't even consider as stable headers/footers
    s = line.strip()
    if len(s) < 4:
        return True
    # pure digits (page numbers), or short numeric tokens
    if re.fullmatch(r"\d{1,4}", s):
        return True
    return False


def normalize_text(text: str) -> str:
    # normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # fix hyphenation across line breaks: inter-\nnational -> international
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # strip indentation line-by-line (very important for slide PDFs)
    lines = [ln.strip() for ln in text.split("\n")]

    # normalize bullet glyphs
    normalized_lines: List[str] = []
    for ln in lines:
        for b in BULLET_CHARS:
            ln = ln.replace(b, "-")
        # Collapse multiple spaces within a line
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        normalized_lines.append(ln)

    text = "\n".join(normalized_lines)

    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def first_last_nonempty_lines(text: str, n: int = 2) -> Tuple[List[str], List[str]]:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return [], []
    header = lines[:n]
    footer = lines[-n:] if len(lines) >= n else lines
    return header, footer


def compute_repeated_header_footer_canon(
    pages_text: List[str],
    n: int = 2,
    threshold_ratio: float = 0.6,
) -> Tuple[Set[str], Set[str]]:
    """
    Returns sets of canonical lines to remove for header/footer.
    A line is removable if it appears on >= threshold_ratio of pages.
    """
    header_counter = Counter()
    footer_counter = Counter()
    num_pages = len(pages_text)

    for t in pages_text:
        header, footer = first_last_nonempty_lines(t, n=n)
        for ln in header:
            if not is_noise_line(ln):
                header_counter[canonical_line(ln)] += 1
        for ln in footer:
            if not is_noise_line(ln):
                footer_counter[canonical_line(ln)] += 1

    header_remove = {
        k for k, c in header_counter.items()
        if c / max(1, num_pages) >= threshold_ratio
    }
    footer_remove = {
        k for k, c in footer_counter.items()
        if c / max(1, num_pages) >= threshold_ratio
    }

    return header_remove, footer_remove


def remove_repeated_lines(
    text: str, header_remove_canon: Set[str], footer_remove_canon: Set[str]
) -> Tuple[str, List[str], List[str]]:
    removed_h: List[str] = []
    removed_f: List[str] = []

    out_lines: List[str] = []
    for ln in text.split("\n"):
        stripped = ln.strip()
        if not stripped:
            out_lines.append(ln)
            continue

        c = canonical_line(stripped)
        if c in header_remove_canon:
            removed_h.append(stripped)
            continue
        if c in footer_remove_canon:
            removed_f.append(stripped)
            continue

        out_lines.append(ln)

    cleaned = "\n".join(out_lines)
    cleaned = normalize_text(cleaned)  # re-normalize after removals

    return cleaned, sorted(set(removed_h)), sorted(set(removed_f))


def main(
    in_path: str = "data/processed/pages.jsonl",
    out_path: str = "data/processed/pages_clean.jsonl",
    header_footer_lines: int = 2,
    threshold_ratio: float = 0.6,
) -> None:
    in_p = Path(in_path)
    out_p = Path(out_path)

    rows = read_jsonl(in_p)

    by_lecture: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_lecture[r["lecture_id"]].append(r)

    out_rows: List[Dict[str, Any]] = []

    for lecture_id, pages in by_lecture.items():
        pages = sorted(pages, key=lambda x: x["page_num"])

        normalized_pages = [normalize_text(p["text_raw"]) for p in pages]

        header_remove_canon, footer_remove_canon = compute_repeated_header_footer_canon(
            normalized_pages,
            n=header_footer_lines,
            threshold_ratio=threshold_ratio,
        )

        for p, norm in zip(pages, normalized_pages):
            cleaned, removed_h, removed_f = remove_repeated_lines(
                norm, header_remove_canon, footer_remove_canon
            )

            rec = CleanPageRecord(
                lecture_id=lecture_id,
                source_file=p["source_file"],
                page_num=p["page_num"],
                text_clean=cleaned,
                removed_header_lines=removed_h,
                removed_footer_lines=removed_f,
            )
            out_rows.append(asdict(rec))

    write_jsonl(out_p, out_rows)
    print(f"Wrote {len(out_rows)} cleaned pages → {out_p.resolve()}")


if __name__ == "__main__":
    main()
