from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Dict, Any, List

import fitz  # PyMuPDF
from tqdm import tqdm


@dataclass
class PageRecord:
    lecture_id: str
    source_file: str
    page_num: int  # 1-based
    text_raw: str
    char_count: int
    is_empty: bool


def iter_pdf_paths(raw_dir: Path) -> List[Path]:
    return sorted([p for p in raw_dir.glob("*.pdf") if p.is_file()])


def lecture_id_from_filename(pdf_path: Path) -> str:
    # e.g., "ADB_Lec01.pdf" -> "ADB_Lec01"
    return pdf_path.stem


def extract_pages(pdf_path: Path, sort_text: bool = True) -> Iterable[PageRecord]:
    lecture_id = lecture_id_from_filename(pdf_path)
    doc = fitz.open(pdf_path)

    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text", sort=sort_text)  # better order for slides
        text_stripped = text.strip()
        char_count = len(text_stripped)

        yield PageRecord(
            lecture_id=lecture_id,
            source_file=pdf_path.name,
            page_num=i + 1,
            text_raw=text,
            char_count=char_count,
            is_empty=(char_count == 0),
        )


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(
    raw_dir: str = "data/raw_pdfs",
    out_path: str = "data/processed/pages.jsonl",
) -> None:
    raw_dir_p = Path(raw_dir)
    out_path_p = Path(out_path)

    pdfs = iter_pdf_paths(raw_dir_p)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {raw_dir_p.resolve()}")

    all_rows: List[Dict[str, Any]] = []
    for pdf in tqdm(pdfs, desc="Extracting PDFs"):
        for rec in extract_pages(pdf, sort_text=True):
            all_rows.append(asdict(rec))

    write_jsonl(out_path_p, all_rows)
    print(f"Wrote {len(all_rows)} page records → {out_path_p.resolve()}")


if __name__ == "__main__":
    main()
