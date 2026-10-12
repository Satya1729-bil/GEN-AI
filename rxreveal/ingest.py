"""Read PDFs and split them into chunks for search.

Medical record PDFs often hold many patients per page, and one patient can run
across a page break. So the whole document is joined first, then split on
"Patient <n>:" headings when the document has them. Each record stays one
chunk, which keeps symptoms, diagnosis and treatment of a patient together.
Documents without patient headings fall back to paragraph-based chunks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

PATIENT_HEADING = re.compile(r"(?=\bPatient\s+\d+\s*:)", re.IGNORECASE)


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text) for each page that has text. Pages start at 1."""
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append((number, text))
    return pages


def _join_pages(pages: list[tuple[int, str]]) -> tuple[str, list[tuple[int, int]]]:
    """Join pages into one string. Returns the text and (start_offset, page) pairs."""
    parts, offsets, cursor = [], [], 0
    for number, text in pages:
        offsets.append((cursor, number))
        parts.append(text)
        cursor += len(text) + 2
    return "\n\n".join(parts), offsets


def _page_at(offset: int, offsets: list[tuple[int, int]]) -> int:
    page = offsets[0][1] if offsets else 1
    for start, number in offsets:
        if start <= offset:
            page = number
        else:
            break
    return page


def _split_long(text: str, size: int, overlap: int) -> list[tuple[int, str]]:
    """Split text into pieces of about `size` characters, on paragraph or line
    breaks where possible. Returns (relative_offset, piece) pairs."""
    if len(text) <= size:
        return [(0, text)]
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
            if cut > size // 2:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append((start, piece))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_pages(
    pages: list[tuple[int, str]], source: str, size: int = 1200, overlap: int = 150
) -> list[Chunk]:
    full, offsets = _join_pages(pages)
    if not full:
        return []

    records = [m.start() for m in PATIENT_HEADING.finditer(full)]
    if len(records) >= 2:
        bounds = records + [len(full)]
        segments = []
        if records[0] > 0 and full[: records[0]].strip():
            segments.append((0, full[: records[0]]))
        segments += [(bounds[i], full[bounds[i] : bounds[i + 1]]) for i in range(len(records))]
    else:
        segments = [(0, full)]

    chunks = []
    for seg_start, seg_text in segments:
        for rel, piece in _split_long(seg_text, size, overlap):
            piece = piece.strip()
            if len(piece) < 20:
                continue
            page = _page_at(seg_start + rel, offsets)
            digest = hashlib.sha1(f"{source}|{seg_start + rel}|{piece}".encode()).hexdigest()[:12]
            chunks.append(Chunk(id=digest, text=piece, source=source, page=page))
    return chunks


def load_pdf(path: Path, size: int = 1200, overlap: int = 150) -> list[Chunk]:
    return chunk_pages(read_pdf_pages(path), source=path.name, size=size, overlap=overlap)


def load_folder(folder: Path, size: int = 1200, overlap: int = 150) -> list[Chunk]:
    """Load every PDF in a folder (and its subfolders)."""
    chunks: list[Chunk] = []
    for path in sorted(folder.rglob("*.pdf")):
        chunks.extend(load_pdf(path, size=size, overlap=overlap))
    return chunks
