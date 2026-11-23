"""Phase 2: PDF text extraction and chunking."""
from typing import BinaryIO, Dict, List

from pypdf import PdfReader


def extract_pages(file: BinaryIO) -> List[Dict]:
    """Return [{'page': 1, 'text': '...'}, ...] for every non-empty page."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks, preferring to break on whitespace."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + chunk_size // 2, end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def process_pdf(file: BinaryIO, source_name: str) -> List[Dict]:
    """Extract and chunk a PDF. Each chunk keeps its source file and page."""
    chunks = []
    for page in extract_pages(file):
        for piece in chunk_text(page["text"]):
            chunks.append(
                {
                    "id": f"{source_name}-{len(chunks)}",
                    "text": piece,
                    "source": source_name,
                    "page": page["page"],
                }
            )
    return chunks
