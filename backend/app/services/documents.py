"""Read a resume or job description from .txt, .pdf or .docx, in memory.

The text is returned to the caller and never written anywhere: no converted copy,
no temp file. Masking happens afterwards, on this text, before any LLM sees it.
"""

import io
from pathlib import Path

SUPPORTED = (".txt", ".pdf", ".docx")
MAX_BYTES = 5 * 1024 * 1024          # 5 MB is plenty for a resume


class DocumentError(ValueError):
    """The file can't be read as a resume or job description."""


def read_document(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise DocumentError(f"File not found: {path}. Check the folder and file name.")
    return read_bytes(path.read_bytes(), path.name)


def read_bytes(data: bytes, filename: str) -> str:
    """Same as read_document, for an upload that is already in memory (used by the API later)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise DocumentError(f"Unsupported file type '{suffix}'. Use {', '.join(SUPPORTED)}.")
    if len(data) > MAX_BYTES:
        raise DocumentError("File is larger than 5 MB.")
    if suffix == ".txt":
        text = data.decode("utf-8", errors="replace")
    elif suffix == ".pdf":
        text = _pdf_text(data)
    else:
        text = _docx_text(data)
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(text) < 30:
        raise DocumentError("Almost no text found. If this is a scanned PDF, export it as text first.")
    return text


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:
        raise DocumentError(f"Could not read this PDF: {error}") from None


def _docx_text(data: bytes) -> str:
    from docx import Document
    try:
        doc = Document(io.BytesIO(data))
    except Exception as error:
        raise DocumentError(f"Could not read this Word file: {error}") from None
    lines = [p.text for p in doc.paragraphs]
    for table in doc.tables:                       # many resume templates use tables
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(lines)
