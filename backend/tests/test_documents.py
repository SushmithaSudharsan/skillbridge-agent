"""Reading resumes from .txt, .pdf and .docx, in memory, then masking."""

import io

import pytest

from app.services.documents import DocumentError, read_bytes, read_document
from app.services.guardrails import mask_pii

LINES = ["Priya Raman", "priya.raman@example.com | +91 98765 43210",
         "Skills: Python, SQL, Pandas", "Data Analyst Intern at Zylo Technologies",
         "Built a churn model in Python with scikit-learn"]


def make_pdf(lines: list[str]) -> bytes:
    """A tiny valid one-page PDF with one line of text per entry."""
    text = "BT /F1 11 Tf 50 800 Td 14 TL " + " ".join(f"({l}) Tj T*" for l in lines) + " ET"
    objects = ["<< /Type /Catalog /Pages 2 0 R >>",
               "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
               f"<< /Length {len(text)} >>\nstream\n{text}\nendstream",
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def make_docx(lines: list[str]) -> bytes:
    from docx import Document
    doc = Document()
    for line in lines[:3]:
        doc.add_paragraph(line)
    table = doc.add_table(rows=1, cols=2)              # templates often put experience in tables
    table.rows[0].cells[0].text = lines[3]
    table.rows[0].cells[1].text = lines[4]
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("name, data", [
    ("resume.txt", "\n".join(LINES).encode()),
    ("resume.pdf", make_pdf(LINES)),
    ("resume.docx", make_docx(LINES)),
], ids=["txt", "pdf", "docx"])   # short names: pytest would otherwise use the file bytes as the
                                 # test name, which breaks Windows' 32,767-character env var limit
def test_every_format_is_read_and_then_masked(name, data):
    text = read_bytes(data, name)
    assert "Built a churn model in Python with scikit-learn" in text
    masked = mask_pii(text)
    for secret in ["Priya", "Raman", "example.com", "98765", "Zylo"]:
        assert secret not in masked, (name, secret)
    assert "Skills: Python, SQL, Pandas" in masked


def test_missing_file_has_a_clear_message(tmp_path):
    with pytest.raises(DocumentError, match="File not found"):
        read_document(tmp_path / "private" / "my_resume.txt")


def test_unsupported_type_is_rejected():
    with pytest.raises(DocumentError, match="Unsupported file type"):
        read_bytes(b"data", "resume.pages")


def test_empty_or_scanned_file_is_rejected():
    with pytest.raises(DocumentError, match="Almost no text"):
        read_bytes(make_pdf([""]), "scan.pdf")


def test_broken_pdf_is_rejected():
    with pytest.raises(DocumentError, match="Could not read this PDF"):
        read_bytes(b"%PDF-1.4 not really a pdf", "broken.pdf")


def test_too_large_is_rejected():
    with pytest.raises(DocumentError, match="larger than 5 MB"):
        read_bytes(b"x" * (5 * 1024 * 1024 + 1), "big.txt")


def test_reading_writes_no_files(tmp_path):
    path = tmp_path / "resume.docx"
    path.write_bytes(make_docx(LINES))
    before = sorted(tmp_path.iterdir())
    read_document(path)
    assert sorted(tmp_path.iterdir()) == before
