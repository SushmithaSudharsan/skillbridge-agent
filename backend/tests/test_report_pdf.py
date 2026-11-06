"""The PDF report: every section, the student's answers, no personal details."""

from datetime import date
from io import BytesIO

from pypdf import PdfReader

from app.services.report import render_markdown
from app.services.report_pdf import render_pdf, t
from test_analysis import result  # noqa: F401  (reuse the analysis fixture)


def pdf_text(data: bytes) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(BytesIO(data)).pages)


def test_pdf_has_every_section(result):
    data = render_pdf(result, on=date(2026, 10, 24))
    assert data.startswith(b"%PDF")
    text = pdf_text(data)
    for heading in ["SKILLBRIDGE REPORT", "Summary", "Learn: what's missing", "Learning plan",
                    "Portfolio projects", "Resume checks", "Your questions and answers"]:
        assert heading in text
    assert "24 October 2026" in text and "Fundamentals of RAG" in text


def test_answers_appear_and_unanswered_is_marked(result):
    answered = result.model_copy(update={"answers": ["About 0.87 R2 on a 20% test split."]})
    assert "About 0.87 R2 on a 20% test split." in pdf_text(render_pdf(answered))
    assert "Not answered" in pdf_text(render_pdf(result))
    assert "Your answer: About 0.87" in render_markdown(answered)


def test_pdf_has_no_personal_details(result):
    text = pdf_text(render_pdf(result))
    assert "Priya" not in text and "example.com" not in text and "98765" not in text


def test_text_is_escaped():
    assert t("<b>x</b> & y") == "&lt;b&gt;x&lt;/b&gt; &amp; y"
    assert t("Beginner → Advanced") == "Beginner -&gt; Advanced"
