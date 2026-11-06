"""The take-away report of an analysis: results only, escaped, printable."""

from datetime import date

from app.services.report import render_html, render_markdown
from test_analysis import result  # noqa: F401  (reuse the analysis fixture)


def test_markdown_has_every_section(result):
    md = render_markdown(result, on=date(2026, 10, 24))
    for heading in ["# SkillBridge report: AI Engineer · Mid", "## Summary", "## Learn: what's missing",
                    "## Learning plan", "## Portfolio projects", "## Resume"]:
        assert heading in md
    assert "Take a course in Fundamentals of RAG (Beginner)" in md


def test_report_has_no_personal_details(result):
    page = render_html(result)
    assert "Priya" not in page and "example.com" not in page and "98765" not in page


def test_html_is_escaped_and_printable(result):
    unsafe = result.model_copy(update={"summary": "<script>x</script>"})
    page = render_html(unsafe)
    assert "<script>x</script>" not in page and "&lt;script&gt;" in page
    assert "@page" in page and "@media print" in page
