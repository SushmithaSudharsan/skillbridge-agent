"""SkillBridge must never save a resume. Reading one must leave no file behind."""


from app.services.catalog import DEFAULT_DATA_DIR, load_catalog
from app.services.llm import FakeLLM
from app.services.matching import load_matcher
from app.services.resume import ResumeDraft, ResumeSkillDraft, read_resume

PROJECT = DEFAULT_DATA_DIR.parent


def snapshot() -> dict[str, float]:
    """Every file in the project (except the venv and git) with its last-modified time."""
    files = {}
    for path in PROJECT.rglob("*"):
        parts = set(path.relative_to(PROJECT).parts)
        if path.is_file() and not parts & {".venv", ".git", "__pycache__", ".pytest_cache"}:
            files[str(path)] = path.stat().st_mtime
    return files


def test_reading_a_resume_writes_nothing(tmp_path):
    before = snapshot()
    catalog = load_catalog()
    llm = FakeLLM(extractions=[ResumeDraft(skills=[
        ResumeSkillDraft(skill="Python", level=2, source="project", evidence="Built X in Python")])])
    read_resume(llm, "Priya Raman\nBuilt X in Python", load_matcher(catalog.skills), catalog.skills)
    assert snapshot() == before
