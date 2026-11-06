"""Skills your work implies. If a resume shows ML work in a project, the student wrote Python
and used NumPy and Pandas to do it, even if those words aren't on that line.

The rule (all in code, all visible in "How we read your resume"):
  - only from real use: a project, a job, or a degree/coursework (never from a skills list)
  - the implied skill gets the SAME level as the work, capped at Intermediate
  - it never lowers a level the resume already proves
  - it is labelled "implied by your <skill> work", quoting that work's evidence
The table is short on purpose: only links an interviewer would accept without argument.
"""

from app.services.models import Level, StudentSkill

IMPLIES: dict[str, list[str]] = {
    "Machine Learning": ["Python", "NumPy", "Pandas"],
    "Scikit-learn": ["Python", "NumPy", "Pandas"],
    "Deep Learning": ["Python", "NumPy"],
    "PyTorch": ["Python", "NumPy"],
    "MLOps": ["Python", "Machine Learning"],
    "Model Monitoring": ["Python"],
    "LLMs": ["Python"],
    "RAG": ["Python", "LLMs"],
    "LangChain": ["Python", "LLMs"],
    "AI Agents": ["Python", "LLMs"],
    "Data Pipelines": ["Python"],
    "REST APIs": ["Python"],
}
FROM_SOURCES = {"project", "job", "course"}
CAP = Level.INTERMEDIATE


def add_implied(skills: list[StudentSkill], rounds: int = 2) -> list[StudentSkill]:
    """Return skills plus what they imply. Two rounds so MLOps -> Machine Learning -> NumPy works."""
    out = list(skills)
    for _ in range(rounds):
        best = {}
        for s in out:
            if s.skill not in best or s.level > best[s.skill].level:
                best[s.skill] = s
        added = False
        for s in list(best.values()):
            if s.source not in FROM_SOURCES | {"implied"}:
                continue
            for target in IMPLIES.get(s.skill, []):
                level = min(s.level, CAP)
                if target in best and best[target].level >= level:
                    continue
                root = s.via or s.skill
                new = StudentSkill(skill=target, level=level, source="implied", evidence=s.evidence, via=root)
                out.append(new)
                best[target] = new
                added = True
        if not added:
            break
    return out
