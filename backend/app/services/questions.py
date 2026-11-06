"""Questions about resume lines, chosen by fixed rules, so the same resume always gets
the same questions (an LLM writing them gave different questions on every run).

A line is worth asking about when:
  vague verb    "owned", "responsible for", "worked on" ...  -> what did YOU do?
  self-rating   "proficient in", "expert in", "familiar with"  -> where, and how much?
  no result     an experience/project line with no number      -> anything you measured?
Self-ratings first, then vague verbs, then missing results; within each, lines that
mention a skill the job asks for come first. Lines are copied exactly from the
masked resume, so the answers can be checked against them.
"""

import re

from app.services.analysis import ResumeQuestion

VAGUE = re.compile(r"\b(owned|responsible for|worked on|involved in|helped(?: with)?|assisted(?: in| with)?|"
                   r"participated in|contributed to|handled)\b", re.IGNORECASE)
SELF_RATING = re.compile(r"\b(proficient (?:in|with)|expert (?:in|with)|familiar with|knowledge of|"
                         r"exposure to|skilled in|strong (?:knowledge|understanding) of)\b", re.IGNORECASE)
SKIP = re.compile(r"^\s*(?:education|skills?|technical skills|coursework|relevant coursework|target role|"
                  r"certifications?|languages|interests|contact|summary|objective)\b\s*:?", re.IGNORECASE)
MIN_WORDS = 6


def _lines(resume: str) -> list[str]:
    out = []
    for raw in resume.splitlines():
        line = raw.strip().lstrip("•*-–· ").strip()
        if len(line.split()) >= MIN_WORDS and "[" not in line[:12]:
            out.append(line)
    return out


def _is_list(line: str) -> bool:
    """A comma list like "Python, SQL, Pandas" or "Programming & Data: Python, SQL, R":
    a skills list, not a claim about work."""
    if ":" in line[:40]:
        line = line.split(":", 1)[1]
    parts = [p.strip() for p in line.split(",")]
    return len(parts) >= 3 and all(len(p.split()) <= 3 for p in parts)


def line_questions(safe_resume: str, job_skills: list[str], limit: int = 3) -> list[ResumeQuestion]:
    found = []
    for order, line in enumerate(_lines(safe_resume)):
        mentions = any(re.search(rf"(?<![\w]){re.escape(s)}(?![\w])", line, re.IGNORECASE) for s in job_skills)
        vague, rating = VAGUE.search(line), SELF_RATING.search(line)
        if _is_list(line) and not rating:
            continue
        if rating:
            q = ResumeQuestion(resume_line=line,
                               issue=f"'{rating.group(1)}' is a self-rating a recruiter can't check.",
                               question="Where have you actually used these, and how much: a course, one project, "
                                        "several projects, or a job?")
            kind = 0
        elif vague:
            q = ResumeQuestion(resume_line=line, issue=f"'{vague.group(1)}' doesn't say what you did yourself.",
                               question="What exactly did you do yourself here: what did you build, write or decide?")
            kind = 1
        elif SKIP.match(line) or re.search(r"\d", line) or _is_list(line):
            continue
        else:
            q = ResumeQuestion(resume_line=line, issue="No result or scale is stated.",
                               question="Did you measure anything here (accuracy, time saved, data size, users)? "
                                        "If not, leave it as it is.")
            kind = 2
        found.append((kind, not mentions, order, q))
    found.sort(key=lambda x: x[:3])
    return [q for *_, q in found[:limit]]
