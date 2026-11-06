"""Eligibility blockers: things employers' systems often filter on before anyone reads your skills.

Found in code from the job description's own sentences (each blocker quotes the sentence). The
resume is only checked where it can answer: degree level and years of experience. Work
authorisation and location can't be read from a resume, so we ask the person to confirm them.
These are labelled "likely auto-filters": a hint to check, never a verdict on the person.
"""

import re
from typing import Literal

from pydantic import BaseModel

from app.services.experience import ExperienceRules

Status = Literal["ok", "check"]

WORK_AUTH = re.compile(r"authori[sz]ed to work|work authori[sz]ation|right to work|work permit|visa|sponsorship|"
                       r"citizen(?:ship)?|green card|security clearance", re.IGNORECASE)
LOCATION = re.compile(r"\bon-?site\b|\bin[- ]office\b|\bhybrid\b|\brelocat\w+|\bmust (?:be )?(?:located|reside|live)\b|"
                      r"\bbased in\b|\bwork from (?:our )?office\b", re.IGNORECASE)
DEGREE_ASK = re.compile(r"\b(?:bachelor'?s?|b\.?\s?tech|b\.?\s?e\b|b\.?\s?sc|bca|master'?s?|m\.?\s?tech|m\.?\s?sc|mca|mba|"
                        r"ph\.?\s?d|doctorate|degree|graduate)\b", re.IGNORECASE)
NEED_WORD = re.compile(r"\b(?:require[sd]?|must|mandatory|minimum|need(?:ed)?|should (?:have|hold)|holding|hold a|"
                       r"qualification|degree in|pursuing|completed)\b", re.IGNORECASE)
RANKS = [(3, r"ph\.?\s?d|doctorate"), (2, r"master'?s?|m\.?\s?tech|m\.?\s?sc|mca|mba|\bm\.?\s?s\b"),
         (1, r"bachelor'?s?|b\.?\s?tech|b\.?\s?e\b|b\.?\s?sc|bca|b\.?\s?com")]
GENERIC_DEGREE = re.compile(r"\bdegree\b|\bgraduate\b", re.IGNORECASE)     # a degree, level not stated: counts as a bachelor's
RANK_NAME = {1: "a bachelor's degree", 2: "a master's degree", 3: "a PhD"}
YEARS_SENTENCE = re.compile(r"\d+(?:\.\d+)?\s*\+?\s*(?:-|–|to)?\s*\d*\s*(?:years?|yrs?)\b", re.IGNORECASE)


class Blocker(BaseModel):
    kind: Literal["work_authorisation", "location", "degree", "experience"]
    jd_quote: str
    status: Status
    note: str


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [re.sub(r"^[\s\-•*·•]+", "", p).strip() for p in parts if len(p.strip()) > 8]


def _first(sentences: list[str], pattern: re.Pattern, also: re.Pattern | None = None) -> str | None:
    for s in sentences:
        if pattern.search(s) and (also is None or also.search(s)):
            return s[:220]
    return None


def _rank(text: str, lowest: bool) -> int:
    found = [r for r, pat in RANKS if re.search(pat, text, re.IGNORECASE)]
    if not found:
        return 1 if GENERIC_DEGREE.search(text) else 0
    return min(found) if lowest else max(found)


def find_blockers(jd_text: str, resume_text: str, role_years: float | None, your_years: float | None,
                  rules: ExperienceRules | None = None, internship_months: float = 0) -> list[Blocker]:
    rules = rules or ExperienceRules()
    sents = _sentences(jd_text)
    out: list[Blocker] = []

    if q := _first(sents, WORK_AUTH):
        out.append(Blocker(kind="work_authorisation", jd_quote=q, status="check",
                           note="A resume can't show this. Confirm you meet it: many employers filter on it automatically."))
    if q := _first(sents, LOCATION):
        out.append(Blocker(kind="location", jd_quote=q, status="check",
                           note="Check the work location or arrangement works for you before you apply."))
    if q := _first(sents, DEGREE_ASK, NEED_WORD):
        need = _rank(q, lowest=True) or 1
        have = _rank(resume_text, lowest=False)
        if have >= need:
            out.append(Blocker(kind="degree", jd_quote=q, status="ok",
                               note=f"The job asks for {RANK_NAME[need]}; your resume shows one."))
        else:
            out.append(Blocker(kind="degree", jd_quote=q, status="check",
                               note=f"The job asks for {RANK_NAME[need]}. Your resume doesn't clearly show one "
                                    f"(or it is still in progress): say so plainly, since filters may check it."))
    if role_years and your_years is not None:
        counted = your_years + (internship_months / 12 if rules.internships_count else 0)
        quote = _first(sents, YEARS_SENTENCE) or f"{role_years:g}+ years of experience"
        if counted >= role_years:
            out.append(Blocker(kind="experience", jd_quote=quote, status="ok",
                               note=f"The job asks for {role_years:g}+ years; you have {counted:g}."))
        else:
            tail = (" The job says projects count, so describe yours as real experience." if rules.projects_count
                    else " Filters often use this number, so apply only if your work is a close match.")
            out.append(Blocker(kind="experience", jd_quote=quote, status="check",
                               note=f"The job asks for {role_years:g}+ years and you show {counted:g}." + tail))
    return out
