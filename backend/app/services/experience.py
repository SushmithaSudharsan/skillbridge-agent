"""Work out a job's seniority from its title and job description."""

import re

from pydantic import BaseModel

from app.services.models import SeniorityDecision

# Years of experience -> seniority. The number is the lowest years that count.
MID_FROM_YEARS = 2
SENIOR_FROM_YEARS = 5

# Matches "3-5 years", "3 to 5 yrs", "5+ years", "minimum 4 years", "2.5 years".
YEARS_PATTERN = re.compile(
    r"""
    (?:(?:minimum|min\.?|at\s+least|over|more\s+than)\s+(?:of\s+)?)?   # optional "minimum" etc.
    (?P<low>\d+(?:\.\d+)?)                                              # first number
    (?:\s*(?:-|–|to)\s*(?P<high>\d+(?:\.\d+)?))?                        # optional "- 5"
    \s*\+?\s*                                                           # optional "+"
    (?:years?|yrs?)\b                                                   # "years" or "yrs"
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Words that mean "this is a fresher job", even without numbers.
FRESHER_PATTERN = re.compile(
    r"\bfreshers?\b|\bentry[- ]level\b|\brecent graduates?\b"
    r"|\bno (?:prior )?experience (?:is )?(?:required|needed)\b",
    re.IGNORECASE,
)

EXPERIENCE_WORD = re.compile(r"\bexp(?:erience|\.)?\b", re.IGNORECASE)

# Weak hints. They only count when the JD gives no years and the title no level word.
# "Mentor junior engineers", "lead a team", "technical direction" -> at least mid.
SENIOR_HINT_PATTERN = re.compile(
    r"\bmentor(?:ing|s)?\s+(?:the\s+)?(?:junior|interns?|new|other|team)\w*"
    r"|\blead(?:ing)?\s+(?:a|the)\s+team\b|\bteam\s+lead\b"
    r"|\btechnical\s+(?:direction|strategy)\b",
    re.IGNORECASE,
)
# "You will be mentored", "learn from senior engineers" -> fresher.
FRESHER_HINT_PATTERN = re.compile(
    r"\b(?:be|get|are)\s+mentored\b|\bmentored\s+by\b"
    r"|\blearn\s+from\s+(?:senior|experienced)\b|\bunder\s+the\s+guidance\s+of\b",
    re.IGNORECASE,
)


def years_required(jd_text: str) -> float | None:
    """The minimum years of experience a JD asks for, or None if it never says.

    Only sentences that mention experience count, so "founded 20 years ago" is
    ignored. If several numbers appear ("3+ years overall, 1+ year with LLMs"),
    the highest minimum wins, because that is the real bar to get the job.
    """
    found: list[float] = []
    for sentence in re.split(r"\n|;|•|\.\s", jd_text):
        if not EXPERIENCE_WORD.search(sentence):
            continue
        for match in YEARS_PATTERN.finditer(sentence):
            found.append(float(match.group("low")))
    return max(found) if found else None


def seniority_from_years(years: float) -> str:
    if years < MID_FROM_YEARS:
        return "fresher"
    if years < SENIOR_FROM_YEARS:
        return "mid"
    return "senior"


def decide_seniority(title_seniority: str | None, jd_text: str = "") -> SeniorityDecision:
    """Combine the title and the JD. Trust order: JD years > JD words > title > JD hints."""
    years = years_required(jd_text)
    if years is not None:
        seniority = seniority_from_years(years)
        note = f"JD asks for {years:g}+ years of experience"
        if title_seniority and title_seniority != seniority:
            note += f"; the title suggests {title_seniority}, but the JD is more specific"
        return SeniorityDecision(seniority=seniority, source="jd_years", note=note)

    if FRESHER_PATTERN.search(jd_text):
        note = "JD says freshers or entry level are welcome"
        if title_seniority and title_seniority != "fresher":
            note += f"; the title suggests {title_seniority}"
        return SeniorityDecision(seniority="fresher", source="jd_words", note=note)

    if title_seniority:
        return SeniorityDecision(seniority=title_seniority, source="title",
                                 note="Taken from a level word in the job title")

    senior_hints = _phrases(SENIOR_HINT_PATTERN, jd_text)
    fresher_hints = _phrases(FRESHER_HINT_PATTERN, jd_text)
    if senior_hints and fresher_hints:
        return SeniorityDecision(
            seniority=None, source="unknown",
            note=f"JD hints point both ways: {_quote(senior_hints)} vs {_quote(fresher_hints)}",
        )
    if senior_hints:
        return SeniorityDecision(
            seniority="mid", source="jd_hints",
            note=f"No years stated, but the JD mentions {_quote(senior_hints)}, so likely mid or above",
        )
    if fresher_hints:
        return SeniorityDecision(
            seniority="fresher", source="jd_hints",
            note=f"No years stated, but the JD mentions {_quote(fresher_hints)}, so likely fresher",
        )

    return SeniorityDecision(seniority=None, source="unknown",
                             note="Neither the title nor the JD says the level")


def _phrases(pattern: re.Pattern, text: str) -> list[str]:
    """The exact phrases a pattern finds, each once, in the order they appear."""
    seen: list[str] = []
    for match in pattern.finditer(text):
        phrase = match.group(0).lower()
        if phrase not in seen:
            seen.append(phrase)
    return seen


def _quote(phrases: list[str]) -> str:
    return ", ".join(f'"{p}"' for p in phrases[:3])


# ---------- Does the JD count internships or projects as experience? ----------

INTERN = r"intern(?:ship)?s?"
COUNTS_INTERNSHIPS = re.compile(
    rf"\b(?:including|incl\.?|includes)\s+(?:\w+\s+){{0,2}}{INTERN}\b"           # "including internships"
    rf"|\b{INTERN}\s+(?:experience\s+)?(?:will\s+)?(?:also\s+)?(?:count|counts|is\s+counted|are\s+counted|"
    rf"(?:is|are|will\s+be)\s+considered|accepted|welcome)\b"                    # "internships count"
    rf"|\b(?:experience|exp\.?)\s*\((?:[^)]*\b){INTERN}\b",                      # "experience (incl. internships)"
    re.IGNORECASE)
COUNTS_PROJECTS = re.compile(
    r"\b(?:academic|personal|college|university)?\s*projects?\s+(?:experience\s+)?(?:will\s+)?(?:also\s+)?"
    r"(?:count|counts|(?:is|are|will\s+be)\s+considered|accepted)\b"
    r"|\b(?:including|incl\.?)\s+(?:academic\s+|personal\s+)?projects?\b",
    re.IGNORECASE)
ASKS_INTERNSHIP = re.compile(
    rf"\b(?:prior\s+|previous\s+|relevant\s+)?{INTERN}\s+(?:experience\s+)?(?:is\s+)?"
    rf"(?:required|mandatory|must|preferred|a\s+plus|desirable)\b"
    rf"|\b(?:at\s+least|minimum)\s+(?:one|1|an?)\s+{INTERN}\b"
    rf"|\b(?:completed|prior)\s+(?:an?\s+)?{INTERN}\b",
    re.IGNORECASE)


class ExperienceRules(BaseModel):
    """What this JD accepts as experience, read in code from its own words."""

    internships_count: bool = False
    projects_count: bool = False
    asks_internship: bool = False
    quote: str = ""


def experience_rules(jd_text: str) -> ExperienceRules:
    found = [m.group(0).strip() for p in (COUNTS_INTERNSHIPS, COUNTS_PROJECTS, ASKS_INTERNSHIP)
             for m in [p.search(jd_text)] if m]
    return ExperienceRules(internships_count=bool(COUNTS_INTERNSHIPS.search(jd_text)),
                           projects_count=bool(COUNTS_PROJECTS.search(jd_text)),
                           asks_internship=bool(ASKS_INTERNSHIP.search(jd_text)),
                           quote="; ".join(dict.fromkeys(found)))
