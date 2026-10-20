"""Work out a job's seniority from its title and job description."""

import re

from skillbridge.models import SeniorityDecision

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
