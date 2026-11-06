"""Is this job a realistic target right now? Decided in code, never by the LLM.

Two facts decide it:
  1. Level:  what the job asks for (fresher / mid / senior, from years or wording in the JD)
             versus the student's full-time work experience (internships and projects
             don't count as years).
  2. Skills: how many of the JD's must-have skills the resume already meets.

  two levels apart (fresher vs senior)       -> Not yet
  one level apart (fresher vs mid)           -> Stretch
  same level or above (more senior than the role is a risk to explain, never a penalty):
                       >= 80% must-haves met -> Strong fit
                       >= 50%                -> Possible fit
                       less                  -> Stretch
What the JD accepts as experience also counts (read from its own words, in code):
  "internships count" / "including internships" -> internship months are added to your years
  "projects will be considered"                 -> one level closer (projects are your experience)
  "internship experience preferred/required"    -> flagged if you have none; a Strong fit becomes
                                                    Possible until you show one
The verdict is given to the advisor LLM as a fact it may not contradict.
"""

from typing import Literal

from pydantic import BaseModel

from app.services.experience import ExperienceRules, seniority_from_years

RANK = {"fresher": 0, "mid": 1, "senior": 2}
NAME = {"fresher": "entry-level", "mid": "mid-level", "senior": "senior"}
NEXT_ROLE = {"fresher": "fresher, junior or associate", "mid": "mid-level", "senior": "senior"}

Verdict = Literal["Strong fit", "Possible fit", "Stretch", "Not yet"]


class Fit(BaseModel):
    verdict: Verdict
    advice: str                 # one or two sentences for the student
    reasons: list[str]          # the facts behind it
    role_level: str | None
    role_years: float | None
    your_years: float | None
    must_met: int
    must_total: int
    overqualified: bool = False

    def fact(self) -> str:
        """The line the LLMs see."""
        return f"Overall fit (computed from the facts, do not contradict): {self.verdict}. {self.advice}"


def _years(y: float) -> str:
    return "no full-time work experience" if y == 0 else f"{y:g} year{'s' if y != 1 else ''} of full-time work"


def assess_fit(role_level: str | None, role_years: float | None, your_years: float | None,
               must_met: int, must_total: int, internship_months: float = 0,
               rules: ExperienceRules | None = None) -> Fit:
    rules = rules or ExperienceRules()
    reasons = []
    if role_years == 0:
        reasons.append("The job is open to freshers (no years of experience required).")
    elif role_years is not None:
        reasons.append(f"The job asks for {role_years:g}+ years of experience.")
    elif role_level:
        reasons.append(f"The job reads as a {NAME[role_level]} role.")
    else:
        reasons.append("The job description doesn't say what level it is.")
    if your_years is not None:
        note = "." if your_years >= 2 or rules.internships_count else " (internships and projects don't count as years here)."
        reasons.append(f"You have {_years(your_years)}" + (f" and {internship_months:g} months of internship"
                                                          if internship_months else "") + note)
    if must_total:
        reasons.append(f"Your resume meets {must_met} of {must_total} must-have skills.")

    counted = your_years
    if your_years is not None and rules.internships_count and internship_months:
        counted = your_years + internship_months / 12
        reasons.append(f"The job counts internships as experience, so your {internship_months:g} months of "
                       f"internship count too ({counted:.1f} years in total).")
    coverage = must_met / must_total if must_total else 1.0
    you = seniority_from_years(counted) if counted is not None else None
    distance = RANK[role_level] - RANK[you] if role_level and you else 0
    if rules.projects_count and distance >= 1:
        distance -= 1
        reasons.append("The job says projects count as experience, so your projects bring you one level closer.")

    overqualified = bool(role_level and you and RANK[you] > RANK[role_level])
    if overqualified:
        reasons.append(f"You have more experience than this {NAME[role_level]} role asks for. That is not a "
                       f"reason you can't do it; it is something a recruiter may ask about.")

    if distance >= 2:
        verdict = "Not yet"
        advice = (f"This is a {NAME[role_level]} role and you have {_years(your_years)}, so it isn't a realistic "
                  f"application yet. Apply to {NEXT_ROLE[you]} roles now and use this posting as a target "
                  f"for later.")
    elif distance == 1:
        verdict = "Stretch"
        advice = (f"This role expects more experience than you have ({_years(your_years)}). Apply only if your "
                  f"projects closely match it; {NEXT_ROLE[you]} roles are a better bet now.")
    else:
        lead = ("You are more experienced than this role asks, and" if overqualified
                else "Your level matches and")
        if coverage >= 0.8:
            verdict, advice = "Strong fit", f"{lead} you meet most must-haves: apply, and close the few gaps below."
        elif coverage >= 0.5:
            verdict, advice = "Possible fit", f"{lead} several must-haves are missing: close the top gaps before applying."
        else:
            verdict, advice = "Stretch", f"{lead} most must-have skills are missing: work through the plan first."
        if overqualified:
            advice += (" Be ready to say why you want a role at this level (a move to a new field, a better "
                       "team, a change of pace), since recruiters often ask.")
    if rules.asks_internship:
        if internship_months:
            reasons.append(f"The job asks for internship experience: your {internship_months:g} months meet that.")
        else:
            reasons.append("The job asks for internship experience, and you haven't listed one.")
            advice += " It also asks for an internship: look for one, or show project work of the same depth."
            if verdict == "Strong fit":
                verdict = "Possible fit"
    if you is None and role_level:
        advice += " (Tell us your years of experience for a fuller verdict.)"
    return Fit(verdict=verdict, advice=advice, reasons=reasons, role_level=role_level, role_years=role_years,
               your_years=your_years, must_met=must_met, must_total=must_total,
               overqualified=overqualified)
