"""Tests for working out seniority from titles and job descriptions."""

import pytest

from skillbridge.experience import decide_seniority, seniority_from_years, years_required


# ---------- Reading years from a JD ----------

@pytest.mark.parametrize(
    "jd, expected",
    [
        ("We need 3-5 years of experience in Python.", 3),
        ("Experience: 5+ years", 5),
        ("Minimum 4 years of relevant experience.", 4),
        ("At least 2 yrs exp in SQL", 2),
        ("3 to 6 years of industry experience", 3),
        ("0-1 year experience", 0),
        ("2.5 years of experience preferred", 2.5),
        ("3+ years of experience overall; 1+ year of experience with LLMs", 3),
    ],
)
def test_years_are_read(jd, expected):
    assert years_required(jd) == expected


@pytest.mark.parametrize(
    "jd",
    [
        "We are a fast-growing startup building ML products.",
        "Our company was founded 20 years ago in Bengaluru.",   # years, but not experience
        "",
    ],
)
def test_no_experience_years_gives_none(jd):
    assert years_required(jd) is None


# ---------- Years -> seniority ----------

@pytest.mark.parametrize(
    "years, expected",
    [(0, "fresher"), (1.9, "fresher"), (2, "mid"), (4.5, "mid"), (5, "senior"), (10, "senior")],
)
def test_seniority_from_years(years, expected):
    assert seniority_from_years(years) == expected


# ---------- Putting title and JD together ----------

def test_jd_years_beat_the_title():
    decision = decide_seniority("senior", "Looking for 2-4 years of experience.")
    assert decision.seniority == "mid"
    assert decision.source == "jd_years"
    assert "title suggests senior" in decision.note


def test_jd_years_used_when_title_is_silent():
    decision = decide_seniority(None, "5-6 years of experience in ML.")
    assert decision.seniority == "senior"


@pytest.mark.parametrize(
    "jd",
    ["Freshers are welcome to apply!", "This is an entry-level role.",
     "Recent graduates encouraged.", "No prior experience required."],
)
def test_fresher_words_in_jd(jd):
    decision = decide_seniority(None, jd)
    assert decision.seniority == "fresher"
    assert decision.source == "jd_words"


def test_title_used_when_jd_is_silent():
    decision = decide_seniority("senior", "Build great ML systems with us.")
    assert decision.seniority == "senior"
    assert decision.source == "title"


def test_unknown_when_nothing_says():
    decision = decide_seniority(None, "Build great ML systems with us.")
    assert decision.seniority is None
    assert decision.source == "unknown"


# ---------- Weak hints when nothing else says ----------

PEOPLEX_JD = """We are seeking a versatile AI Engineer with strong Python skills.
You'll collaborate closely with leadership and mentor junior team members while
shaping the company's technical direction.
* Mentor interns and junior engineers to build a strong learning-oriented tech culture."""


def test_mentoring_juniors_hints_mid():
    decision = decide_seniority(None, PEOPLEX_JD)
    assert decision.seniority == "mid"
    assert decision.source == "jd_hints"
    assert '"mentor junior"' in decision.note        # quotes the real words it found


@pytest.mark.parametrize(
    "jd",
    ["You will be mentored by senior engineers.",
     "Learn from experienced data scientists in a supportive team."],
)
def test_being_mentored_hints_fresher(jd):
    decision = decide_seniority(None, jd)
    assert decision.seniority == "fresher"
    assert decision.source == "jd_hints"


def test_hints_pointing_both_ways_give_unknown():
    jd = "You will be mentored by our CTO and later lead a team of analysts."
    decision = decide_seniority(None, jd)
    assert decision.seniority is None
    assert "both ways" in decision.note


def test_title_level_word_beats_hints():
    decision = decide_seniority("senior", "You will be mentored by our CTO.")
    assert decision.seniority == "senior"
    assert decision.source == "title"


def test_a_mentor_noun_is_not_a_hint():
    decision = decide_seniority(None, "Every new hire gets a mentor in week one.")
    assert decision.source == "unknown"
