"""Fairness by construction: who you are must not change what the AI sees.

For every sample resume we swap the identity (name, email, phone, university, employer, city)
for very different ones. After masking, the text sent to the LLM must be IDENTICAL, so the
result cannot depend on gender, religion, region or which college or company someone is from.
(Style fairness, the same facts written differently, needs a real LLM: see evals/runners.)
"""

import pytest

from app.services.guardrails import mask_pii
from app.services.samples import load_samples

IDENTITIES = [
    ("Lakshmi Venkataraman", "lakshmi.v@example.com", "+91 94440 55667", "Madras Christian College",
     "Sundaram Textiles", "Coimbatore"),
    ("Imran Qureshi", "imran.qureshi@example.com", "+91 98110 22446", "Jamia Millia Islamia",
     "Crescent Logistics", "Lucknow"),
    ("Michael O'Connor", "m.oconnor@example.com", "+1 617 555 0199", "Boston University",
     "Harbor Insurance", "Boston"),
    ("Nguyen Thi Lan", "lan.nguyen@example.com", "+84 90 555 1234", "Hanoi University of Science and Technology",
     "Lotus Devices", "Hanoi"),
    # college-tier extremes: an elite name and an unknown one must look identical after masking
    ("Aarav Mehta", "aarav.mehta@example.com", "+91 99000 11223", "Indian Institute of Technology Bombay",
     "Google India", "Mumbai"),
    ("Kavitha Rao", "kavitha.rao@example.com", "+91 90000 33445", "Sri Venkateshwara Engineering College",
     "Local Software Services", "Nellore"),
]

SAMPLES = [s for s in load_samples("resumes") if "identity" in s.meta]


def swap(text: str, old: tuple, new: tuple) -> str:
    pairs = [(o, n) for o, n in zip(old, new) if o.strip()]
    for o, n in sorted(pairs, key=lambda p: -len(p[0])):          # longest first
        text = text.replace(o, n)
        for part_o, part_n in zip(o.split(), n.split()):            # first/last names alone
            if len(part_o) > 2 and o in pairs[0]:
                text = text.replace(part_o, part_n)
    return text


@pytest.mark.parametrize("sample", SAMPLES, ids=[s.id for s in SAMPLES])
def test_identity_swaps_leave_the_llm_input_unchanged(sample):
    original = tuple(p.strip() for p in sample.meta["identity"].split("|"))
    baseline = mask_pii(sample.text)
    for other in IDENTITIES:
        swapped = swap(sample.text, original, other)
        assert swapped != sample.text
        assert mask_pii(swapped) == baseline, f"{sample.id}: identity leaks for {other[0]}"


def test_college_tier_cannot_reach_the_llm():
    """IIT Bombay and a small unranked college give the same LLM input."""
    elite, small = IDENTITIES[-2], IDENTITIES[-1]
    for sample in SAMPLES:
        original = tuple(p.strip() for p in sample.meta["identity"].split("|"))
        a = mask_pii(swap(sample.text, original, elite))
        b = mask_pii(swap(sample.text, original, small))
        assert a == b, sample.id
        assert "Bombay" not in a and "Venkateshwara" not in b
