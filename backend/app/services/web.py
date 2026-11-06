"""The web demo's logic, independent of the web framework (so it is fully unit-tested).

Rules it enforces (the frontend only displays):
  - access code required; if none is configured the demo stays locked
  - sample resumes only: no uploads, so no real personal data ever reaches the server
  - per-visitor rate limit, and a daily spend cap counted from real token usage
  - sessions live in memory only, expire after inactivity, and are deleted on request
"""

import hmac
import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Callable

from app.core.config import Settings
from app.core.tracing import price_for, CallRecord, TracedLLM
from app.services.analysis import Analysis, advise, assemble, prepare
from app.services.answers import rerun_with_answers
from app.services.catalog import Catalog, load_catalog
from app.services.learning import LEVEL_NAMES, load_guide
from app.services.matching import load_matcher
from app.services.report_pdf import render_pdf
from app.services.samples import get_sample, load_samples

log = logging.getLogger("skillbridge.web")


class DemoError(Exception):
    status = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class Forbidden(DemoError):
    status = 403


class NotFound(DemoError):
    status = 404


class TooMany(DemoError):
    status = 429


class Unavailable(DemoError):
    status = 503


class SessionStore:
    """In-memory only. Nothing is written to disk or a database."""

    def __init__(self, ttl_seconds: int, max_items: int, clock: Callable[[], float] = time.monotonic):
        self.ttl, self.max_items, self.clock = ttl_seconds, max_items, clock
        self._items: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def put(self, value: dict) -> str:
        with self._lock:
            self._purge()
            if len(self._items) >= self.max_items:      # drop the oldest rather than grow without limit
                oldest = min(self._items, key=lambda k: self._items[k][0])
                del self._items[oldest]
            key = uuid.uuid4().hex
            self._items[key] = (self.clock(), value)
            return key

    def get(self, key: str) -> dict:
        with self._lock:
            self._purge()
            if key not in self._items:
                raise NotFound("This session has ended or expired. Start a new analysis.")
            _, value = self._items[key]
            self._items[key] = (self.clock(), value)    # activity keeps it alive
            return value

    def delete(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)

    def __len__(self) -> int:
        return len(self._items)

    def _purge(self) -> None:
        now = self.clock()
        for key in [k for k, (t, _) in self._items.items() if now - t > self.ttl]:
            del self._items[key]


class Budget:
    """Today's LLM spend (UTC day) and per-visitor rate limits."""

    def __init__(self, daily_usd: float, per_hour: int, clock: Callable[[], float] = time.time):
        self.daily_usd, self.per_hour, self.clock = daily_usd, per_hour, clock
        self.spent, self.day = 0.0, self._today()
        self.calls: dict[str, deque] = {}
        self._lock = threading.Lock()

    def _today(self) -> str:
        return datetime.fromtimestamp(self.clock(), timezone.utc).strftime("%Y-%m-%d")

    def record(self, call: CallRecord) -> None:
        with self._lock:
            self._roll()
            self.spent += call.cost_usd or 0.0

    def check(self, visitor: str) -> None:
        with self._lock:
            self._roll()
            if self.spent >= self.daily_usd:
                raise TooMany("Today's demo budget is used up. Please try again tomorrow.")
            recent = self.calls.setdefault(visitor, deque())
            now = self.clock()
            while recent and now - recent[0] > 3600:
                recent.popleft()
            if len(recent) >= self.per_hour:
                raise TooMany("You've reached the demo limit for this hour. Please try again later.")
            recent.append(now)

    def _roll(self) -> None:
        if self._today() != self.day:
            self.spent, self.day = 0.0, self._today()


class DemoService:
    def __init__(self, settings: Settings, llm_factory: Callable[[], TracedLLM], catalog: Catalog | None = None):
        self.settings = settings
        self.llm_factory = llm_factory
        self.catalog = catalog or load_catalog()
        self.guide = load_guide()
        self.matcher = load_matcher(self.catalog.skills)
        self.sessions = SessionStore(settings.session_ttl_minutes * 60, settings.max_sessions)
        self.budget = Budget(settings.daily_budget_usd, settings.rate_limit_per_hour)

    # ---------- access ----------

    def check_access(self, code: str | None) -> None:
        if not self.settings.access_code:
            raise Unavailable("The demo isn't configured yet (no access code set).")
        if not code or not hmac.compare_digest(code.encode(), self.settings.access_code.encode()):
            raise Forbidden("Wrong or missing access code.")

    # ---------- endpoints ----------

    def samples(self) -> dict:
        def listing(kind: str) -> list[dict]:
            return [{"id": s.id, "title": s.title, **{k: v for k, v in s.meta.items() if k != "identity"},
                     "text": s.text if kind == "jds" else _preview(s.text)}
                    for s in load_samples(kind) if s.meta.get("demo") != "no"]
        return {"resumes": listing("resumes"), "jds": listing("jds")}

    def analyze(self, visitor: str, sample_id: str, jd_text: str, title: str = "",
                years: float = 0, internship_months: float = 0) -> dict:
        sample = get_sample("resumes", sample_id)
        if sample is None or sample.meta.get("demo") == "no":
            raise NotFound("Unknown sample resume.")
        if len(jd_text) > self.settings.max_jd_chars:
            raise DemoError(f"The job description is too long (max {self.settings.max_jd_chars} characters).")
        self.budget.check(visitor)
        llm = self._llm()
        try:
            prep = prepare(llm, sample.text, jd_text, self.catalog, self.matcher, self.guide, title,
                           years=years, internship_months=internship_months)
            analysis = assemble(prep, *advise(llm, prep, self.matcher))
        except ValueError as error:
            raise DemoError(str(error)) from error
        except Exception as error:                 # provider down, timeout, rate limit: say so clearly
            log.exception("analysis_failed", extra={"fields": {"event": "analysis_failed"}})
            raise Unavailable("The AI service didn't respond. Please try again in a minute.") from error
        session_id = self.sessions.put({"prep": prep, "analysis": analysis, "llm": llm, "visitor": visitor})
        log.info("analysis", extra={"fields": {"event": "analysis", "calls": len(llm.records),
                                               "cost_usd": round(llm.total_cost, 6), "fit": analysis.fit.verdict}})
        return {"session_id": session_id, **to_view(analysis, llm)}

    def answer(self, visitor: str, session_id: str, answers: list[str]) -> dict:
        session = self.sessions.get(session_id)
        analysis: Analysis = session["analysis"]
        expected = len(analysis.questions) + len(analysis.gap_questions)
        if len(answers) != expected:
            raise DemoError(f"Expected {expected} answers (use an empty string to skip one).")
        if any(a.strip() for a in answers):
            self.budget.check(visitor)
        analysis.answers = [a.strip() for a in answers]
        llm = session["llm"]
        try:
            new, new_prep = rerun_with_answers(llm, session["prep"], analysis, self.matcher, self.catalog.skills,
                                               self.guide)
        except Exception as error:
            log.exception("answers_failed", extra={"fields": {"event": "answers_failed"}})
            raise Unavailable("The AI service didn't respond. Your answers are kept; try again.") from error
        session.update(analysis=new, prep=new_prep)
        return {"session_id": session_id, **to_view(new, llm)}

    def report_pdf(self, session_id: str) -> bytes:
        return render_pdf(self.sessions.get(session_id)["analysis"])

    def end(self, session_id: str) -> None:
        self.sessions.delete(session_id)

    # ---------- helpers ----------

    def _llm(self) -> TracedLLM:
        try:
            llm = self.llm_factory()
        except RuntimeError as error:              # e.g. SKILLBRIDGE_MODEL not set
            raise Unavailable("The demo's AI model isn't configured.") from error
        model = getattr(llm, "model_name", None)
        if model and price_for(model) is None:     # an unpriced model would cost $0 on paper and never hit the cap
            raise Unavailable("The demo is off: the AI model has no known price, so the spend cap can't work. "
                              "Set SKILLBRIDGE_PRICE_IN and SKILLBRIDGE_PRICE_OUT (USD per million tokens).")
        llm.on_record = self.budget.record
        return llm


def _preview(text: str, lines: int = 14) -> str:
    return "\n".join(text.splitlines()[:lines])


def to_view(a: Analysis, llm: TracedLLM | None = None) -> dict:
    """The analysis as plain JSON for the page. Display-only: every decision was made above."""
    def row(m) -> dict:
        return {"skill": m.skill, "kind": m.kind, "status": m.status, "must_have": m.must_have,
                "current": LEVEL_NAMES[m.current], "needed": LEVEL_NAMES[m.needed],
                "job_quote": m.job_quote, "resume_quote": m.resume_quote, "visibility": m.visibility,
                "related": ({"skill": m.related, "level": LEVEL_NAMES[m.related_level], "quote": m.related_quote}
                            if m.related else None)}
    must = [m for m in a.matches if m.must_have and not m.is_soft]
    hard = [m for m in a.matches if not m.is_soft]
    return {
        "title": a.title, "seniority": a.seniority, "summary": a.summary,
        "fit": a.fit.model_dump(include={"verdict", "advice", "reasons"}) if a.fit else None,
        "counts": {"requirements": len(hard), "met": len(hard) - len(a.gaps), "gaps": len(a.gaps),
                   "must_gaps": sum(m.status != "meets" for m in must)},
        "learn": [row(m) for m in a.gaps],
        "soft": {"shown": [row(m) for m in a.soft["shown"]],
                 "interview": [{"skill": m.skill, "job_quote": m.job_quote} for m in a.soft["interview"]]},
        "blockers": [b.model_dump() for b in a.blockers],
        "overqualified": bool(a.fit and a.fit.overqualified),
        "show_better": [b.model_dump() for b in a.show_better],
        "strengths": [row(m) for m in a.strengths],
        "learning": [{"skill": s.skill, "level": LEVEL_NAMES[s.level], "area": s.area, "topics": s.topics,
                      "reason": s.reason, "suggested": s.suggested} for s in a.learning],
        "projects": [p.model_dump() for p in a.projects],
        "checks": [{k: c[k] for k in ("ok", "area", "text")} for c in a.checks],
        "questions": a.asked(),
        "changes": [{"skill": c.skill, "before": LEVEL_NAMES[c.before], "after": LEVEL_NAMES[c.after],
                     "said": c.said, "where": c.where} for c in a.changes],
        "removed_claims": len(a.dropped),
        "usage": {"calls": len(llm.records), "cost_usd": round(llm.total_cost, 5)} if llm else None,
    }
