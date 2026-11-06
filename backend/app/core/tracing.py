"""Per-call tracing: what each LLM call cost, how long it took, which prompts produced it.

Every call through TracedLLM writes one structured log line:
  {"event": "llm_call", "step": "ResumeDraft", "model": ..., "prompt_set": "a1b2c3d4",
   "input_tokens": 2310, "output_tokens": 410, "latency_ms": 1840, "cost_usd": 0.00039, ...}
No prompt or resume text is logged (privacy). Token counts come from the provider; when it
doesn't report them they are estimated (~4 characters per token) and marked "estimated".
"""

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from pydantic import BaseModel

from app.core import prompts

log = logging.getLogger("skillbridge.llm")

# USD per 1M tokens (input, output). Check the provider's pricing page before relying on these;
# override with SKILLBRIDGE_PRICE_IN / SKILLBRIDGE_PRICE_OUT for the model you deploy.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gpt-4o-mini": (0.15, 0.60),
}


def price_for(model_name: str) -> tuple[float, float] | None:
    if os.environ.get("SKILLBRIDGE_PRICE_IN") and os.environ.get("SKILLBRIDGE_PRICE_OUT"):
        return float(os.environ["SKILLBRIDGE_PRICE_IN"]), float(os.environ["SKILLBRIDGE_PRICE_OUT"])
    short = model_name.split(":", 1)[-1]
    return PRICES.get(short)


@dataclass
class CallRecord:
    step: str
    model: str
    prompt_set: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float | None
    estimated: bool
    ok: bool
    error: str = ""


@dataclass
class TracedLLM:
    """Wraps any LLM (real or fake) with the same chat/extract interface."""

    inner: object
    model_name: str
    on_record: Callable[[CallRecord], None] | None = None
    records: list[CallRecord] = field(default_factory=list)

    def chat(self, messages: list[dict], tools: list):
        size = sum(len(str(m.get("content", ""))) for m in messages)
        return self._timed("chat", size, lambda: self.inner.chat(messages, tools),
                           lambda r: len(getattr(r, "text", "") or ""))

    def extract(self, schema: type[BaseModel], prompt: str, system: str = ""):
        return self._timed(schema.__name__, len(prompt) + len(system),
                           lambda: self.inner.extract(schema, prompt, system=system),
                           lambda r: len(r.model_dump_json()) if isinstance(r, BaseModel) else 0)

    def _timed(self, step: str, input_chars: int, call: Callable, output_chars: Callable):
        start = time.perf_counter()
        result, error = None, ""
        try:
            result = call()
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
            raise
        finally:
            usage = getattr(self.inner, "last_usage", None)
            estimated = usage is None
            if estimated:
                usage = {"input_tokens": input_chars // 4,
                         "output_tokens": output_chars(result) // 4 if result is not None else 0}
            price = price_for(self.model_name)
            cost = (usage["input_tokens"] * price[0] + usage["output_tokens"] * price[1]) / 1e6 if price else None
            record = CallRecord(step=step, model=self.model_name, prompt_set=prompts.set_version(),
                                input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
                                latency_ms=int((time.perf_counter() - start) * 1000),
                                cost_usd=round(cost, 6) if cost is not None else None,
                                estimated=estimated, ok=not error, error=error)
            self.records.append(record)
            log.info("llm_call", extra={"fields": {"event": "llm_call", **asdict(record)}})
            if self.on_record:
                self.on_record(record)

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd or 0 for r in self.records)
