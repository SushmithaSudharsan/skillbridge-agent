"""Settings from environment variables. Secrets never live in code or in the image."""

import os
from dataclasses import dataclass


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    access_code: str | None            # DEMO_ACCESS_CODE; unset = the demo stays locked (deny by default)
    daily_budget_usd: float            # stop calling the LLM once today's spend reaches this
    rate_limit_per_hour: int           # analyses + answer updates per visitor per hour
    session_ttl_minutes: int           # sessions are forgotten after this much inactivity
    max_sessions: int
    max_jd_chars: int
    trust_proxy: bool                  # read the visitor IP from X-Forwarded-For (behind Azure's proxy)
    enable_docs: bool                  # /docs is off unless asked for
    log_level: str


def load_env_file() -> None:
    """Read .env into the environment (local runs). In the cloud, real environment variables are
    used and win: existing variables are never overwritten."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:          # python-dotenv not installed: rely on real environment variables
        return
    load_dotenv(find_dotenv(usecwd=True), override=False)   # the .env in the folder you run from


def get_settings() -> Settings:
    load_env_file()
    return Settings(
        access_code=os.environ.get("DEMO_ACCESS_CODE") or None,
        daily_budget_usd=_float("DAILY_BUDGET_USD", 1.0),
        rate_limit_per_hour=_int("RATE_LIMIT_PER_HOUR", 10),
        session_ttl_minutes=_int("SESSION_TTL_MINUTES", 30),
        max_sessions=_int("MAX_SESSIONS", 200),
        max_jd_chars=_int("MAX_JD_CHARS", 15000),
        trust_proxy=os.environ.get("TRUST_PROXY", "false").lower() == "true",
        enable_docs=os.environ.get("ENABLE_DOCS", "false").lower() == "true",
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )
