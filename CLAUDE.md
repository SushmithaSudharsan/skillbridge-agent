# SkillBridge: notes for AI assistants

Follow the owner's `ai-ml-project-standards` skill. Ask before changing the stack, folders or dependencies.
Never deploy, publish or change cloud infrastructure: prepare files and steps only.

## Invariants (do not break)
- The LLM reads and writes; Python verifies. Every LLM claim is checked against source text (`quote_is_in`).
- Fit, gaps, levels and learning order are computed in code. The advisor may not contradict the fit.
- Personal data: mask before any LLM call (`guardrails.mask_pii`, masking by position). No resume or report is
  stored. The web demo accepts sample resumes only.
- Prompts live in `prompts/*.md`. Bump the `version:` header whenever wording changes.
- Level ceilings: skills list = Beginner; study = Intermediate; answers raise at most to Intermediate.
- Anything that can be decided by rules must not be left to the LLM (e.g. resume questions are rule-based).

## Commands
- Tests: `pytest` (no API key; a scripted FakeLLM is used). Fairness: `pytest backend/tests/test_fairness.py`.
- Web: `uvicorn app.main:app --app-dir backend --reload`. CLI: `python -m app.cli --help`.
- Real-LLM evals: `python evals/runners/run_benchmark.py --runs 1 --limit 5` (costs a few cents).

## Where things go
- Routes: `backend/app/api/`; logic: `backend/app/services/`; config/logging/tracing/prompts: `backend/app/core/`.
- New sample resumes/JDs: `backend/data/sample_*/` with `# title:`, `# role:`, `# level:`, `# identity:` headers.
- New eval cases: `evals/datasets/cases.json` (mark who labelled them).
