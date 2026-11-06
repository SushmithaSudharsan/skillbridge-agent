# SkillBridge

Honest resume-to-job gap analysis. Paste a job description, and SkillBridge says whether the job
is a realistic target now, what to **learn**, what you already have but should **show better**,
and asks questions whose answers can change the plan. It never tells you to add skills you don't
have: no keyword stuffing.

## Design in one line

**The LLM reads and writes; Python verifies.** Every claim the LLM makes must quote the resume or
the JD word for word, or it is dropped. Fit, gaps, levels and the learning order are computed in code.

## How it works

| Step | What | LLM calls |
| --- | --- | --- |
| Mask | names, contacts, employers, universities, cities removed **by position** (contact block, employer slot, degree line), not by name lists | 0 |
| Read the JD | requirements of three kinds: skill, task/experience, way of working; each with a quote | 1 |
| Read the resume | evidence for each requirement (any field, open vocabulary), each with a quote | 1 |
| Compare (code) | level ceilings by evidence type; skills implied by real work; fit verdict from level + must-haves | 0 |
| Advise | summary, portfolio projects, "show better" suggestions, learning areas for uncurated skills; all checked | 1 |
| Your answers | can raise a level (capped at Intermediate) or lower an oversold one; plan is rebuilt | 1-2 |
| Optional | LLM judge with retry (`--review`), 3-advisor debate (`--debate`), follow-up chat agent | more |

## Honest and fair by design

- **Fit is computed, not guessed**: Strong fit / Possible fit / Stretch / Not yet, from the role's level vs your
  experience (internships and projects count only when the JD says so) and must-haves met.
- **Learn / Show better / Strengths**: a skill you have but only list, or show in a different context, is a
  presentation problem, not a learning gap.
- **Identity can't change the result**: tests swap name, university, employer and city across very different
  identities; the text sent to the LLM must be byte-for-byte identical (`backend/tests/test_fairness.py`).
- **Style fairness** (same facts, different writing) is measured with a real LLM: `evals/runners/run_fairness.py`.

## Privacy

Resumes are read in memory, masked before any LLM call, never stored. The web demo only analyses fictional
sample resumes, keeps sessions in memory for 30 minutes and deletes them when the tab closes. Logs record
tokens, cost and latency per call, never text.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows (source .venv/bin/activate on macOS/Linux)
pip install -e ".[dev,web,llm]"
pip freeze > requirements.lock                          # exact versions you tested with
copy .env.example .env                                  # then fill in your key, model and access code

pytest                                                  # 260+ tests, no API key needed
uvicorn app.main:app --app-dir backend --reload         # web demo on http://localhost:8000
python -m app.cli analyze --resume private/me.pdf --jd private/job.txt --years 0   # CLI, your own resume
docker compose -f deploy/docker-compose.yml up --build  # the production image, locally
```

CLI extras: `--review` (LLM judge), `--debate` (advisor panel), `chat`, `benchmark`, `calibrate`, `privacy`.

## Measuring it

| Question | Metric | Where |
| --- | --- | --- |
| Are the gaps right? | precision / recall / F1 vs labelled pairs | `evals/runners/run_benchmark.py` |
| Better than a plain prompt? | same metrics for a one-prompt baseline, judge win rate with 95% Wilson interval | same |
| Same answer every run? | mean Jaccard of gap sets across runs | same |
| Fair across writing styles? | Jaccard between twin resumes (same facts) | `evals/runners/run_fairness.py` |
| Can the judge be trusted? | Cohen's kappa vs human scores | `python -m app.cli calibrate` |
| What does it cost? | tokens, latency and USD per call, per analysis | JSON logs from `app/core/tracing.py` |

The labels in `evals/datasets/cases.json` are **drafts**; review them and add real JDs before quoting numbers.
CI (`.github/workflows/evals.yml`) blocks prompt or analysis changes that drop precision, recall or raise cost.

## Layout

```
backend/app/api        HTTP routes (thin)          backend/app/services   analysis logic, LLM, reports
backend/app/core       config, logging, tracing,   backend/app/schemas    request validation
                       prompt loader               backend/data           vocabulary, guides, fictional samples
backend/tests          unit, fairness, API tests   prompts/               versioned prompt files
evals/                 datasets and gate runners   frontend/              the demo page (HTML/CSS/JS)
deploy/                Dockerfile, compose         .github/workflows      CI and eval gate
```
