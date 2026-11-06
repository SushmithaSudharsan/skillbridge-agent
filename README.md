# SkillBridge

**An honest resume checker for a specific job.** Paste a job description and pick a resume. SkillBridge tells you whether the job is a realistic target right now, what to learn, what you already have but your resume hides, and which things to check before you apply.

It never tells you to add skills you don't have. No keyword stuffing.

> **Demo:** _link here_ · **Video:** _link here_ (the demo uses fictional sample resumes only)

---

## Why this exists

Most resume tools give you a match percentage and tell you to add more keywords. That is hard to trust and it pushes people towards exaggerating. SkillBridge does the opposite:

- It says **why** you are or aren't a fit, quoting the job description and your resume.
- It separates a **skill gap** (you should learn it) from a **presentation gap** (you have it, but your resume doesn't show it).
- It treats **soft skills** the way recruiters do: you show them through what you did, and you prepare an example for the interview. They are never scored as a gap.

## What you get

| Result | What it means |
| --- | --- |
| **Fit verdict** | Strong fit / Possible fit / Stretch / Not yet. Worked out in code from the role's level, your experience and how many must-have skills you meet. |
| **Likely auto-filters** | Work authorisation, location, degree and minimum years, quoted from the job. Employers' systems often filter on these before anyone reads your skills. |
| **Learn** | Skills and experience the job needs that your resume doesn't show, with a learning plan. |
| **Show better** | Things you have but only list, or show in a different context. A presentation problem, not a learning gap. |
| **Related skills** | Using TensorFlow when the job wants PyTorch is labelled "related, not the same". It helps, but never counts as meeting the requirement. |
| **Ways of working** | Soft skills the job mentions. Not scored. You get "shown on your resume" or "prepare one example for the interview". |
| **Questions** | A few questions whose answers can change the plan. An answer can raise a level (capped at Intermediate) or lower one you oversold. |
| **PDF report** | Download the full report. |

If you are more senior than the role, that is not treated as a gap or a penalty. You are told to be ready to explain why you want a role at this level.

## The idea in one line

**The AI reads and writes. Python checks.**

Every claim the AI makes must quote the resume or the job description word for word, or it is thrown away. The fit verdict, levels and learning order are calculated in code, so the AI cannot overrule them.

## How an analysis works

```
Resume + job description
   │
   ├─ 1. Mask personal details (code)         names, contacts, employers, universities, cities
   ├─ 2. Read the job description (AI)        what it asks for, each with a quote
   ├─ 3. Read the resume (AI)                 evidence for each requirement, each with a quote
   ├─ 4. Compare (code)                       levels, implied skills, related skills, blockers, fit verdict
   ├─ 5. Advise (AI, then checked by code)    summary, project ideas, "show better" tips
   └─ 6. Your answers (AI + code)             update the plan, never inflate it
```

Only three steps call the AI. Everything that decides the result is ordinary code that you can test.

## Fair by design

- **Identity can't change the result.** Tests swap name, university, employer and city, including an elite college against an unknown one. The text sent to the AI must be identical every time.
- **Writing style shouldn't change the result.** With a real model, `run_fairness.py --variants` checks that the same facts give the same gaps and verdict when the resume has a career break, shuffled sections, AI-polished wording or plain-English wording.
- **Honest about its limits.** Nothing is claimed that the text doesn't support, and the report says how many AI claims were removed.

## Privacy

- Resumes are read in memory, masked before any AI call, and never saved.
- The web demo only analyses **fictional sample resumes**: no uploads, no personal data on the server.
- Sessions live in memory for 30 minutes and are deleted when you close the tab.
- Logs record tokens, cost and time per call. They never record text.

## Run it yourself

You need Python 3.11+ and a Gemini API key.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,web,llm]"
copy .env.example .env            # then add your key, model and an access code

pytest                            # tests need no API key
uvicorn app.main:app --app-dir backend --reload     # http://localhost:8000
```

Analyse your own resume from the command line:

```bash
python -m app.cli analyze --resume private/me.pdf --jd private/job.txt --years 0
```

Extras: `--review` (AI judge), `--debate` (three-advisor panel), `chat`, `benchmark`, `calibrate`, `privacy`.

Settings live in `.env` (see `.env.example`): model, access code, daily spend cap, rate limit.

## Cost safety

The web demo needs an access code, limits each visitor per hour, and stops when a daily spend cap is reached. It refuses to start if the model's price is unknown, because the cap could not work. Set `SKILLBRIDGE_PRICE_IN` and `SKILLBRIDGE_PRICE_OUT` for a new model.

## How it is measured

| Question | Metric | Where |
| --- | --- | --- |
| Are the gaps right? | Precision, recall, F1 against labelled examples | `evals/runners/run_benchmark.py` |
| Better than a plain prompt? | Same metrics against a one-prompt baseline | same |
| Same answer every time? | Overlap (Jaccard) of gaps across runs | same |
| Fair across writing styles? | Overlap and verdict agreement across variants | `evals/runners/run_fairness.py` |
| Can the AI judge be trusted? | Cohen's kappa against human scores | `python -m app.cli calibrate` |
| What does it cost? | Tokens, time and USD per call | JSON logs |

The labels in `evals/datasets/cases.json` are drafts. Review them and add real job descriptions before quoting any numbers.

## Project layout

```
backend/app/services   the analysis logic: fit, blockers, related skills, reports, AI calls
backend/app/api        web routes (thin)
backend/app/core       settings, logging, cost tracing, prompt loader
backend/data           skill vocabulary, learning guides, fictional sample resumes and jobs
backend/tests          unit, fairness and API tests
prompts/               every AI prompt as its own versioned file
evals/                 datasets and the quality gates
frontend/              the demo page (plain HTML, CSS, JavaScript)
deploy/                Dockerfile and docker-compose
.github/workflows      tests, lint and eval checks on every change
```

## Why the prompts are separate files

Each prompt has a `version:` at the top and lives in its own file. A prompt change shows up as a one-file diff, and every logged AI call records which prompt version it used. The eval workflow blocks changes that lower quality or raise cost.

## Built with

Python, FastAPI, Pydantic, LangChain with Gemini, ReportLab, pytest, Docker, GitHub Actions.

## Limits

- It is guidance, not a guarantee of any hiring outcome.
- Related-skill pairings and the learning guide are hand-written and cover common tech roles best.
- Work authorisation and location can't be read from a resume, so they are always marked "check".
