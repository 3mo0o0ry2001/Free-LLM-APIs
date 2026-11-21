# Free LLM APIs Router

A command-line Multi-LLM Router that sends the same prompt to every configured provider in parallel, evaluates the usable answers, and ranks them using a weighted score.

The project currently supports Groq, Gemini, OpenRouter, Cohere, and Mistral through OpenAI-compatible APIs. A provider is active only when its API key is present, so you can run the router with any subset of them.

> API availability, model access, quotas, and pricing are controlled by each provider and may change. The project does not guarantee that every configured model remains free or available.

## Features

- Parallel requests to all active providers with `asyncio`
- Provider-specific timeouts, token limits, and retry counts
- Retries for timeouts and HTTP `429`, `500`, `502`, `503`, and `504`
- Detection of failed and truncated responses
- Anonymous answer judging with Groq when it is configured
- Neutral quality score when Groq is unavailable as the judge
- Weighted ranking and automatic best/fallback selection
- No hardcoded credentials

## How it works

```text
User prompt
    |
    v
Parallel provider requests
    |-- Groq
    |-- Gemini
    |-- OpenRouter
    |-- Cohere
    `-- Mistral
    |
    v
Timeout, retry, failure, and truncation handling
    |
    v
Anonymous quality judge
    |
    v
Weighted scoring
    |-- Quality:     40%
    |-- Completion:  25%
    |-- Latency:     20%
    `-- Reliability: 15%
    |
    v
Best provider + fallback provider
```

Inference requests run in parallel. Quality evaluation runs afterward, so the displayed inference time and judge time are reported separately.

## Provider configuration

The defaults live in `PROVIDER_CONFIG` inside `app.py`.

| Provider | Environment variable | Default model | Timeout | Max tokens | Retries |
|---|---|---|---:|---:|---:|
| Groq | `GROQ_API_KEY` | `openai/gpt-oss-20b` | 10s | 700 | 1 |
| Gemini | `GEMINI_API_KEY` | `gemini-3.8-flash` | 12s | 700 | 0 |
| OpenRouter | `OPENROUTER_API_KEY` | `nvidia/nemotron-3-ultra-550b-a55b:free` | 10s | 700 | 0 |
| Cohere | `COHERE_API_KEY` | `command-a-plus-05-2026` | 10s | 1000 | 1 |
| Mistral | `MISTRAL_API_KEY` | `mistral-small-latest` | 10s | 700 | 0 |

If a key is missing, the provider is skipped and the application continues with the remaining configured providers.

## Scoring

Each usable response receives a score from 0 to 100:

```text
Final score =
    quality * 0.40
  + completion * 0.25
  + latency * 0.20
  + reliability * 0.15
```

- **Quality:** the anonymous judge score. If Groq is unavailable or judging fails, usable answers receive `50`.
- **Completion:** `100` for a completed response and `50` for a truncated response.
- **Latency:** decreases linearly from `100` to `0` as latency approaches the provider timeout.
- **Reliability:** `100` on the first successful attempt and `70` after a retry.
- **Failed requests:** always receive a final score of `0`.

The highest-scoring usable response becomes the best provider. The next highest-scoring response becomes the fallback.

## Requirements

- Windows PowerShell
- Python 3.13 or another Python 3 version compatible with the pinned dependencies
- At least one provider API key

## Setup on Windows

1. Open PowerShell in the project folder.

2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation for the current process, run:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Create your local environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

5. Open `.env` and add one or more keys:

   ```dotenv
   GROQ_API_KEY=your_key_here
   GEMINI_API_KEY=your_key_here
   OPENROUTER_API_KEY=your_key_here
   COHERE_API_KEY=your_key_here
   MISTRAL_API_KEY=your_key_here
   ```

   Leave unused values empty or remove their lines. Never commit `.env`.

## Run

```powershell
python app.py
```

Enter a prompt when `You:` appears. Type `exit` to close the application.

On startup, the application lists active providers and reports missing keys as skipped. After each prompt, it prints every provider result, timing, status, quality score, final score, the fastest successful provider, the best provider, and the fallback.

## Troubleshooting

### No API providers configured

No recognized key was loaded. Confirm that `.env` is in the same folder as `app.py`, at least one supported variable contains a value, and you started the application from the project folder.

### A provider is skipped

Add that provider's environment variable to `.env`. A skipped provider does not stop the others.

### HTTP 429 or 5xx errors

The provider may be rate-limited, overloaded, or temporarily unavailable. Providers with configured retries wait two seconds before retrying. Other providers still return their own results.

### Request timed out

The provider exceeded its configured timeout. Providers with retries try again; otherwise the request is marked as failed.

### Judge warning

If the Groq judge fails or returns invalid JSON, the router continues and assigns a neutral quality score of `50` to each usable answer.

### A response is truncated

The provider reached its token limit. The response remains eligible, but its completion score is reduced.

## Arabic evaluation

`run_arabic_eval.py` runs a 20-question Arabic test set (`eval_set_ar.jsonl`) through the router and scores every provider with rule-based checks instead of an LLM judge.

The set covers Egyptian, Gulf, Levantine and Maghrebi dialects, grammar, diacritics, Arabic and Latin numerals, dialect sentiment, sarcasm, e-commerce intent, Arabizi, JSON extraction, instruction following, and a hallucination test about a book that does not exist.

Answers are normalized before checking (hamza forms, alef maqsura, ta marbuta, diacritics, tatweel, Arabic-Indic digits). An Arabic question answered mostly in English counts as a failure.

```powershell
python run_arabic_eval.py --runs 3 --delay 4
python run_arabic_eval.py --runs 3 --providers Groq,Cohere,OpenRouter
python run_arabic_eval.py --runs 1 --limit 1 --no-judge --delay 0   # quick check
```

Results go to `eval_results/`: a CSV with every answer, its failed checks and any provider error, and a JSON summary with pass rates per provider and per category.

Limitations:

- Groq is both a contestant and the quality judge, so `avg_judge_quality` is biased for Groq. The main metric is the deterministic pass rate.
- Keyword checks can be fooled by long open-ended answers. Hallucination results were also verified by hand.
- Free tiers change. Gemini's free tier allowed 20 requests per day for the configured model, which is not enough for a 3-run evaluation.
- `leaked_reasoning_%` flags answers where the model's reasoning text appears in the output. `pass_rate_excl_leaks_%` shows the pass rate on the remaining answers.

Published results from November 2026 are in `eval_results/` (`*_155458` and `*_162111` are the two full runs).

## Benchmark v1 (100 questions, reference-checked)

`evals/` measures the router on a fixed 100-question set: factual (18), hallucination traps (2), reasoning (20), coding (20), instruction following (10) and Arabic (30, including the 20 questions above). Every question has a deterministic check, so the main score does not depend on an LLM judge:

- reasoning and numeric facts: the final number must match
- coding: the extracted function runs against hidden unit tests in a separate Python process with a 10 s timeout
- instruction following: exact format rules (line counts, JSON keys, casing, word limits)
- Arabic: the normalized keyword and format checks from `eval_set_ar.jsonl`

Each question also stores a `reference` answer that must pass its check and a `wrong` answer that must fail it. `pytest` checks both for all 100 questions.

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest                                                     # offline, no API calls
python -m evals.run --limit 3 --run-id smoke                         # quick check with real APIs
python -m evals.run --run-id v1                                      # full run; rerun the same command to resume
python -m evals.analyze eval_results/runs/v1                         # re-score saved outputs, no API calls, coding questions skipped
python -m evals.analyze eval_results/runs/v1 --exec                  # also run the 20 coding checks (see the notes below)
```

Each run folder in `eval_results/runs/<run-id>/` contains:

| File | Contents |
|---|---|
| `config.json` | question-set hash, models, timeouts, judge model, seed, git commit |
| `raw.jsonl` | every answer, latency, status, error and judge output, appended as the run goes |
| `rate_limited.jsonl` | questions set aside because of a 429/quota error; resuming asks them again |
| `answers.csv` | every answer with pass/fail and the checks it failed |
| `failures_for_review.csv` | failed answers to spot-check by hand |
| `summary.json`, `results.md` | all metrics |

Metrics per provider: win rate (how often the router picked it), reference accuracy with a 95% bootstrap CI, average weighted score, average judge score, median and p95 latency, and error, timeout, rate-limit, truncation and leaked-reasoning rates. Reference accuracy is strict (the answer must be what the prompt asked for). *Lenient accuracy* also accepts short-answer questions whose final answer is right but is preceded by leaked reasoning or glued to a duplicate (`72.72`), and *right but dirty* is the gap; every rescued answer is flagged in `answers.csv`. Router metrics: accuracy of the router's pick compared with the best single provider and with an oracle that picks any correct answer.

The judge audit reports how often the judge scores the correct answer above the wrong one in the same question, the judge failure rate, provider bias and position bias on questions where every answer was correct, and length bias.

Notes:

- The judge shuffles answer order with a seed (`--seed`). Set `JUDGE_MODEL` to use a Groq model that is not a contestant.
- Coding checks execute model-written code, so they are off by default. Pass `--exec` to `evals.run` or `evals.analyze` to run them; without it the analysis drops the coding questions (`--no-exec` is still accepted and means the same as the default). The published v1 numbers were produced with code execution on.
- With `--exec`, each answer runs in a child Python process with a minimal environment (only `PATH`, plus `SYSTEMROOT` on Windows; no API keys), with `HOME` and `TEMP` pointing at a fresh temporary directory that is also its working directory, and a 10 second timeout. This lowers the risk but it is not a sandbox: the code still runs as your user, can read and write your files and can use the network. Only use `--exec` on answers you generated yourself. For real isolation run the analysis inside a container or VM with no network and no access to your home folder.
- Free tiers: a full run makes about 100 calls per provider and about 200 to Groq (contestant + judge). Use `--providers` to leave out a provider whose daily quota is too small.

## Security

- `.env` and `.env.*` are ignored by Git; `.env.example` is the only exception.
- The repository should contain variable names and placeholders only, never real API keys.
- If a key is exposed, revoke it in the provider dashboard before doing anything else. Removing it from the latest commit is not enough because Git history may retain it.
- Treat a run folder you did not create (from a pull request or a download) as untrusted. `evals.run` and `evals.analyze` refuse run folders whose files are symbolic links or point outside the folder, write their outputs to a new file and then move it into place, and only accept a `question_set` from `config.json` that is a file inside `evals/` (UNC and other absolute paths are refused; pass `--set` to choose a different file yourself).
- Model output and run-folder text are printed with terminal escape sequences and control characters removed, so an answer cannot change your clipboard, window title or earlier lines on screen.
- Saved provider error bodies have account identifiers (`user_id`, `org_id` and similar) replaced with `[redacted]` before they are written to `raw.jsonl`, `rate_limited.jsonl` or the CSV files.
- The judge prompt wraps each answer between markers that carry a random boundary, removes that boundary from the answer text, and tells the judge that the content is data, not instructions. This reduces prompt injection from a contestant's answer but does not eliminate it: a model can still try to persuade the judge, so treat judge scores as a secondary signal next to the deterministic checks.

## Project structure

```text
Free-LLM-APIs/
|-- app.py            # Router, provider clients, judge, scoring, and CLI
|-- run_arabic_eval.py # Arabic evaluation runner
|-- eval_set_ar.jsonl # 20 Arabic test questions with rule-based checks
|-- evals/            # Benchmark v1: questions_v1.jsonl, checks, runner, analyzer
|-- tests/            # Offline tests (fake providers, no API calls)
|-- eval_results/     # Published evaluation runs
|-- requirements.txt  # Pinned Python dependencies
|-- .env.example      # Safe configuration template
|-- .gitignore        # Secrets, virtual environments, caches, and editor files
`-- README.md          # Setup, usage, architecture, and troubleshooting
```

## Possible next steps

- Add structured run logging and historical benchmarks
- Track provider cost and token usage
- Expose the router through FastAPI or a small web interface
