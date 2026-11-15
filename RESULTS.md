# Benchmark v1: results

Measured 14–15 Nov 2026 with `python -m evals.run --run-id v1`. Raw outputs, config and the full generated report are in `eval_results/runs/v1/` (`raw.jsonl`, `config.json`, `answers.csv`, `results.md`, `summary.json`).

## Setup

- **Providers tested (3):** Groq `openai/gpt-oss-20b` (also the quality judge), OpenRouter `nvidia/nemotron-3-ultra-550b-a55b:free`, Cohere `command-a-plus-05-2026`. Mistral had no API key. Gemini failed all 3 calls of the smoke test and its free daily quota (20 requests, per the README) cannot cover 100 questions, so it was left out. This is a 3-provider result, not 5.
- **Questions (100, fixed, `evals/questions_v1.jsonl`):** Arabic 30, coding 20, reasoning 20, factual 18, instruction following 10, hallucination traps 2. One run, no repetitions.
- **Scoring:** every question has a deterministic check, so the main number does not depend on an LLM judge: the final number must match, code runs against hidden unit tests, format rules are checked exactly. Each check was validated on a known-correct and a known-wrong answer.
  - **Strict accuracy:** the answer is exactly what the prompt asked for.
  - **Lenient accuracy:** also accepts short-answer questions where the final answer is right but preceded by leaked reasoning or glued to a duplicate (`72.72`). The gap is "right but dirty".
- **Intervals:** 95% bootstrap over questions. Judge answer order is shuffled with a fixed seed (1234).
- **Run history:** OpenRouter's free tier allows 50 requests a day, so the run was done in two sessions. Three questions that hit the limit were set aside (`rate_limited.jsonl`) and asked again; no quota error is counted as a model failure.

## Results

| Provider | Strict accuracy % (95% CI) | Lenient % | Router win rate % | Latency p50 / p95 (s) | Failed calls | Truncated | Avg judge score |
|---|---|---|---|---|---|---|---|
| OpenRouter | 96.0 (92.0–99.0) | 96.0 | 5.0 | 2.42 / 6.36 | 3 (all timeouts) | 1 | 93.9 |
| Groq | 90.0 (84.0–95.0) | 90.0 | 61.0 | 0.53 / 1.04 | 3 (HTTP 400) | 3 | 89.8 |
| Cohere | 68.0 (59.0–77.0) | 81.0 | 34.0 | 0.89 / 2.17 | 0 | 0 | 72.8 |
| **Router pick** | **94.0 (89.0–98.0)** | | | | 0 questions without an answer | | |

At least one provider was right on 99 of 100 questions (oracle).

Strict accuracy by category (%):

| Category | n | OpenRouter | Groq | Cohere | Router pick |
|---|---|---|---|---|---|
| Arabic | 30 | 93.3 | 80.0 | 46.7 | 80.0 |
| Coding | 20 | 90.0 | 85.0 | 100.0 | 100.0 |
| Factual | 18 | 100.0 | 100.0 | 83.3 | 100.0 |
| Hallucination | 2 | 100.0 | 50.0 | 100.0 | 100.0 |
| Instruction | 10 | 100.0 | 100.0 | 70.0 | 100.0 |
| Reasoning | 20 | 100.0 | 100.0 | 50.0 | 100.0 |

## What the numbers show

1. **The router did not beat the best single provider.** 94% against 96% for OpenRouter; the difference is −2.0 points with a 95% interval of −7.0 to +3.0, so this run shows no difference either way. Its measurable advantage is availability: it returned an answer for all 100 questions, while Groq and OpenRouter each failed 3 calls (Cohere failed none).
2. **All 6 router misses were Arabic questions, and in all 6 the router had picked Cohere.** The 66 picks of Groq or OpenRouter were all correct; Cohere's 34 picks were correct 28 times. The router's loss against OpenRouter comes from the judge scoring Cohere's wrong answers too high, not from the speed weight.
3. **The judge is the weak link, mostly in Arabic.** In 69 same-question comparisons where one answer passed its check and one failed, the judge scored the passing answer higher in 85.5% (59), tied once and scored the failing one higher in 13.0% (9). Split by language: **24 of 33 Arabic comparisons against 35 of 36 for everything else.** Example: for `نجحَ الطالبُ` the correct parse of `الطالبُ` is `فاعل`. Cohere and Groq answered `مبتدأ` and the judge scored them 95 and 90; the correct answer got 20. In three other misses, English reasoning leaked by Cohere scored 95 while being wrong.
4. **No clear length or self-preference bias.** Among 42 pairs where both answers were correct, the longer one scored higher in 31% (Spearman −0.27 between length and score). Groq is both judge and contestant, but its judge residual (+2.6, CI 0.5–4.9) is no larger than OpenRouter's (+3.0, CI 0.9–5.5). Cohere's residual is −5.4 (CI −10.2 to −1.6), which probably reflects its leaked output, and this audit cannot separate the two. Position effects (−0.5, −1.6, +2.2 for positions 1–3) were not tested for significance.
5. **Cohere returns reasoning mixed into the answer.** In 24 of 100 answers the output began with the model's own reasoning (often starting with a literal `<EOS_TOKEN>`), frequently followed by the real answer glued to the end (`72.72`, `Friday.Friday`). This is why strict accuracy is 68% and lenient 81%. It reproduced in 10 of 10 calls of a separate probe and appeared in a minority of the run's answers, so it varies over time.
6. **Groq's 3 coding failures were an API rejection, not wrong code.** The API returned HTTP 400 `tool_use_failed` (the model tried to call a tool named `code`). The code visible in the error text for all three (`code_05`, `code_08`, `code_10`) passes the hidden tests when run. The router does not retry this error.

## What this does not show

- **Small and single-run.** 100 questions, one run, so no variance across repetitions, and the intervals are wide. Three of the Arabic findings rest on 33 comparisons.
- **A snapshot.** Free-tier models, quotas and latencies change; Cohere's behavior already differed between the two sessions.
- **Not a ranking of the models.** Cohere's score mostly reflects how its API returns text, and the questions are short-answer tasks with checkable answers. They do not measure writing quality or long reasoning.
- **Check coverage.** Arabic keyword checks can reject valid phrasings. All failed Arabic answers from Groq and Cohere (content failures) were read by hand against the prompts and no check errors were found, but two labels are judgment calls (`ar_sent_02`, a sarcastic sentence labelled negative, and `ar_dial_lev_01`). One Cohere answer (`ar_math_01`) was right but not counted by the lenient score, so Cohere's lenient figure is at most 1 point low.
- **Router weights.** Win rate depends on the 40/25/20/15 weights; OpenRouter has the best accuracy and judge score but wins 5% of the time because it is slower.
- **Latency is per provider.** It was not measured for the router as a whole, which waits for all providers.

## Reproduce

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest                                   # offline, no API calls
python -m evals.run --run-id v1 --providers Groq,OpenRouter,Cohere
python -m evals.analyze eval_results\runs\v1       # re-score saved outputs, no API calls
```

`config.json` records the question-set hash, models, timeouts, judge model, seed and the base git commit. It is marked dirty because the benchmark code (`evals/` and the fixes in `app.py`) was added on top of that commit.
