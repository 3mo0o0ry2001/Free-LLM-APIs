# Free LLM APIs Router: evaluation `v1`

- Question set: `evals\questions_v1.jsonl` (sha256 `78e06aa51cd5`), 100 questions x 1 run(s), 300 answers scored
- Started 2026-11-14T20:52:49, git `9aafdc7fecec66cacb0f1cab49d897e8aaa4900a-dirty`, judge: `openai/gpt-oss-20b`, seed 1234
- Primary metric: **reference accuracy** = answer passes the deterministic check for that question (exact value, executed unit tests, format rules). Failed calls count as wrong.
- 95% CIs are bootstrap intervals over questions.
- Completeness: 100/100 question-runs scored; 3 attempt(s) set aside for rate limits (`rate_limited.jsonl`)

## Router

| Metric | Value |
|---|---|
| Router pick accuracy % | 94.0 (CI 89.0–98.0) |
| Best single provider | OpenRouter at 96.0% |
| Router minus best single (pts) | -2.0 (CI -7.0–3.0) |
| Oracle: any provider correct % | 99.0 |
| Router picked a correct answer when one existed % | 94.9 |
| Questions where best score was a tie % | 1.0 |
| Questions with no usable answer % | 0.0 |

## Providers

| Provider | Model | Calls | Win rate % | Ref. accuracy % (CI) | Avg weighted score | Lenient acc. % | Right but dirty % | Avg judge score | Latency p50 s | Latency p95 s | Error % | Timeout % | Rate-limit % | Truncated % | Leaked reasoning % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenRouter | `nvidia/nemotron-3-ultra-550b-a55b:free` | 100 | 5.0 | 96.0 (92.0–99.0) | 88.5 | 96.0 | 0.0 | 93.9 | 2.42 | 6.36 | 3.0 | 3.0 | 0.0 | 1.0 | 0.0 |
| Groq | `openai/gpt-oss-20b` | 100 | 61.0 | 90.0 (84.0–95.0) | 91.2 | 90.0 | 0.0 | 89.8 | 0.53 | 1.04 | 3.0 | 0.0 | 0.0 | 3.0 | 0.0 |
| Cohere | `command-a-plus-05-2026` | 100 | 34.0 | 68.0 (59.0–77.0) | 86.8 | 81.0 | 13.0 | 72.8 | 0.89 | 2.17 | 0.0 | 0.0 | 0.0 | 0.0 | 24.0 |

Latency is measured on usable answers only and includes the provider's own retries.

**Strict vs lenient.** Reference accuracy is strict: the answer must be exactly what the prompt asked for. Lenient accuracy also accepts short-answer questions where the model's final answer is correct but is preceded by leaked reasoning or glued to a duplicate (`72.72`). *Right but dirty* is the gap. Every rescued answer is marked `rescued_by_lenient` in `answers.csv` for manual review.

## Reference accuracy by category (%)

| Category | n | OpenRouter | Groq | Cohere | Router pick |
|---|---|---|---|---|---|
| arabic | 30 | 93.3 | 80.0 | 46.7 | 80.0 |
| coding | 20 | 90.0 | 85.0 | 100.0 | 100.0 |
| factual | 18 | 100.0 | 100.0 | 83.3 | 100.0 |
| hallucination | 2 | 100.0 | 50.0 | 100.0 | 100.0 |
| instruction | 10 | 100.0 | 100.0 | 70.0 | 100.0 |
| reasoning | 20 | 100.0 | 100.0 | 50.0 | 100.0 |

## Judge audit

| Check | Value |
|---|---|
| Judge call failure rate % | 2.0 |
| Pass-vs-fail pairs compared | 69 |
| Judge scores the correct answer higher % | 85.5 |
| Ties % | 1.4 |
| Judge scores the wrong answer higher % | 13.0 |
| Pairwise agreement, ties = 0.5 (CI) | 86.2 (73.9–96.4) |
| Both-correct pairs: longer answer scored higher % | 31.0 of 42 pairs |
| Spearman(length, residual), correct answers | -0.274 |

Provider bias: mean judge score minus the question mean, only on questions where every judged answer passed the reference check (0 = no preference).

| Provider | n | Mean residual | CI |
|---|---|---|---|
| Cohere | 62 | -5.4 | -10.2–-1.6 |
| Groq | 59 | 2.6 | 0.5–4.9 |
| OpenRouter | 61 | 3.0 | 0.9–5.5 |

Position bias (answer order is shuffled per question with a fixed seed; all-pass questions only):

| Position shown | n | Mean residual |
|---|---|---|
| 1 | 62 | -0.5 |
| 2 | 62 | -1.6 |
| 3 | 58 | 2.2 |

## What these numbers do not show

- Free-tier models, quotas and latencies change; this is a snapshot of the dates above.
- Deterministic checks verify one property (a value, passing tests, a format). They do not grade explanation quality, and keyword checks can misfire: see `failures_for_review.csv`.
- The judge model is also a contestant when it is the Groq model, so its scores may favor it.
- Win rate depends on the router's weights (40/25/20/15): a fast correct answer can beat a slower correct one, and latency is scored relative to each provider's own timeout.
- Ties in the final score go to the provider listed first in `PROVIDER_CONFIG`.
