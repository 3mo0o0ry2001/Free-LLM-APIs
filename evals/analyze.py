"""
Score a saved run and write the results.

    python -m evals.analyze eval_results/runs/<run-id>
    python -m evals.analyze eval_results/runs/<run-id> --no-exec

Reads raw.jsonl + config.json, applies the deterministic checks from the
question set, and writes into the run folder:

    answers.csv              every answer with pass/fail and failed checks
    failures_for_review.csv  failed answers, to spot-check the checks by hand
    summary.json             every number in results.md, machine-readable
    results.md               tables + judge audit

Nothing here calls an API. Confidence intervals are 95% bootstrap
intervals that resample questions (all repetitions of a question together).
"""

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from evals.checks import LEAK_PATTERN, run_checks, run_checks_lenient

BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 7
USABLE = {"success", "truncated"}


# ============================================================
# SMALL STATS HELPERS
# ============================================================

def pct(num, den):
    return round(100.0 * num / den, 1) if den else None


def mean(xs):
    xs = list(xs)
    return statistics.fmean(xs) if xs else None


def percentile(xs, q):
    """Linear-interpolated percentile, q in [0, 100]."""
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def bootstrap_ci(clusters: dict, stat, iters=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED):
    """
    clusters: {question_id: [items]}; stat: f(list_of_items) -> float | None.
    Returns (low, high) of the 2.5 / 97.5 percentiles, or (None, None).
    """
    keys = list(clusters)
    if len(keys) < 2:
        return (None, None)
    rng = random.Random(seed)
    values = []
    for _ in range(iters):
        sample = []
        for k in rng.choices(keys, k=len(keys)):
            sample.extend(clusters[k])
        v = stat(sample)
        if v is not None:
            values.append(v)
    if not values:
        return (None, None)
    return (round(percentile(values, 2.5), 1), round(percentile(values, 97.5), 1))


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


# ============================================================
# LOAD + SCORE
# ============================================================

def error_kind(status, error):
    if status in USABLE:
        return None
    e = (error or "").lower()
    if "timed out" in e:
        return "timeout"
    if "429" in e or "rate limit" in e or "quota" in e or "resource_exhausted" in e:
        return "rate_limit"
    if "empty response" in e or "no choices" in e or "no message" in e:
        return "empty"
    return "other_error"


def load(run_dir: Path, set_path: Path | None, allow_exec: bool):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    set_path = set_path or Path(config["question_set"])
    questions = {}
    with set_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                questions[q["id"]] = q
    records = []
    with (run_dir / "raw.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    answers, qrows = [], []
    exec_cache = {}
    for rec in records:
        q = questions[rec["id"]]
        if not allow_exec and "python_tests" in q["checks"]:
            continue  # coding questions cannot be scored without running the code
        rows = []
        for r in rec["results"]:
            text = r.get("answer") or ""
            usable = r["status"] in USABLE and bool(text)
            if usable:
                key = (rec["id"], text)
                if key not in exec_cache:
                    exec_cache[key] = (
                        run_checks(text, q["checks"], q["lang"], allow_exec=allow_exec),
                        run_checks_lenient(text, q["checks"], q["lang"], allow_exec=allow_exec),
                    )
                failed, lenient_failed = exec_cache[key]
            else:
                failed = lenient_failed = ["no_answer"]
            row = {
                "run": rec["run"], "id": rec["id"], "category": rec["category"],
                "provider": r["provider"], "model": r["model"], "status": r["status"],
                "latency_s": r.get("latency"), "attempts": r.get("attempts"),
                "error_kind": error_kind(r["status"], r.get("error")),
                "error": r.get("error"),
                "passed": not failed, "failed_checks": ";".join(failed),
                "passed_lenient": not lenient_failed,
                "rescued_by_lenient": bool(failed) and not lenient_failed,
                "leaked_reasoning": bool(LEAK_PATTERN.search(text)),
                "length_chars": len(text),
                "judge_score": r.get("quality_score") if r.get("judge_status") == "ok" else None,
                "judge_status": r.get("judge_status"),
                "judge_position": r.get("judge_position"),
                "final_score": r.get("final_score") or 0.0,
                "router_best": rec["router"]["best"] == r["provider"],
                "answer": text,
            }
            rows.append(row)
        answers.extend(rows)
        best = next((x for x in rows if x["router_best"]), None)
        qrows.append({
            "run": rec["run"], "id": rec["id"], "category": rec["category"],
            "best": rec["router"]["best"], "best_was_tie": rec["router"].get("best_was_tie", False),
            "best_passed": bool(best and best["passed"]),
            "any_passed": any(x["passed"] for x in rows),
            "any_usable": any(x["status"] in USABLE for x in rows),
            "judge_status": rec["judge"].get("status"),
            "rows": rows,
        })
    return config, questions, answers, qrows


# ============================================================
# METRICS
# ============================================================

def by_question(items, key="id"):
    out = defaultdict(list)
    for it in items:
        out[it[key]].append(it)
    return out


def provider_table(answers, qrows):
    providers = sorted({a["provider"] for a in answers})
    n_q = len(qrows)
    table = []
    for p in providers:
        rs = [a for a in answers if a["provider"] == p]
        ok = [a for a in rs if a["status"] in USABLE]
        lat = [a["latency_s"] for a in ok if a["latency_s"] is not None]
        kinds = Counter(a["error_kind"] for a in rs if a["error_kind"])
        judged = [a["judge_score"] for a in rs if a["judge_score"] is not None]
        acc_ci = bootstrap_ci(by_question(rs), lambda s: pct(sum(x["passed"] for x in s), len(s)))
        table.append({
            "provider": p,
            "model": rs[0]["model"],
            "calls": len(rs),
            "win_rate_%": pct(sum(q["best"] == p for q in qrows), n_q),
            "reference_accuracy_%": pct(sum(a["passed"] for a in rs), len(rs)),
            "reference_accuracy_ci": acc_ci,
            "lenient_accuracy_%": pct(sum(a["passed_lenient"] for a in rs), len(rs)),
            "right_but_dirty_%": pct(sum(a["rescued_by_lenient"] for a in rs), len(rs)),
            "accuracy_when_answered_%": pct(sum(a["passed"] for a in ok), len(ok)),
            "avg_weighted_score": round(mean(a["final_score"] for a in rs), 1),
            "avg_judge_score": round(mean(judged), 1) if judged else None,
            "latency_median_s": round(percentile(lat, 50), 2) if lat else None,
            "latency_p95_s": round(percentile(lat, 95), 2) if lat else None,
            "error_rate_%": pct(sum(a["status"] == "failed" for a in rs), len(rs)),
            "timeout_rate_%": pct(kinds["timeout"], len(rs)),
            "rate_limit_rate_%": pct(kinds["rate_limit"], len(rs)),
            "other_error_rate_%": pct(kinds["other_error"] + kinds["empty"], len(rs)),
            "truncated_rate_%": pct(sum(a["status"] == "truncated" for a in rs), len(rs)),
            "leaked_reasoning_%": pct(sum(a["leaked_reasoning"] for a in rs), len(rs)),
            "first_error": next((a["error"] for a in rs if a["error"]), None),
        })
    table.sort(key=lambda r: r["reference_accuracy_%"] or 0, reverse=True)
    return table


def router_metrics(qrows, ptable):
    clusters = by_question(qrows)
    providers = [r["provider"] for r in ptable]

    def acc_of(p):
        return lambda s: pct(sum(any(x["provider"] == p and x["passed"] for x in q["rows"]) for q in s), len(s))

    router_acc = lambda s: pct(sum(q["best_passed"] for q in s), len(s))
    best_single = max(providers, key=lambda p: acc_of(p)(qrows) or 0) if providers else None

    def diff(s):
        a, b = router_acc(s), acc_of(best_single)(s)
        return None if a is None or b is None else a - b

    with_correct = [q for q in qrows if q["any_passed"]]
    return {
        "questions_scored": len(qrows),
        "router_accuracy_%": router_acc(qrows),
        "router_accuracy_ci": bootstrap_ci(clusters, router_acc),
        "best_single_provider": best_single,
        "best_single_accuracy_%": acc_of(best_single)(qrows) if best_single else None,
        "router_minus_best_single_pts": round(diff(qrows), 1) if best_single else None,
        "router_minus_best_single_ci": bootstrap_ci(clusters, diff) if best_single else (None, None),
        "oracle_accuracy_%": pct(len(with_correct), len(qrows)),
        "router_picked_correct_when_possible_%": pct(sum(q["best_passed"] for q in with_correct), len(with_correct)),
        "best_ties_%": pct(sum(q["best_was_tie"] for q in qrows), len(qrows)),
        "no_usable_answer_%": pct(sum(not q["any_usable"] for q in qrows), len(qrows)),
    }


def category_table(answers, qrows):
    cats = sorted({a["category"] for a in answers})
    providers = sorted({a["provider"] for a in answers})
    out = {}
    for c in cats:
        row = {}
        for p in providers:
            rs = [a for a in answers if a["category"] == c and a["provider"] == p]
            row[p] = pct(sum(a["passed"] for a in rs), len(rs))
        qs = [q for q in qrows if q["category"] == c]
        row["Router pick"] = pct(sum(q["best_passed"] for q in qs), len(qs))
        row["n_questions"] = len({q["id"] for q in qs})
        out[c] = row
    return out


def judge_audit(qrows):
    """How well does the LLM judge agree with the reference checks, and is it biased?"""
    with_usable = [q for q in qrows if q["any_usable"]]
    failed_calls = sum(q["judge_status"] == "failed" for q in with_usable)
    judged_q = []
    for q in with_usable:
        js = [r for r in q["rows"] if r["judge_score"] is not None]
        if len(js) >= 2:
            judged_q.append((q, js))
    if not judged_q:
        return {"judge_calls": len(with_usable), "judge_failure_rate_%": pct(failed_calls, len(with_usable)),
                "note": "Not enough judged questions for an audit."}

    # 1. Agreement with the reference: pairs (pass, fail) inside one question.
    higher = tie = lower = 0
    pair_items = defaultdict(list)
    for q, js in judged_q:
        for a in js:
            for b in js:
                if a["passed"] and not b["passed"]:
                    d = a["judge_score"] - b["judge_score"]
                    v = 1.0 if d > 0 else 0.5 if d == 0 else 0.0
                    higher += d > 0
                    tie += d == 0
                    lower += d < 0
                    pair_items[q["id"]].append(v)
    n_pairs = higher + tie + lower
    flat = lambda s: pct(sum(s), len(s)) if s else None

    # 2. Residual = judge score minus the question mean.
    resid = []
    for q, js in judged_q:
        m = mean(r["judge_score"] for r in js)
        for r in js:
            resid.append({**r, "resid": r["judge_score"] - m, "all_pass": all(x["passed"] for x in js)})

    providers = sorted({r["provider"] for r in resid})
    provider_bias = {}
    all_pass = [r for r in resid if r["all_pass"]]
    for p in providers:
        rs = [r for r in all_pass if r["provider"] == p]
        if rs:
            ci = bootstrap_ci(by_question(rs), lambda s: round(mean(x["resid"] for x in s), 1))
            provider_bias[p] = {"n": len(rs), "mean_residual": round(mean(r["resid"] for r in rs), 1), "ci": ci}

    # Top pick when every judged answer passed the reference check.
    top = Counter()
    n_allpass_q = 0
    for q, js in judged_q:
        if all(x["passed"] for x in js):
            n_allpass_q += 1
            best = max(x["judge_score"] for x in js)
            winners = [x["provider"] for x in js if x["judge_score"] == best]
            for w in winners:
                top[w] += 1 / len(winners)

    # Position: only all-pass questions, so correctness cannot leak in.
    position_bias = {}
    for pos in sorted({r["judge_position"] for r in all_pass if r["judge_position"] is not None}):
        rs = [r for r in all_pass if r["judge_position"] == pos]
        position_bias[str(pos)] = {"n": len(rs), "mean_residual": round(mean(r["resid"] for r in rs), 1)}

    # 3. Length bias, only among answers that all passed (correctness held equal).
    longer_wins = shorter_wins = 0
    for q, js in judged_q:
        ok = [x for x in js if x["passed"]]
        for i in range(len(ok)):
            for j in range(i + 1, len(ok)):
                a, b = ok[i], ok[j]
                if a["length_chars"] == b["length_chars"] or a["judge_score"] == b["judge_score"]:
                    continue
                long_, short = (a, b) if a["length_chars"] > b["length_chars"] else (b, a)
                if long_["judge_score"] > short["judge_score"]:
                    longer_wins += 1
                else:
                    shorter_wins += 1
    passed_resid = [r for r in resid if r["passed"]]

    return {
        "judge_calls": len(with_usable),
        "judge_failure_rate_%": pct(failed_calls, len(with_usable)),
        "questions_audited": len(judged_q),
        "pairs_pass_vs_fail": n_pairs,
        "judge_ranks_correct_higher_%": pct(higher, n_pairs),
        "judge_ties_%": pct(tie, n_pairs),
        "judge_ranks_wrong_higher_%": pct(lower, n_pairs),
        "pairwise_agreement_ties_half_%": flat([v for vs in pair_items.values() for v in vs]),
        "pairwise_agreement_ci": bootstrap_ci(pair_items, flat),
        "provider_bias_all_pass_questions": provider_bias,
        "top_pick_share_all_pass_questions": {
            "n_questions": n_allpass_q,
            "share_%": {p: pct(v, n_allpass_q) for p, v in sorted(top.items())},
        },
        "position_bias": position_bias,
        "length_pairs_both_correct": longer_wins + shorter_wins,
        "longer_answer_scored_higher_%": pct(longer_wins, longer_wins + shorter_wins),
        "spearman_length_vs_residual_correct_answers": spearman(
            [r["length_chars"] for r in passed_resid], [r["resid"] for r in passed_resid]),
    }


# ============================================================
# OUTPUT
# ============================================================

def fmt(v):
    if v is None:
        return "–"
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return "–" if v[0] is None else f"{v[0]}–{v[1]}"
    return str(v)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_markdown(path, config, ptable, router, cats, audit, n_answers):
    runs = config["runs"]
    expected = len(config["question_ids"]) * runs
    deferred_file = path.with_name("rate_limited.jsonl")
    deferred = sum(1 for l in deferred_file.open(encoding="utf-8") if l.strip()) if deferred_file.exists() else 0
    complete = router["questions_scored"]
    status_line = (f"- Completeness: {complete}/{expected} question-runs scored"
                   + ("" if complete == expected else " (**incomplete run: numbers are partial**)")
                   + f"; {deferred} attempt(s) set aside for rate limits (`rate_limited.jsonl`)")
    lines = [
        f"# Free LLM APIs Router: evaluation `{config['run_id']}`",
        "",
        f"- Question set: `{config['question_set']}` (sha256 `{config['question_set_sha256'][:12]}`), "
        f"{len(config['question_ids'])} questions x {runs} run(s), {n_answers} answers scored",
        f"- Started {config.get('created_at')}, git `{config.get('git_commit')}`, "
        f"judge: `{config.get('judge_model')}`, seed {config.get('seed')}",
        "- Primary metric: **reference accuracy** = answer passes the deterministic check for that question "
        "(exact value, executed unit tests, format rules). Failed calls count as wrong.",
        "- 95% CIs are bootstrap intervals over questions.",
        status_line,
        "",
        "## Router",
        "",
        md_table(["Metric", "Value"], [
            ["Router pick accuracy %", f"{fmt(router['router_accuracy_%'])} (CI {fmt(router['router_accuracy_ci'])})"],
            ["Best single provider", f"{router['best_single_provider']} at {fmt(router['best_single_accuracy_%'])}%"],
            ["Router minus best single (pts)",
             f"{fmt(router['router_minus_best_single_pts'])} (CI {fmt(router['router_minus_best_single_ci'])})"],
            ["Oracle: any provider correct %", fmt(router["oracle_accuracy_%"])],
            ["Router picked a correct answer when one existed %", fmt(router["router_picked_correct_when_possible_%"])],
            ["Questions where best score was a tie %", fmt(router["best_ties_%"])],
            ["Questions with no usable answer %", fmt(router["no_usable_answer_%"])],
        ]),
        "",
        "## Providers",
        "",
        md_table(
            ["Provider", "Model", "Calls", "Win rate %", "Ref. accuracy % (CI)", "Avg weighted score",
             "Lenient acc. %", "Right but dirty %", "Avg judge score", "Latency p50 s", "Latency p95 s", "Error %", "Timeout %", "Rate-limit %",
             "Truncated %", "Leaked reasoning %"],
            [[r["provider"], f"`{r['model']}`", r["calls"], r["win_rate_%"],
              f"{fmt(r['reference_accuracy_%'])} ({fmt(r['reference_accuracy_ci'])})",
              r["avg_weighted_score"], r["lenient_accuracy_%"], r["right_but_dirty_%"],
              r["avg_judge_score"], r["latency_median_s"], r["latency_p95_s"],
              r["error_rate_%"], r["timeout_rate_%"], r["rate_limit_rate_%"], r["truncated_rate_%"],
              r["leaked_reasoning_%"]] for r in ptable]),
        "",
        "Latency is measured on usable answers only and includes the provider's own retries.",
        "",
        "**Strict vs lenient.** Reference accuracy is strict: the answer must be exactly what the prompt asked "
        "for. Lenient accuracy also accepts short-answer questions where the model's final answer is correct "
        "but is preceded by leaked reasoning or glued to a duplicate (`72.72`). *Right but dirty* is the gap. "
        "Every rescued answer is marked `rescued_by_lenient` in `answers.csv` for manual review.",
        "",
        "## Reference accuracy by category (%)",
        "",
    ]
    cols = [r["provider"] for r in ptable] + ["Router pick"]
    lines.append(md_table(["Category", "n"] + cols,
                          [[c, row["n_questions"]] + [row.get(p) for p in cols] for c, row in cats.items()]))
    lines += ["", "## Judge audit", ""]
    if "note" in audit:
        lines.append(audit["note"])
    else:
        lines.append(md_table(["Check", "Value"], [
            ["Judge call failure rate %", fmt(audit["judge_failure_rate_%"])],
            ["Pass-vs-fail pairs compared", audit["pairs_pass_vs_fail"]],
            ["Judge scores the correct answer higher %", fmt(audit["judge_ranks_correct_higher_%"])],
            ["Ties %", fmt(audit["judge_ties_%"])],
            ["Judge scores the wrong answer higher %", fmt(audit["judge_ranks_wrong_higher_%"])],
            ["Pairwise agreement, ties = 0.5 (CI)",
             f"{fmt(audit['pairwise_agreement_ties_half_%'])} ({fmt(audit['pairwise_agreement_ci'])})"],
            ["Both-correct pairs: longer answer scored higher %",
             f"{fmt(audit['longer_answer_scored_higher_%'])} of {audit['length_pairs_both_correct']} pairs"],
            ["Spearman(length, residual), correct answers", fmt(audit["spearman_length_vs_residual_correct_answers"])],
        ]))
        lines += ["", "Provider bias: mean judge score minus the question mean, only on questions where every "
                      "judged answer passed the reference check (0 = no preference).", ""]
        lines.append(md_table(["Provider", "n", "Mean residual", "CI"],
                              [[p, v["n"], v["mean_residual"], v["ci"]]
                               for p, v in audit["provider_bias_all_pass_questions"].items()]))
        lines += ["", "Position bias (answer order is shuffled per question with a fixed seed; all-pass questions only):", ""]
        lines.append(md_table(["Position shown", "n", "Mean residual"],
                              [[int(k) + 1, v["n"], v["mean_residual"]] for k, v in audit["position_bias"].items()]))
    lines += [
        "",
        "## What these numbers do not show",
        "",
        "- Free-tier models, quotas and latencies change; this is a snapshot of the dates above.",
        "- Deterministic checks verify one property (a value, passing tests, a format). They do not grade "
        "explanation quality, and keyword checks can misfire: see `failures_for_review.csv`.",
        "- The judge model is also a contestant when it is the Groq model, so its scores may favor it.",
        "- Win rate depends on the router's weights (40/25/20/15): a fast correct answer can beat a slower "
        "correct one, and latency is scored relative to each provider's own timeout.",
        "- Ties in the final score go to the provider listed first in `PROVIDER_CONFIG`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csvs(run_dir, answers):
    cols = ["run", "id", "category", "provider", "model", "status", "latency_s", "attempts", "error_kind",
            "passed", "passed_lenient", "rescued_by_lenient", "failed_checks", "leaked_reasoning", "length_chars", "judge_status", "judge_score",
            "judge_position", "final_score", "router_best", "error", "answer"]
    with (run_dir / "answers.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(answers)
    with (run_dir / "failures_for_review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["check_was_wrong (fill in y/n)"], extrasaction="ignore")
        w.writeheader()
        w.writerows(a for a in answers if not a["passed"] and a["status"] in USABLE)


def analyze(run_dir: Path, set_path: Path | None = None, allow_exec: bool = True, quiet: bool = False):
    config, _questions, answers, qrows = load(run_dir, set_path, allow_exec)
    if not answers:
        sys.exit("raw.jsonl is empty.")
    ptable = provider_table(answers, qrows)
    router = router_metrics(qrows, ptable)
    cats = category_table(answers, qrows)
    audit = judge_audit(qrows)
    summary = {"run_id": config["run_id"], "exec_enabled": allow_exec, "router": router,
               "providers": ptable, "accuracy_by_category_%": cats, "judge_audit": audit}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csvs(run_dir, answers)
    write_markdown(run_dir / "results.md", config, ptable, router, cats, audit, len(answers))
    if not quiet:
        print((run_dir / "results.md").read_text(encoding="utf-8"))
        print(f"Wrote {run_dir / 'results.md'}, summary.json, answers.csv, failures_for_review.csv")
    return summary


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--set", default=None, help="override the question set path saved in config.json")
    ap.add_argument("--no-exec", action="store_true", help="do not execute model-written code")
    args = ap.parse_args()
    analyze(Path(args.run_dir), Path(args.set) if args.set else None, allow_exec=not args.no_exec)


if __name__ == "__main__":
    main()
