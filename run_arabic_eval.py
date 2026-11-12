"""
Arabic evaluation runner for Free LLM APIs Router.

Place this file and eval_set_ar.jsonl in the repo root (next to app.py), then:

    python run_arabic_eval.py --runs 3 --delay 4

Primary metric = deterministic pass rate (rule-based checks, no LLM judge).
The Groq judge score is recorded as a secondary signal only, because Groq is
both a contestant and the judge (self-preference bias), and answers are shown
to the judge in a fixed order (position bias).
"""

import argparse
import asyncio
import csv
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import app  # Router module: PROVIDERS, ask_all_models, evaluate_quality, apply_final_scores

# ============================================================
# ARABIC NORMALIZATION
# ============================================================

DIACRITICS = re.compile(r"[ً-ْٰـ]")  # tashkeel + superscript alef + tatweel
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ARABIC_LETTER = re.compile(r"[ء-ي]")
LATIN_LETTER = re.compile(r"[A-Za-z]")


def normalize_ar(text: str) -> str:
    text = text.translate(ARABIC_INDIC)
    text = DIACRITICS.sub("", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", text).strip().lower()


def arabic_ratio(text: str) -> float:
    ar = len(ARABIC_LETTER.findall(text))
    lat = len(LATIN_LETTER.findall(text))
    return ar / (ar + lat) if (ar + lat) else 0.0


def extract_json_obj(text: str):
    text = re.sub(r"```(?:json)?", "", text)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ============================================================
# DETERMINISTIC CHECKS
# ============================================================

DEFAULT_ARABIC_RATIO = 0.6  # Arabic prompt -> answer should be mostly Arabic

# Reasoning text leaking into the final answer (seen with Cohere via the compat API)
LEAK_PATTERN = re.compile(
    r"^\s*(<EOS_TOKEN>|(the )?user (asks|wants)|يسأل المستخدم|المستخدم يطلب|نحتاج|we need|1\.\s+تحليل الطلب)",
    re.I,
)


def run_checks(answer: str, checks: dict) -> list[str]:
    """Return a list of failed check names (empty list = pass)."""
    failed = []
    norm = normalize_ar(answer)

    def has(term):
        return normalize_ar(term) in norm

    if "contains_any" in checks and not any(has(t) for t in checks["contains_any"]):
        failed.append("contains_any")
    if "contains_any_raw" in checks and not any(t in answer for t in checks["contains_any_raw"]):
        failed.append("contains_any_raw")
    for i, group in enumerate(checks.get("contains_all_groups", [])):
        if not any(has(t) for t in group):
            failed.append(f"contains_group_{i}")
    if any(has(t) for t in checks.get("not_contains", [])):
        failed.append("not_contains")
    if any(t in answer for t in checks.get("not_contains_raw", [])):
        failed.append("not_contains_raw")
    digits = answer.translate(ARABIC_INDIC)
    if "regex_any" in checks and not any(re.search(p, digits) for p in checks["regex_any"]):
        failed.append("regex_any")
    if any(re.search(p, digits) for p in checks.get("regex_none", [])):
        failed.append("regex_none")
    if "max_words" in checks and len(answer.split()) > checks["max_words"]:
        failed.append("max_words")
    if "numbered_items" in checks:
        items = re.findall(r"^\s*[0-9٠-٩]+[.)\-]", answer, re.M)
        if len(items) != checks["numbered_items"]:
            failed.append("numbered_items")
    if "json_fields" in checks:
        obj = extract_json_obj(answer)
        if not isinstance(obj, dict):
            failed.append("json_parse")
        else:
            for key, expected in checks["json_fields"].items():
                if normalize_ar(expected) not in normalize_ar(str(obj.get(key, ""))):
                    failed.append(f"json_{key}")
    min_ratio = checks.get("arabic_ratio_min", DEFAULT_ARABIC_RATIO)
    if min_ratio and arabic_ratio(answer) < min_ratio:
        failed.append("arabic_ratio")
    return failed


# ============================================================
# RUNNER
# ============================================================

def load_eval_set(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def run(eval_set, runs, delay, use_judge):
    rows = []
    total = len(eval_set) * runs
    n = 0
    for run_idx in range(1, runs + 1):
        for item in eval_set:
            n += 1
            print(f"[{n}/{total}] run={run_idx} {item['id']}", flush=True)
            results = await app.ask_all_models(item["prompt"])
            if use_judge:
                await app.evaluate_quality(item["prompt"], results)
            app.apply_final_scores(results)
            for r in results:
                answer = r.get("answer") or ""
                usable = r["status"] in {"success", "truncated"} and bool(answer)
                failed = run_checks(answer, item["checks"]) if usable else ["no_answer"]
                rows.append({
                    "run": run_idx,
                    "id": item["id"],
                    "category": item["category"],
                    "provider": r["provider"],
                    "model": r["model"],
                    "status": r["status"],
                    "latency_s": round(r.get("latency") or 0, 3),
                    "passed": int(not failed),
                    "failed_checks": ";".join(failed),
                    "leaked_reasoning": int(bool(LEAK_PATTERN.search(answer))),
                    "error": (r.get("error") or "")[:1000],
                    "arabic_ratio": round(arabic_ratio(answer), 3),
                    "judge_quality": r.get("quality_score"),
                    "router_final_score": round(r.get("final_score") or 0, 2),
                    "answer": answer.replace("\n", " ⏎ ")[:500],
                })
            if delay:
                await asyncio.sleep(delay)
    return rows


def summarize(rows):
    by_provider = defaultdict(list)
    for r in rows:
        by_provider[r["provider"]].append(r)
    summary = []
    for provider, rs in by_provider.items():
        ok = [r for r in rs if r["status"] in {"success", "truncated"}]
        lat = sorted(r["latency_s"] for r in ok)
        summary.append({
            "provider": provider,
            "model": rs[0]["model"],
            "n": len(rs),
            "pass_rate_%": round(100 * sum(r["passed"] for r in rs) / len(rs), 1),
            "failure_rate_%": round(100 * sum(r["status"] == "failed" for r in rs) / len(rs), 1),
            "truncated_%": round(100 * sum(r["status"] == "truncated" for r in rs) / len(rs), 1),
            "wrong_language_%": round(100 * sum("arabic_ratio" in r["failed_checks"] for r in rs) / len(rs), 1),
            "leaked_reasoning_%": round(100 * sum(r["leaked_reasoning"] for r in rs) / len(rs), 1),
            "pass_rate_excl_leaks_%": round(100 * sum(r["passed"] for r in rs if not r["leaked_reasoning"])
                                            / max(1, sum(not r["leaked_reasoning"] for r in rs)), 1),
            "first_error": next((r["error"] for r in rs if r["error"]), ""),
            "median_latency_s": round(statistics.median(lat), 2) if lat else None,
            "avg_judge_quality": round(statistics.mean(
                r["judge_quality"] for r in ok if r["judge_quality"] is not None), 1)
                if any(r["judge_quality"] is not None for r in ok) else None,
        })
    summary.sort(key=lambda s: s["pass_rate_%"], reverse=True)

    by_cat = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cat[r["category"]][r["provider"]].append(r["passed"])
    category_table = {
        cat: {p: round(100 * sum(v) / len(v)) for p, v in provs.items()}
        for cat, provs in sorted(by_cat.items())
    }
    return summary, category_table


def write_outputs(rows, summary, category_table, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    detail = out_dir / f"arabic_eval_detail_{stamp}.csv"
    with detail.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = out_dir / f"arabic_eval_summary_{stamp}.json"
    summ.write_text(json.dumps(
        {"summary": summary, "pass_rate_by_category_%": category_table,
         "caveats": ["Groq is both judge and contestant: avg_judge_quality is biased for Groq.",
                     "Free-tier models/quotas change; results are a snapshot of this date."]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return detail, summ


def print_summary(summary, category_table):
    print("\n=== Provider summary (sorted by deterministic pass rate) ===")
    cols = ["provider", "pass_rate_%", "failure_rate_%", "truncated_%",
            "wrong_language_%", "leaked_reasoning_%", "pass_rate_excl_leaks_%",
            "median_latency_s", "avg_judge_quality", "n"]
    print(" | ".join(cols))
    for s in summary:
        print(" | ".join(str(s[c]) for c in cols))
    for s in summary:
        if s["first_error"]:
            print(f"\n[ERROR] {s['provider']} ({s['model']}): {s['first_error']}")
    print("\n=== Pass rate by category (%) ===")
    for cat, provs in category_table.items():
        print(f"{cat:24s} " + "  ".join(f"{p}={v}" for p, v in sorted(provs.items())))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval_set_ar.jsonl")
    ap.add_argument("--runs", type=int, default=3, help="repeat each prompt to measure variance")
    ap.add_argument("--delay", type=float, default=4.0, help="seconds between prompts (rate limits)")
    ap.add_argument("--no-judge", action="store_true", help="skip the Groq judge (saves quota)")
    ap.add_argument("--out-dir", default="eval_results")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (quick checks)")
    ap.add_argument("--providers", default="", help="comma-separated subset, e.g. Groq,Cohere,OpenRouter")
    args = ap.parse_args()

    if args.providers:
        wanted = {p.strip().lower() for p in args.providers.split(",") if p.strip()}
        unknown = wanted - {name.lower() for name in app.PROVIDERS}
        if unknown:
            sys.exit(f"Unknown or unconfigured provider(s): {', '.join(sorted(unknown))}. "
                     f"Active: {', '.join(app.PROVIDERS)}")
        for name in list(app.PROVIDERS):
            if name.lower() not in wanted:
                del app.PROVIDERS[name]
        print(f"Running only: {', '.join(app.PROVIDERS)}")

    if not app.PROVIDERS:
        sys.exit("No providers configured. Add API keys to .env first.")

    eval_set = load_eval_set(Path(args.eval_set))
    if args.limit:
        eval_set = eval_set[: args.limit]
    rows = asyncio.run(run(eval_set, args.runs, args.delay, not args.no_judge))
    summary, category_table = summarize(rows)
    print_summary(summary, category_table)
    detail, summ = write_outputs(rows, summary, category_table, Path(args.out_dir))
    print(f"\nSaved: {detail}\nSaved: {summ}")


if __name__ == "__main__":
    main()
