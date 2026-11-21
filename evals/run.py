"""
Run the router on a fixed question set and save every raw output.

    python -m evals.run                         # new run, all providers
    python -m evals.run --run-id 2026-11-15     # same id again = resume
    python -m evals.run --limit 3 --runs 1      # smoke test

Raw results go to eval_results/runs/<run-id>/raw.jsonl (one line per
question x repetition, appended as it goes, so Ctrl-C or a quota stop
loses nothing). Answers are scored afterwards by evals.analyze, so a
fix to a check never requires calling the APIs again.
"""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import app

from evals import analyze, safe_paths

RESULT_FIELDS = [
    "provider", "model", "status", "latency", "timeout", "attempts", "finish_reason",
    "error", "answer", "quality_score", "final_score", "judge_status", "judge_position",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "app.py", "evals"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def provider_snapshot() -> dict:
    return {
        name: {k: v for k, v in cfg.items() if k != "client"}
        for name, cfg in app.PROVIDERS.items()
    }


def done_keys(raw_path: Path) -> set[tuple[int, str]]:
    if not safe_paths.exists(raw_path.parent, raw_path.name):
        return set()
    keys = set()
    with safe_paths.open_read(raw_path.parent, raw_path.name) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                keys.add((rec["run"], rec["id"]))
    return keys


async def run_one(item: dict, run_idx: int, seed: int, use_judge: bool) -> dict:
    started = time.time()
    results = await app.ask_all_models(item["prompt"])
    judge_meta = {"status": "disabled"}
    if use_judge:
        rng = random.Random(f"{seed}:{run_idx}:{item['id']}")
        judge_meta = await app.evaluate_quality(item["prompt"], results, rng=rng)
    app.apply_final_scores(results)
    best = app.get_best_provider(results)
    fallback = app.get_fallback(results, best)
    usable_scores = sorted((r["final_score"] for r in results if r["status"] != "failed"), reverse=True)
    return {
        "run": run_idx,
        "id": item["id"],
        "category": item["category"],
        "lang": item["lang"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "results": [{k: r.get(k) for k in RESULT_FIELDS} for r in results],
        "judge": judge_meta,
        "router": {
            "best": best["provider"] if best else None,
            "fallback": fallback["provider"] if fallback else None,
            "best_was_tie": len(usable_scores) > 1 and usable_scores[0] == usable_scores[1],
        },
    }


async def run_all(questions, args, raw_path: Path):
    todo = [(r, q) for r in range(1, args.runs + 1) for q in questions]
    already = done_keys(raw_path)
    todo = [(r, q) for r, q in todo if (r, q["id"]) not in already]
    print(f"{len(already)} already done, {len(todo)} to run.")
    run_dir = raw_path.parent
    safe_paths.entry(run_dir, "rate_limited.jsonl")
    streak = 0
    with safe_paths.open_append(run_dir, raw_path.name) as f:
        for n, (run_idx, item) in enumerate(todo, 1):
            rec = await run_one(item, run_idx, args.seed, not args.no_judge)
            limited = [r["provider"] for r in rec["results"]
                       if analyze.error_kind(r["status"], r["error"]) == "rate_limit"]
            if limited and not args.keep_rate_limited:
                # Quota errors are not model failures: set the record aside
                # (kept for transparency) so a later resume re-asks it.
                with safe_paths.open_append(run_dir, "rate_limited.jsonl") as d:
                    d.write(json.dumps(rec, ensure_ascii=False) + "\n")
                streak += 1
                print(f"[{n}/{len(todo)}] run={run_idx} {item['id']:<18} DEFERRED rate limit: {', '.join(limited)}",
                      flush=True)
                if streak >= args.stop_after_rate_limits:
                    print(f"\n{args.stop_after_rate_limits} rate-limited questions in a row. Stopping. "
                          f"Resume later with: python -m evals.run --run-id {raw_path.parent.name} (same flags), "
                          f"or drop the limited provider with --providers.")
                    return
            else:
                streak = 0
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                statuses = " ".join(f"{r['provider']}={r['status'][0]}" for r in rec["results"])
                print(f"[{n}/{len(todo)}] run={run_idx} {item['id']:<18} best={rec['router']['best']}  {statuses}",
                      flush=True)
            if args.delay and n < len(todo):
                await asyncio.sleep(args.delay)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="evals/questions_v1.jsonl")
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"),
                    help="reuse an existing id to resume that run")
    ap.add_argument("--runs", type=int, default=1, help="repetitions of every question (variance)")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between questions (rate limits)")
    ap.add_argument("--seed", type=int, default=1234, help="seed for the judge's answer order")
    ap.add_argument("--providers", default="", help="comma-separated subset, e.g. Groq,Cohere,OpenRouter")
    ap.add_argument("--categories", default="", help="comma-separated subset of categories")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge (saves Groq quota)")
    ap.add_argument("--exec", action="store_true",
                    help="analysis: execute model-written code (untrusted code on this machine; off by default)")
    ap.add_argument("--no-exec", action="store_true", help="analysis: do not execute model-written code (the default)")
    ap.add_argument("--stop-after-rate-limits", type=int, default=3,
                    help="stop after this many consecutive rate-limited questions")
    ap.add_argument("--keep-rate-limited", action="store_true",
                    help="record rate-limited questions as failures instead of deferring them")
    ap.add_argument("--out-dir", default="eval_results/runs")
    args = ap.parse_args()
    allow_exec = args.exec and not args.no_exec
    if allow_exec:
        print(analyze.EXEC_WARNING, file=sys.stderr)

    if args.providers:
        wanted = {p.strip().lower() for p in args.providers.split(",") if p.strip()}
        unknown = wanted - {n.lower() for n in app.PROVIDERS}
        if unknown:
            sys.exit(f"Unknown or unconfigured provider(s): {', '.join(sorted(unknown))}. "
                     f"Active: {', '.join(app.PROVIDERS)}")
        for name in list(app.PROVIDERS):
            if name.lower() not in wanted:
                del app.PROVIDERS[name]
    if not app.PROVIDERS:
        sys.exit("No providers configured. Add API keys to .env first.")

    set_path = Path(args.set)
    questions = load_questions(set_path)
    if args.categories:
        cats = {c.strip() for c in args.categories.split(",")}
        questions = [q for q in questions if q["category"] in cats]
    if args.limit:
        questions = questions[: args.limit]

    run_dir = Path(args.out_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name in ("config.json", "raw.jsonl", "rate_limited.jsonl", *analyze.OUTPUT_FILES):
            safe_paths.entry(run_dir, name)
    except safe_paths.UnsafePathError as e:
        sys.exit(f"Refusing to use {run_dir}: {e}")
    cfg_path = run_dir / "config.json"
    config = {
        "run_id": args.run_id,
        "question_set": str(set_path),
        "question_set_sha256": sha256(set_path),
        "question_ids": [q["id"] for q in questions],
        "runs": args.runs,
        "seed": args.seed,
        "judge_enabled": not args.no_judge,
        "judge_model": (os.getenv("JUDGE_MODEL") or app.PROVIDERS.get("Groq", {}).get("model"))
        if not args.no_judge else None,
        "providers": provider_snapshot(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if cfg_path.exists():
        old = json.loads(safe_paths.read_text(run_dir, "config.json"))
        for key in ("question_set_sha256", "question_ids", "providers", "judge_model", "seed"):
            if old.get(key) != config[key]:
                sys.exit(f"Cannot resume {args.run_id}: '{key}' differs from the saved config. "
                         f"Use a new --run-id.")
        config["created_at"] = old.get("created_at")
        config["resumed_at"] = old.get("resumed_at", []) + [time.strftime("%Y-%m-%dT%H:%M:%S")]
    else:
        config["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    safe_paths.write_text(run_dir, "config.json", json.dumps(config, ensure_ascii=False, indent=2))

    print(f"Run {args.run_id}: {len(questions)} questions x {args.runs} run(s), "
          f"providers: {', '.join(app.PROVIDERS)}")
    try:
        asyncio.run(run_all(questions, args, run_dir / "raw.jsonl"))
    except KeyboardInterrupt:
        print(f"\nInterrupted. Resume with: python -m evals.run --run-id {args.run_id} "
              f"(same flags)")
        return

    if not safe_paths.exists(run_dir, "raw.jsonl") or not safe_paths.read_text(run_dir, "raw.jsonl").strip():
        print("Nothing scored yet, so no results to analyze.")
        return
    analyze.analyze(run_dir, set_path, allow_exec=allow_exec)


if __name__ == "__main__":
    main()
