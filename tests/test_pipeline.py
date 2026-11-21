"""
End-to-end test of run -> raw.jsonl -> analyze with FAKE providers.
No network. The numbers asserted here describe the fakes, not real models.
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import app
from evals import run as eval_run

SET = Path(__file__).resolve().parents[1] / "evals" / "questions_v1.jsonl"
ITEMS = [json.loads(l) for l in SET.read_text(encoding="utf-8").splitlines() if l.strip()]
REF = {i["prompt"]: i["reference"] for i in ITEMS}
WRONG = {i["prompt"]: i["wrong"] for i in ITEMS}


class FakeChat:
    """OpenAI-style chat client that always answers with a fixed policy."""

    def __init__(self, policy, delay=0.0, finish="stop"):
        self.policy, self.delay, self.finish = policy, delay, finish
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, model, messages, **kw):
        await asyncio.sleep(self.delay)
        prompt = messages[-1]["content"]
        if self.policy == "glued":  # right answer after leaked reasoning, duplicated: "72.72"
            text = f"The user asks: {prompt}\n\nThus final answer: {REF[prompt]}.{REF[prompt]}"
        else:
            text = REF[prompt] if self.policy == "good" else WRONG[prompt]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                                        finish_reason=self.finish)])


class FakeResponses:
    """Groq-style Responses API client: correct contestant + reference-aware judge."""

    def __init__(self, incomplete=False):
        self.incomplete = incomplete
        self.responses = SimpleNamespace(create=self._create)
        self.judge_inputs = []

    async def _create(self, model, input, **kw):
        if "impartial evaluator" in input:
            self.judge_inputs.append(input)
            prompt = input.split("Original user request:\n", 1)[1].split("\n\nEvaluate every answer", 1)[0]
            blocks = re.findall(r"<<<BEGIN ANSWER ([A-Z]) (\w+)>>>\n(.*?)\n<<<END ANSWER \1 \2>>>", input, re.S)
            blocks = [(lab, txt) for lab, _boundary, txt in blocks]
            scores = {lab: (90 if txt.strip() == REF[prompt].strip() else 30) for lab, txt in blocks}
            return SimpleNamespace(output_text=json.dumps(scores), status="completed", incomplete_details=None)
        status = "incomplete" if self.incomplete else "completed"
        details = SimpleNamespace(reason="max_output_tokens") if self.incomplete else None
        return SimpleNamespace(output_text=REF[input], status=status, incomplete_details=details)


def cfg(client, timeout=5.0, retries=0):
    return {"client": client, "model": "fake-model", "timeout": timeout, "max_tokens": 100, "retries": retries}


@pytest.fixture
def fake_providers(monkeypatch):
    groq = FakeResponses()
    providers = {
        "Groq": cfg(groq),
        "Cohere": cfg(FakeChat("wrong")),
        "Mistral": cfg(FakeChat("good", delay=1.0), timeout=0.05),
    }
    monkeypatch.setattr(app, "PROVIDERS", providers)
    monkeypatch.setattr(app, "RETRY_DELAY", 0)
    return groq


def run_cli(monkeypatch, tmp_path, *extra):
    argv = ["evals.run", "--set", str(SET), "--out-dir", str(tmp_path), "--run-id", "t", "--delay", "0", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    eval_run.main()
    return tmp_path / "t"


def test_full_pipeline_and_resume(fake_providers, monkeypatch, tmp_path, capsys):
    run_dir = run_cli(monkeypatch, tmp_path, "--exec")
    raw = (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw) == 100

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    prov = {p["provider"]: p for p in summary["providers"]}
    assert prov["Groq"]["reference_accuracy_%"] == 100.0
    assert prov["Cohere"]["reference_accuracy_%"] == 0.0
    assert prov["Mistral"]["timeout_rate_%"] == 100.0
    assert prov["Groq"]["win_rate_%"] == 100.0
    assert summary["router"]["router_accuracy_%"] == 100.0
    assert summary["judge_audit"]["judge_ranks_correct_higher_%"] == 100.0
    assert summary["judge_audit"]["judge_failure_rate_%"] == 0.0
    assert (run_dir / "results.md").exists() and (run_dir / "answers.csv").exists()

    # Judge never sees provider names.
    assert not any(n in inp for inp in fake_providers.judge_inputs for n in ("Groq", "Cohere", "Mistral"))

    # Same run id again -> nothing re-run, nothing duplicated.
    run_cli(monkeypatch, tmp_path)
    assert "0 to run" in capsys.readouterr().out
    assert len((run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()) == 100


def test_resume_refuses_changed_config(fake_providers, monkeypatch, tmp_path):
    run_cli(monkeypatch, tmp_path, "--limit", "2")
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, tmp_path, "--limit", "3")


def test_judge_order_is_shuffled_and_seeded(fake_providers):
    item = ITEMS[0]
    orders = set()
    for seed in range(12):
        rec = asyncio.run(eval_run.run_one(item, 1, seed, True))
        orders.add(tuple(rec["judge"]["order"]))
    assert len(orders) > 1  # not a fixed order
    a = asyncio.run(eval_run.run_one(item, 1, 99, True))["judge"]["order"]
    b = asyncio.run(eval_run.run_one(item, 1, 99, True))["judge"]["order"]
    assert a == b  # reproducible


def test_groq_truncation_detected(monkeypatch):
    monkeypatch.setattr(app, "PROVIDERS", {"Groq": cfg(FakeResponses(incomplete=True))})
    r = asyncio.run(app.ask_provider("Groq", app.PROVIDERS["Groq"], ITEMS[0]["prompt"]))
    assert r["status"] == "truncated"


def test_judge_failure_is_flagged(monkeypatch):
    class BrokenJudge(FakeResponses):
        async def _create(self, model, input, **kw):
            if "impartial evaluator" in input:
                return SimpleNamespace(output_text="not json", status="completed", incomplete_details=None)
            return await super()._create(model, input, **kw)

    j = BrokenJudge()
    j.responses = SimpleNamespace(create=j._create)
    monkeypatch.setattr(app, "PROVIDERS", {"Groq": cfg(j), "Cohere": cfg(FakeChat("wrong"))})
    results = asyncio.run(app.ask_all_models(ITEMS[0]["prompt"]))
    meta = asyncio.run(app.evaluate_quality(ITEMS[0]["prompt"], results))
    assert meta["status"] == "failed"
    assert all(r["judge_status"] == "failed" and r["quality_score"] == 50.0 for r in results)


def test_rate_limits_are_deferred_not_counted(monkeypatch, tmp_path, capsys):
    class QuotaError(Exception):
        status_code = 429

    class Limited(FakeChat):
        async def _create(self, model, messages, **kw):
            raise QuotaError("Error code: 429 - rate limit exceeded")

    monkeypatch.setattr(app, "PROVIDERS", {"Groq": cfg(FakeResponses()), "Gemini": cfg(Limited("good"))})
    monkeypatch.setattr(app, "RETRY_DELAY", 0)
    run_dir = run_cli(monkeypatch, tmp_path, "--limit", "5", "--stop-after-rate-limits", "2")
    out = capsys.readouterr().out
    assert "Stopping" in out
    assert not (run_dir / "raw.jsonl").read_text(encoding="utf-8").strip()
    assert len((run_dir / "rate_limited.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_right_but_dirty_is_reported_separately(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "PROVIDERS", {"Groq": cfg(FakeResponses()), "Cohere": cfg(FakeChat("glued"))})
    run_dir = run_cli(monkeypatch, tmp_path, "--categories", "reasoning", "--no-exec")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cohere = next(p for p in summary["providers"] if p["provider"] == "Cohere")
    assert cohere["reference_accuracy_%"] == 0.0      # strict: not what the prompt asked for
    assert cohere["lenient_accuracy_%"] == 100.0      # the final answers are right
    assert cohere["right_but_dirty_%"] == 100.0
    groq = next(p for p in summary["providers"] if p["provider"] == "Groq")
    assert groq["right_but_dirty_%"] == 0.0
