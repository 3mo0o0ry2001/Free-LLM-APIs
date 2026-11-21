import json
from pathlib import Path

import pytest

from evals.checks import extract_code, numbers_in, run_checks

SET = Path(__file__).resolve().parents[1] / "evals" / "questions_v1.jsonl"
ITEMS = [json.loads(l) for l in SET.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_set_shape():
    assert len(ITEMS) == 100
    assert len({i["id"] for i in ITEMS}) == 100
    for it in ITEMS:
        assert {"id", "category", "lang", "prompt", "checks", "reference", "wrong"} <= it.keys()


@pytest.mark.parametrize("item", ITEMS, ids=[i["id"] for i in ITEMS])
def test_reference_passes_and_wrong_fails(item):
    # Code execution is opt-in; the reference set is ours, so run the coding checks.
    assert run_checks(item["reference"], item["checks"], item["lang"], allow_exec=True) == []
    assert run_checks(item["wrong"], item["checks"], item["lang"], allow_exec=True) != []


@pytest.mark.parametrize("answer,expected,ok", [
    ("72", 72, True),
    ("**72**", 72, True),
    ("Average speed = 180 / 2.5 = 72 km/h", 72, True),
    ("1024 (2^10)", 1024, True),
    ("The ball costs 5 cents.", 5, True),
    ("$50", 50, True),
    ("5,050", 5050, True),
    ("٢٢٥", 225, True),
    ("180 / 2.5 = 75", 72, False),
    ("", 72, False),
])
def test_final_number_formats(answer, expected, ok):
    assert (run_checks(answer, {"final_number": expected}) == []) is ok


def test_numbers_in_handles_arabic_and_commas():
    assert numbers_in("السعر ١٬٢٥٠ درهم") == [1250.0]
    assert numbers_in("3.5 hours") == [3.5]


def test_extract_code_prefers_block_with_entry():
    ans = "Example:\n```python\nprint(f(1))\n```\nCode:\n```python\ndef f(x):\n    return x\n```"
    assert "def f" in extract_code(ans, "f")


def test_code_check_with_explanation_and_main_guard():
    ans = ("Here you go:\n```python\ndef gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n\n"
           "if __name__ == '__main__':\n    print(gcd(4, 6))\n```\nThis uses Euclid.")
    assert run_checks(ans, {"python_tests": {"entry": "gcd", "tests": "assert gcd(48, 18) == 6"}},
                      allow_exec=True) == []


def test_code_check_infinite_loop_times_out(monkeypatch):
    import evals.checks as c
    monkeypatch.setattr(c, "CODE_TIMEOUT_S", 1)
    ans = "def f():\n    while True:\n        pass"
    assert run_checks(ans, {"python_tests": {"entry": "f", "tests": "f()"}}, allow_exec=True) == ["python_tests"]


def test_code_check_skipped_without_exec():
    assert run_checks("def f(): return 1", {"python_tests": {"entry": "f", "tests": ""}},
                      allow_exec=False) == ["exec_skipped"]


def test_arabic_ratio_default_only_for_arabic():
    assert run_checks("Abu Dhabi", {"contains_any": ["Abu Dhabi"]}, lang="en") == []
    assert "arabic_ratio" in run_checks("Abu Dhabi", {"contains_any": ["Abu Dhabi"]}, lang="ar")


# ---- lenient scoring: shapes modeled on Cohere answers seen in run v1 ----
from evals.checks import run_checks_lenient, tail_answer  # noqa: E402

NUM72 = {"final_number": 72}


@pytest.mark.parametrize("answer,checks", [
    ("The user asks...\n\nWe need to compute 180 / 2.5 = 72 km/h.\n\nThus final answer: 72.72", NUM72),
    ("50\n\nWait, let me double-check. 40 / 0.8 = $50. So the answer is 50.50", {"final_number": 50}),
    ("Both approaches agree: 5 minutes.\n\nTherefore, the final answer is:\n\n55", {"final_number": 5}),
    ("100 / 7 = 14 remainder 2, so Thursday then Friday.\n\nAnswer: Friday.Friday",
     {"contains_any": ["Friday"], "not_contains": ["Thursday"], "max_words": 3}),
])
def test_lenient_rescues_glued_correct_answers(answer, checks):
    assert run_checks(answer, checks) != []          # strict: fails, as the user would see it
    assert run_checks_lenient(answer, checks) == []   # lenient: the final answer is right


@pytest.mark.parametrize("answer,checks", [
    ("Average of the six numbers.\n\nSo the answer is 18.5", {"final_number": 18}),  # genuinely wrong
    ("Speed is 180 / 2.5 = 75.75", NUM72),
    ("Because of this I would say 7272", NUM72),
    ("Long argument.\n\nAnswer: Thursday.Thursday",
     {"contains_any": ["Friday"], "max_words": 3}),
])
def test_lenient_does_not_rescue_wrong_answers(answer, checks):
    assert run_checks_lenient(answer, checks) != []


def test_lenient_never_applies_to_open_ended_or_code_checks():
    refuse = {"contains_any": ["not aware"]}  # no brevity check: stays strict
    long_wrong = "The novel follows a glassblower...\n\nI am not aware"
    assert run_checks_lenient(long_wrong, refuse) == run_checks(long_wrong, refuse) == []
    hallucinated = "I am not aware of it, but the plot is X.\n\nThe heroine escapes."
    assert run_checks_lenient(hallucinated, {"contains_any": ["escapes"], "min_words": 99}) != []
    code = {"python_tests": {"entry": "f", "tests": "assert f() == 1"}}
    assert run_checks_lenient("def f(): return 2", code) != []


def test_lenient_is_never_stricter_than_strict():
    for it in ITEMS:
        for text in (it["reference"], it["wrong"]):
            if run_checks(text, it["checks"], it["lang"]) == []:
                assert run_checks_lenient(text, it["checks"], it["lang"]) == []


def test_tail_answer_strips_label_and_glue():
    assert tail_answer("blah\nFinal answer: **42.42**") == "42"
    assert tail_answer("it takes 5 minutes\n\n55", expected_number=5) == "5"
    assert tail_answer("x\n\n55", expected_number=5) == "55"  # 5 never stated before: not a glue
    assert tail_answer("x\n\n1515", expected_number=5) == "1515"
