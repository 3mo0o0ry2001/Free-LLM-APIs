"""
Deterministic answer checks (no LLM involved).

A check spec is a dict stored in the question file, e.g.
    {"contains_any": ["Canberra"]}
    {"final_number": 42}
    {"python_tests": {"entry": "fib", "tests": "assert fib(10) == 55"}}

run_checks(answer, checks, lang) returns the list of failed check names.
An empty list means the answer passed.
"""

import json
import os
import re
import secrets
import subprocess
import sys
import tempfile

# ============================================================
# NORMALIZATION
# ============================================================

DIACRITICS = re.compile(r"[ً-ْٰـ]")  # tashkeel + superscript alef + tatweel
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ARABIC_LETTER = re.compile(r"[ء-ي]")
LATIN_LETTER = re.compile(r"[A-Za-z]")

DEFAULT_ARABIC_RATIO = 0.6  # Arabic prompt -> answer should be mostly Arabic


def normalize(text: str) -> str:
    """Case-fold and normalize Arabic spelling variants and digits."""
    text = text.translate(ARABIC_INDIC)
    text = DIACRITICS.sub("", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.replace("**", "").replace("`", "")
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


NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    text = text.translate(ARABIC_INDIC).replace("٫", ".")
    text = re.sub(r"(?<=\d)[,٬](?=\d{3}\b)", "", text)  # 1,250 -> 1250
    return [float(n) for n in NUMBER.findall(text)]


# Reasoning text leaking into the final answer (seen with Cohere via the compat API)
LEAK_PATTERN = re.compile(
    r"^\s*(<EOS_TOKEN>|(the )?user (asks|wants)|يسأل المستخدم|المستخدم يطلب|نحتاج|we need|1\.\s+تحليل الطلب)",
    re.I,
)

# ============================================================
# CODE EXECUTION
# ============================================================

CODE_BLOCK = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```", re.S | re.I)
CODE_TIMEOUT_S = 10

# Only what the interpreter needs to start (SYSTEMROOT is required on Windows).
# Everything else, including the provider API keys loaded from .env, is left out.
CHILD_ENV_KEYS = ("PATH", "SYSTEMROOT")


def child_env(tmp: str) -> dict:
    """Minimal environment for model-written code: no secrets, home and temp in `tmp`."""
    env = {k: os.environ[k] for k in CHILD_ENV_KEYS if k in os.environ}
    env.update(HOME=tmp, USERPROFILE=tmp, TEMP=tmp, TMP=tmp, TMPDIR=tmp)
    return env


def extract_code(answer: str, entry: str) -> str:
    """Pick the fenced block that defines `entry`; fall back to raw text."""
    blocks = CODE_BLOCK.findall(answer)
    for block in blocks:
        if re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", block, re.M):
            return block
    if blocks:
        return "\n\n".join(blocks)
    lines = answer.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"\s*(def |import |from |class )", line):
            return "\n".join(lines[i:])
    return answer


def run_python_tests(answer: str, spec: dict) -> bool:
    """
    Execute model-written code plus hidden asserts in a child process.

    This runs code produced by an LLM on your machine, so it only happens when
    the caller passes allow_exec=True (the --exec flag). The child gets a
    minimal environment (no API keys, HOME and TEMP inside a fresh temp
    directory), that directory as its cwd, `python -I`, no stdin and a hard
    timeout. That reduces risk but it is NOT a sandbox: the code can still read
    and write any file your account can, and use the network. Real isolation
    needs a container or VM with no network and no access to your files.
    """
    code = extract_code(answer, spec["entry"])
    # Fresh marker per execution, printed only after every hidden assert has run.
    # It is never sent to a model. Limitation: it is written into the same file
    # as the candidate code, so code that deliberately reads its own source
    # (__file__) can still find it. This stops blind spoofing, not a targeted attack.
    marker = f"__PASS_{secrets.token_hex(16)}__"
    program = f"{code}\n\n# ---- hidden tests ----\n{spec['tests']}\nprint({marker!r})\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "candidate.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],
                cwd=tmp,
                env=child_env(tmp),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=CODE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return False
    lines = proc.stdout.strip().splitlines()
    return proc.returncode == 0 and bool(lines) and lines[-1].strip() == marker


# ============================================================
# CHECKS
# ============================================================

def run_checks(answer: str, checks: dict, lang: str = "en", allow_exec: bool = False) -> list[str]:
    """Return a list of failed check names (empty list = pass)."""
    failed = []
    norm = normalize(answer)

    def has(term):
        return normalize(term) in norm

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
    if "regex_full" in checks and not re.fullmatch(checks["regex_full"], answer.strip(), re.S):
        failed.append("regex_full")

    words = len(answer.split())
    if "max_words" in checks and words > checks["max_words"]:
        failed.append("max_words")
    if "min_words" in checks and words < checks["min_words"]:
        failed.append("min_words")

    if "final_number" in checks:
        # Accept the expected value as the last number of the answer
        # ("180 / 2.5 = 72") or the first number of the last line
        # ("1024 (2^10)").
        tol = checks.get("tolerance", 1e-6)
        expected = float(checks["final_number"])
        lines = [l for l in answer.strip().splitlines() if l.strip()]
        all_nums = numbers_in(answer)
        last_line_nums = numbers_in(lines[-1]) if lines else []
        candidates = all_nums[-1:] + last_line_nums[:1]
        if not any(abs(c - expected) <= tol for c in candidates):
            failed.append("final_number")

    if "numbered_items" in checks:
        items = re.findall(r"^\s*[0-9٠-٩]+[.)\-]", answer, re.M)
        if len(items) != checks["numbered_items"]:
            failed.append("numbered_items")
    if "bullet_items" in checks:
        items = re.findall(r"^\s*[-*•]\s+\S", answer, re.M)
        if len(items) != checks["bullet_items"]:
            failed.append("bullet_items")
    if "line_count" in checks:
        lines = [l for l in answer.strip().splitlines() if l.strip()]
        if len(lines) != checks["line_count"]:
            failed.append("line_count")
    if checks.get("all_uppercase") and any(c.islower() for c in answer):
        failed.append("all_uppercase")
    if checks.get("all_lowercase") and any(c.isupper() for c in answer):
        failed.append("all_lowercase")

    if "json_fields" in checks:
        obj = extract_json_obj(answer)
        if not isinstance(obj, dict):
            failed.append("json_parse")
        else:
            for key, expected in checks["json_fields"].items():
                if normalize(str(expected)) not in normalize(str(obj.get(key, ""))):
                    failed.append(f"json_{key}")
    if "json_keys_exact" in checks:
        obj = extract_json_obj(answer)
        if not isinstance(obj, dict) or sorted(obj) != sorted(checks["json_keys_exact"]):
            failed.append("json_keys_exact")

    if "python_tests" in checks:
        if not allow_exec:
            failed.append("exec_skipped")
        elif not run_python_tests(answer, checks["python_tests"]):
            failed.append("python_tests")

    default_ratio = DEFAULT_ARABIC_RATIO if lang == "ar" else 0.0
    min_ratio = checks.get("arabic_ratio_min", default_ratio)
    if min_ratio and arabic_ratio(answer) < min_ratio:
        failed.append("arabic_ratio")
    return failed


# ============================================================
# LENIENT (TAIL) SCORING
# ============================================================
# Some providers (seen with Cohere via the OpenAI-compatible API) return their
# reasoning followed by the final answer glued on the end, e.g. "...is 72.72"
# or "Friday.Friday" or "...the final answer is:\n\n55". The strict checks
# score that as wrong because the user did not get "only the number".
# The lenient score asks a different question: is the answer the model
# arrived at correct? It is reported next to the strict score, never
# instead of it, and every rescued answer is flagged in answers.csv.

GLUED_TAIL = re.compile(r"(\w+(?:\.\d+)?)\.\1\s*$")  # "72.72", "Friday.Friday"
BREVITY_CHECKS = ("final_number", "max_words", "regex_full")


def lenient_applies(checks: dict) -> bool:
    """Only questions that demand a short answer; open-ended checks stay strict."""
    return any(k in checks for k in BREVITY_CHECKS) and "python_tests" not in checks


def tail_answer(answer: str, expected_number=None) -> str:
    """Last line of the answer, label stripped, glued duplicate removed."""
    lines = [l.strip() for l in answer.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    tail = lines[-1]
    if ":" in tail:
        after = tail.rsplit(":", 1)[1].strip()
        if after:
            tail = after
    tail = tail.strip("*` ")
    tail = GLUED_TAIL.sub(r"\1", tail)
    if expected_number is not None:
        # "5" + "5" -> "55". Without a separator this looks like a real wrong
        # answer ("7272"), so only accept it when the reasoning before it
        # already states the expected value on its own.
        e = float(expected_number)
        es = str(int(e)) if e.is_integer() else str(e)
        head = "\n".join(lines[:-1])
        if re.search(rf"(?<![\d.]){re.escape(es)}(?![\d.])", head):
            tail = re.sub(rf"(?<![\d.]){re.escape(es + es)}(\s*\.?\s*)$", es, tail)
    return tail


def run_checks_lenient(answer: str, checks: dict, lang: str = "en", allow_exec: bool = False) -> list[str]:
    """Strict checks first; if they fail and the question wants a short answer, score the tail."""
    failed = run_checks(answer, checks, lang, allow_exec)
    if not failed or not lenient_applies(checks):
        return failed
    tail = tail_answer(answer, checks.get("final_number"))
    return failed if not tail or tail == answer.strip() else (run_checks(tail, checks, lang, allow_exec) or [])
