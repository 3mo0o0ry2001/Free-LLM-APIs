"""
Helpers for text that comes from providers or from a run directory.

strip_control: remove terminal escape sequences and control characters before
printing, so model output cannot rewrite the clipboard (OSC 52), the window
title (OSC 0) or earlier lines on screen (CSI). Newlines and tabs are kept.
"""

import re

_ESC_SEQ = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"           # CSI ... final byte
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC ... BEL or ST
    r"|\x1b[PX^_][^\x1b]*(?:\x1b\\)?"     # DCS / SOS / PM / APC
    r"|\x1b[@-Z\\-_]"                     # other two-byte escapes
    r"|\x9b[0-?]*[ -/]*[@-~]"             # 8-bit CSI
    r"|\x9d[^\x07\x9c]*[\x07\x9c]?"       # 8-bit OSC
)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")  # keeps \t and \n


def strip_control(text) -> str:
    """Printable copy of `text`: escape sequences and control characters removed."""
    if text is None:
        return ""
    s = str(text).replace("\r\n", "\n")
    s = _ESC_SEQ.sub("", s)
    return _CONTROL.sub("", s)


# Account identifiers that providers echo back in error bodies (OpenRouter's 429
# includes the caller's user_id). Saved errors keep the status code and message
# so rate limits are still recognised, but the identifier value is replaced.
_ID_FIELD = re.compile(
    r"""(['"])(user_id|userId|org_id|organization|organization_id|account_id|project_id)\1(\s*:\s*)(['"])(.*?)\4"""
)


def redact_ids(text):
    """Copy of `text` with account-identifier values replaced by [redacted]."""
    if text is None:
        return None
    return _ID_FIELD.sub(lambda m: f"{m[1]}{m[2]}{m[1]}{m[3]}{m[4]}[redacted]{m[4]}", str(text))
