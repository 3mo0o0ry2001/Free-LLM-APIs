"""
Safe file access inside a run directory.

A run directory can come from someone else (a pull request, a shared or
downloaded folder), so its entries are not trusted. Every file the tools read
or write there goes through these helpers:

- the entry must not be a symbolic link and, if it exists, must be a regular file;
- its resolved path must sit directly inside the resolved run directory;
- outputs are written to a new file created exclusively (O_EXCL, no-follow),
  then moved over the target with os.replace, which replaces the directory
  entry instead of writing through whatever it points to.

On Windows os.O_NOFOLLOW does not exist, so the lstat checks are what protect
the open; a symlink created between the check and the open is not caught there.
"""

import os
import secrets
from pathlib import Path

NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
BINARY = getattr(os, "O_BINARY", 0)


class UnsafePathError(ValueError):
    """A run-directory entry is a symlink, not a regular file, or escapes the directory."""


def entry(run_dir: Path, name: str) -> Path:
    """Path of `name` inside `run_dir`, after the symlink and containment checks."""
    run_dir = Path(run_dir)
    if run_dir.is_symlink():
        raise UnsafePathError(f"{run_dir} is a symbolic link; refusing to use it as a run directory")
    if Path(name).name != name or name in ("", ".", ".."):
        raise UnsafePathError(f"{name!r} is not a plain file name")
    target = run_dir / name
    if target.is_symlink():
        raise UnsafePathError(f"{target} is a symbolic link; refusing to read or write through it")
    if target.exists() and not target.is_file():
        raise UnsafePathError(f"{target} is not a regular file")
    if target.resolve().parent != run_dir.resolve():
        raise UnsafePathError(f"{target} resolves outside {run_dir}")
    return target


def open_read(run_dir: Path, name: str, encoding: str = "utf-8"):
    """Open an existing run-directory file for reading without following links."""
    target = entry(run_dir, name)
    fd = os.open(target, os.O_RDONLY | NOFOLLOW | BINARY)
    return os.fdopen(fd, "r", encoding=encoding)


def read_text(run_dir: Path, name: str, encoding: str = "utf-8") -> str:
    with open_read(run_dir, name, encoding) as f:
        return f.read()


def exists(run_dir: Path, name: str) -> bool:
    return entry(run_dir, name).exists()


def open_append(run_dir: Path, name: str, encoding: str = "utf-8"):
    """Open a run-directory log for appending; creates it if missing, never follows a link."""
    target = entry(run_dir, name)
    fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT | NOFOLLOW | BINARY, 0o644)
    return os.fdopen(fd, "a", encoding=encoding)


def write_text(run_dir: Path, name: str, text: str, encoding: str = "utf-8", newline: str | None = None):
    """Replace `name` with `text`: exclusive temp file, then os.replace over the entry."""
    target = entry(run_dir, name)
    tmp = Path(run_dir) / f".{name}.{secrets.token_hex(8)}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW | BINARY, 0o644)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline=newline) as f:
            f.write(text)
        entry(run_dir, name)  # re-check right before the swap
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
