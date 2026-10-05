"""Decide whether a CI run must replay the image's hash-pinned Poetry install.

The install is slow and only a change to one of ``WATCHED_PATHS`` can break
it, so the CI lane runs it only when the diff from the base revision touches
one of them. Every way the diff cannot be determined -- no base, the all-zero
SHA a branch-creating push carries, a base object absent from the clone, any
git failure -- runs the install, so an unreadable diff never skips the check.

It runs on the runner's own ``python3`` before anything is installed, so it is
standard-library only.

Usage::

    python3 tools/image_closure_needed.py --base <sha> [--github-output <file>]

Prints the verdict and its reason. With ``--github-output`` it also appends
``run=true`` or ``run=false`` to that file. Exit 0 whenever a verdict was
reached (run or skip); 2 on a usage error or an unwritable output file.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# The only repository-specific part: every path whose change can break the
# image's Poetry install, plus the files that define this check itself.
WATCHED_PATHS = frozenset(
    {
        "deployment/Dockerfile",
        "deployment/poetry-requirements.txt",
        ".github/workflows/ci.yml",
        "tools/image_closure_needed.py",
    }
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GIT_TIMEOUT_S = 30
_ZERO_SHA = re.compile(r"^0+$")


class Verdict(NamedTuple):
    """Whether to run the install, and why."""

    run: bool
    reason: str


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    """Run git in ``repo``; None when git itself could not be run."""
    try:
        return subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def changed_paths(repo: Path, base: str, head: str = "HEAD") -> list[str] | None:
    """Paths that differ between ``base`` and ``head``; None when the diff cannot be read."""
    if not base or _ZERO_SHA.match(base):
        return None
    probe = _git(repo, "cat-file", "-e", f"{base}^{{commit}}")
    if probe is None or probe.returncode != 0:
        return None
    diff = _git(repo, "diff", "--name-only", "--no-renames", base, head, "--")
    if diff is None or diff.returncode != 0:
        return None
    return [line for line in diff.stdout.splitlines() if line]


def decide(changed: list[str] | None) -> Verdict:
    """Run unless the diff is known and touches none of ``WATCHED_PATHS``."""
    if changed is None:
        return Verdict(run=True, reason="base revision unknown or diff unreadable")
    hits = sorted(WATCHED_PATHS.intersection(changed))
    if hits:
        return Verdict(run=True, reason="changed: " + ", ".join(hits))
    return Verdict(run=False, reason="no watched path changed")


def main(argv: list[str] | None = None) -> int:
    """Print the verdict for ``--base`` against HEAD; see the module docstring for exit codes."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--base", required=True, help="base revision; empty means unknown")
    parser.add_argument("--github-output", type=Path, default=None)
    args = parser.parse_args(argv)

    verdict = decide(changed_paths(REPO_ROOT, args.base.strip()))
    value = "true" if verdict.run else "false"
    if args.github_output is not None:
        try:
            with args.github_output.open("a", encoding="utf-8") as out:
                out.write(f"run={value}\n")
        except OSError as exc:
            sys.stderr.write(f"cannot write {args.github_output}: {exc}\n")
            return 2
    sys.stdout.write(f"run={value}: {verdict.reason}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
