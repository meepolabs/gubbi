"""Fail unless a workflow's ``required`` job lists every other job in ``needs``.

Branch protection names only ``required``, so a job missing from its ``needs``
gates nothing while still looking like CI. This runs as a step of ``required``
itself, on the runner's own ``python3`` with nothing installed, so it is
standard-library only and reads the workflow line by line instead of with a
YAML parser.

It accepts exactly one layout and fails closed on anything else, so a reformat
can never make it pass vacuously:

    jobs:                     # trailing comments are allowed throughout
      <id>:                   # unquoted job ids at two spaces
        ...                   # job bodies at four or more spaces
      required:
        needs:                # exactly one block-list ``needs:`` key
          - <id>              # unquoted items at six spaces

Blank lines and full-line comments are ignored, and the bodies of block
scalars (``run: |``) are skipped, never read as jobs or ``needs``.

Usage: ``python3 tools/check_required_needs.py [--self-test] <workflow.yml>``

``--self-test`` is the positive control: it drops each entry of the
``required`` job's ``needs`` block in turn and fails unless the check reports
every removal.

Exit 0 when ``needs`` is complete, 1 on a finding, 2 on a usage error, an
unreadable file, or a layout the check does not accept.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

AGGREGATOR = "required"

_ID = r"[A-Za-z_][A-Za-z0-9_-]*"
_COMMENT = r"(?:[ \t]+#.*)?"
_JOBS_KEY = re.compile(rf"^jobs:{_COMMENT}[ \t]*$")
_JOBS_ANY = re.compile(r"^[\"']?jobs[\"']?[ \t]*:")
_TOP_LEVEL = re.compile(r"^[^\s#]")
_TOP_LEVEL_KEY = re.compile(rf"^{_ID}:")
_JOB_HEADER = re.compile(rf"^ {{2}}({_ID}):{_COMMENT}[ \t]*$")
_BODY_KEY = re.compile(rf"^ {{4}}{_ID}:")
_NEEDS_ANY = re.compile(r"^ {4}[\"']?needs[\"']?[ \t]*:")
_NEEDS_KEY = re.compile(rf"^ {{4}}needs:{_COMMENT}[ \t]*$")
_NEEDS_ITEM = re.compile(rf"^ {{6}}- ({_ID}){_COMMENT}[ \t]*$")
_BLOCK_SCALAR = re.compile(
    rf"^ *(?:- +)?[A-Za-z0-9_.-]+:[ \t]*[|>][+-]?[0-9]?[+-]?{_COMMENT}[ \t]*$"
)

_FOREIGN_BREAK = re.compile("\r(?!\n)|[\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")

_JOB_INDENT = 2
_BODY_INDENT = 4


class LayoutError(ValueError):
    """The workflow is not in the one layout this check accepts."""


class Entry(NamedTuple):
    """A job id together with the 1-based line it appears on."""

    name: str
    line: int


class Workflow(NamedTuple):
    """The job headers and the ``required`` job's ``needs`` entries, in file order."""

    jobs: list[Entry]
    needs: list[Entry]


def _lines(source: str) -> list[str]:
    """Split on ``\\n`` / ``\\r\\n`` only, refusing any other character Python would split on."""
    foreign = _FOREIGN_BREAK.search(source)
    if foreign is not None:
        line_no = source.count("\n", 0, foreign.start()) + 1
        raise LayoutError(f"line {line_no}: unsupported line-break character {foreign.group()!r}")
    return source.splitlines()


def _spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _indent(line_no: int, line: str) -> int:
    spaces = _spaces(line)
    if line[spaces:].startswith("\t"):
        raise LayoutError(f"line {line_no}: tab in indentation")
    return spaces


def _structural_lines(lines: list[str], start: int, end: int) -> list[tuple[int, str]]:
    """``(line number, line)`` for every line in ``lines[start:end]`` that carries structure."""
    kept: list[tuple[int, str]] = []
    i = start
    while i < end:
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        kept.append((i, line))
        if _BLOCK_SCALAR.search(line):
            key_indent = _indent(i, line)
            while i < end and (not lines[i].strip() or _spaces(lines[i]) > key_indent):
                i += 1
    return kept


def _jobs_section(lines: list[str]) -> tuple[int, int]:
    keys = [i for i, line in enumerate(lines) if _JOBS_ANY.match(line)]
    if not keys:
        raise LayoutError("no top-level `jobs:` key")
    if len(keys) > 1:
        raise LayoutError(f"line {keys[1] + 1}: a second top-level `jobs:` key")
    if not _JOBS_KEY.match(lines[keys[0]]):
        raise LayoutError(f"line {keys[0] + 1}: `jobs:` must be an unquoted key on its own line")
    start = keys[0] + 1
    end = next((i for i in range(start, len(lines)) if _TOP_LEVEL.match(lines[i])), len(lines))
    if end < len(lines) and not _TOP_LEVEL_KEY.match(lines[end]):
        raise LayoutError(f"line {end + 1}: expected a top-level key after `jobs:`")
    return start, end


def _job_bodies(lines: list[str]) -> tuple[list[Entry], dict[str, list[tuple[int, str]]]]:
    jobs: list[Entry] = []
    bodies: dict[str, list[tuple[int, str]]] = {}
    for line_no, line in _structural_lines(lines, *_jobs_section(lines)):
        indent = _indent(line_no, line)
        if indent >= _BODY_INDENT and jobs:
            bodies[jobs[-1].name].append((line_no, line))
            continue
        header = _JOB_HEADER.match(line) if indent == _JOB_INDENT else None
        if header is None:
            raise LayoutError(f"line {line_no}: expected an unquoted `<id>:` job header")
        if header.group(1) in bodies:
            raise LayoutError(f"line {line_no}: job `{header.group(1)}` is defined twice")
        jobs.append(Entry(header.group(1), line_no))
        bodies[header.group(1)] = []
    return jobs, bodies


def _needs_entries(body: list[tuple[int, str]]) -> list[Entry]:
    for line_no, line in body:
        if _indent(line_no, line) == _BODY_INDENT and not _BODY_KEY.match(line):
            raise LayoutError(f"line {line_no}: expected an unquoted key in job `{AGGREGATOR}`")
    keys = [at for at, (_, line) in enumerate(body) if _NEEDS_ANY.match(line)]
    if not keys:
        raise LayoutError(f"job `{AGGREGATOR}` has no `needs:` key")
    if len(keys) > 1:
        raise LayoutError(f"line {body[keys[1]][0]}: job `{AGGREGATOR}` has a second `needs:` key")
    key_no, key_line = body[keys[0]]
    if not _NEEDS_KEY.match(key_line):
        raise LayoutError(f"line {key_no}: `needs:` must be a block list, one job per line")

    entries: list[Entry] = []
    for line_no, line in body[keys[0] + 1 :]:
        if _indent(line_no, line) <= _BODY_INDENT:
            break
        item = _NEEDS_ITEM.match(line)
        if item is None:
            raise LayoutError(f"line {line_no}: expected a six-space unquoted `- <id>` entry")
        entries.append(Entry(item.group(1), line_no))
    return entries


def parse_workflow(source: str) -> Workflow:
    """Read the job headers and the ``required`` job's ``needs``; raise LayoutError otherwise."""
    jobs, bodies = _job_bodies(_lines(source))
    if AGGREGATOR not in bodies:
        raise LayoutError(f"no `{AGGREGATOR}` job")
    return Workflow(jobs, _needs_entries(bodies[AGGREGATOR]))


def find_problems(source: str) -> list[str]:
    """Every way the ``required`` job's ``needs`` differs from the job set; empty means complete."""
    workflow = parse_workflow(source)
    aggregator = next(job for job in workflow.jobs if job.name == AGGREGATOR)
    others = {job.name for job in workflow.jobs if job.name != AGGREGATOR}
    listed = {entry.name for entry in workflow.needs}
    found: list[tuple[int, str]] = [
        (job.line, f"job `{job.name}` is missing from `{AGGREGATOR}.needs`")
        for job in workflow.jobs
        if job.name in others - listed
    ]
    seen: set[str] = set()
    for entry in workflow.needs:
        if entry.name == AGGREGATOR:
            found.append((entry.line, f"`{AGGREGATOR}` lists itself in `needs`"))
        elif entry.name not in others:
            found.append((entry.line, f"`{entry.name}` in `needs` is not a job in this workflow"))
        if entry.name in seen:
            found.append((entry.line, f"`{entry.name}` is listed in `needs` more than once"))
        seen.add(entry.name)
    if not others:
        found.append((aggregator.line, f"`{AGGREGATOR}` has no other job to aggregate"))
    return [f"line {line}: {message}" for line, message in sorted(found)]


def self_test_failures(source: str) -> list[str]:
    """Each ``required.needs`` entry whose removal the check would NOT report."""
    entries = parse_workflow(source).needs
    lines = source.splitlines(keepends=True)
    if not entries:
        return [f"`{AGGREGATOR}.needs` has no entries to remove"]
    return [
        f"line {entry.line}: dropping `{entry.name}` from `needs` went unreported"
        for entry in entries
        if not find_problems("".join(lines[: entry.line - 1] + lines[entry.line :]))
    ]


def main(argv: list[str] | None = None) -> int:
    """Check the workflow named on the command line; see the module docstring for exit codes."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    try:
        source = args.workflow.read_text(encoding="utf-8")
        problems = find_problems(source)
        if not problems and args.self_test:
            problems = self_test_failures(source)
    except (OSError, UnicodeDecodeError, LayoutError) as exc:
        sys.stderr.write(f"{args.workflow}: cannot check: {exc}\n")
        return 2

    for problem in problems:
        sys.stderr.write(f"{args.workflow}: {problem}\n")
    if problems:
        return 1
    sys.stdout.write(f"{args.workflow}: `{AGGREGATOR}` needs every other job\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
