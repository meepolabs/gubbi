"""Dump period-resolution fixtures to stdout as deterministic JSON.

The journal web client re-implements the period vocabulary of
``gubbi.tools.context.resolve_period`` in TypeScript. This script emits the
resolver's literal output for a fixed matrix of (period, today) inputs so both
sides can pin the same bytes: the web repo commits a copy under
``apps/journal/src/lib/timeline/`` and this repo commits one under
``tests/fixtures/``, where a drift test regenerates and diffs it.

The resolver is pure -- no database, cache, environment, or running backend.
Rows are sorted by ``(period, today)`` so regeneration is order-stable, and the
output is ASCII-only with tab indentation because the web copy is checked by
prettier with ``useTabs``. Exit code is 0 on success.

Usage::

    python scripts/dump_period_fixtures.py > tests/fixtures/period_resolution.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Documented in the fixture itself so a reader of the committed artifact knows
# why two valid-looking inputs are missing from the matrix.
FIXTURE_COMMENT: list[str] = [
    "GENERATED -- do not hand-edit. Every row is the literal return of the Python",
    "resolver gubbi.tools.context.resolve_period(period, today), which is the",
    "authority for the period vocabulary published in gubbi/docs/tools-reference.md",
    "on the journal_timeline `period` parameter row (today, this-week, last-week,",
    "this-month, last-month, YYYY, YYYY-MM, YYYY-WNN).",
    "",
    "Two copies of these bytes exist, one per repo, and both are pinned: the web",
    "suite asserts its period module against them, and this repo's own drift test",
    "regenerates them and diffs, so a resolver change on either side goes red.",
    "Regenerate BOTH from this repo's generator whenever the resolver changes.",
    "",
    "`label` is recorded for completeness but is presentation, which each surface owns",
    "independently -- the web renders its own wording and never asserts this field.",
    "",
    "Two inputs are deliberately absent because the resolver raises on them today:",
    "'9999-12' (a valid month rejected -- the December branch builds date(10000, 1, 1))",
    "and '9999-W52' (raises OverflowError rather than ValueError, so it escapes the",
    "resolver's own handlers). Both are Python-side defects tracked separately; adding",
    "them would pin bugs as contract.",
]

_DEFAULT_TODAY = "2026-06-11"

# Periods probed against a single reference date. Deliberately excluded:
# '9999-12' and '9999-W52' -- see the note in FIXTURE_COMMENT.
_SINGLE_DATE_PERIODS: tuple[str, ...] = (
    "today",
    "this-week",
    "this-month",
    "",
    "abc",
    "202",
    "0000",
    "0099",
    "1000",
    "2026",
    "9999",
    "2024-02",
    "2026-02",
    "2026-03",
    "2026-12",
    "2026-00",
    "2026-13",
    "2026-W01",
    "2026-W12",
    "2026-W1",
    "2026-W012",
    "2026-W53",
    "2026-W54",
    "2026-W00",
    "2020-W53",
    "2021-W01",
    "2024-W01",
    "2025-W52",
    "2025-W53",
    "1000-W01",
    "2026_w12",
    "  2026-W12  ",
)

# Backward-looking relative periods get a second date that crosses the year
# boundary, where their arithmetic lands in the previous year.
_YEAR_BOUNDARY_CASES: tuple[tuple[str, str], ...] = (
    ("last-week", "2026-01-05"),
    ("last-month", "2026-01-01"),
)

CASES: tuple[tuple[str, str], ...] = (
    *((period, _DEFAULT_TODAY) for period in _SINGLE_DATE_PERIODS),
    ("last-week", _DEFAULT_TODAY),
    ("last-month", _DEFAULT_TODAY),
    *_YEAR_BOUNDARY_CASES,
)


def build_document() -> dict[str, Any]:
    """Resolve every case and return the fixture document."""
    from datetime import date

    from gubbi.tools.context import resolve_period

    rows: list[dict[str, Any]] = []
    for period, today in CASES:
        try:
            date_from, date_to, label = resolve_period(period, today=date.fromisoformat(today))
        except ValueError:
            # ValueError only: any other exception is an unintended resolver defect
            # and should crash regeneration rather than be pinned as contract.
            rows.append(
                {
                    "date_from": None,
                    "date_to": None,
                    "label": None,
                    "period": period,
                    "rejected": True,
                    "today": today,
                }
            )
            continue
        rows.append(
            {
                "date_from": date_from,
                "date_to": date_to,
                "label": label,
                "period": period,
                "rejected": False,
                "today": today,
            }
        )

    return {
        "_comment": FIXTURE_COMMENT,
        "rows": sorted(rows, key=lambda row: (row["period"], row["today"])),
    }


def main() -> int:
    """Serialise the fixture document to stdout and return an exit code."""
    # The package is run from a source checkout (poetry package-mode is off), so
    # the repository root -- not this script's directory -- must be on sys.path
    # for ``import gubbi`` to resolve when invoked as
    # ``scripts/dump_period_fixtures.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    json.dump(build_document(), sys.stdout, sort_keys=True, indent="\t", ensure_ascii=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
