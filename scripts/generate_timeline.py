"""Generate static timeline markdown files for MkDocs browsing.

Reads all topic files, extracts dated entries, and generates:
  - timeline/YYYY/_index.md   (yearly summary)
  - timeline/YYYY/MM.md       (monthly entries)

Run by the daily cron job, NOT by the server.

Usage:
    python scripts/generate_timeline.py /data/journal
"""

import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import frontmatter

ENTRY_DATE_PATTERN = re.compile(
    r"^## (\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)",
    re.MULTILINE,
)

SKIP_DIRS = {"timeline", "knowledge", ".git"}


def extract_first_sentence(text: str, max_len: int = 120) -> str:
    """Extract first sentence from entry content."""
    # Skip inline tags line
    lines = text.strip().split("\n")
    content = ""
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            content = stripped
            break

    if not content:
        return ""

    for sep in (". ", "? ", "! "):
        idx = content.find(sep)
        if 0 < idx < max_len:
            return content[: idx + 1]

    return content[:max_len] + ("..." if len(content) > max_len else "")


def collect_entries(journal_root: Path) -> dict:
    """Collect all dated entries from topic files.

    Returns: dict[year][month] -> list of (date, topic, summary, tags)
    """
    entries_by_month: dict = defaultdict(lambda: defaultdict(list))

    topics_dir = journal_root / "topics"
    if not topics_dir.exists():
        return entries_by_month

    for md_file in topics_dir.rglob("*.md"):
        rel = md_file.relative_to(topics_dir)
        topic = str(rel.with_suffix("")).replace("\\", "/")

        try:
            post = frontmatter.load(str(md_file))
        except Exception:
            continue

        body = post.content
        parts = ENTRY_DATE_PATTERN.split(body)

        for i in range(1, len(parts), 2):
            entry_date = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""

            # Extract tags
            first_line = content.split("\n")[0] if content else ""
            tags = re.findall(r"#([a-z0-9-]+)", first_line)

            summary = extract_first_sentence(content)
            date_part = entry_date.split(" ")[0]  # YYYY-MM-DD

            try:
                year = int(date_part[:4])
                month = int(date_part[5:7])
            except (ValueError, IndexError):
                continue

            entries_by_month[year][month].append(
                {
                    "date": date_part,
                    "topic": topic,
                    "summary": summary,
                    "tags": tags,
                    "source_path": f"topics/{topic}",
                }
            )

    # Sort entries within each month by date
    for year in entries_by_month:
        for month in entries_by_month[year]:
            entries_by_month[year][month].sort(key=lambda e: e["date"])

    return entries_by_month


def generate_monthly_file(
    timeline_dir: Path,
    year: int,
    month: int,
    entries: list,
) -> None:
    """Generate a monthly timeline file."""
    month_dir = timeline_dir / str(year)
    month_dir.mkdir(parents=True, exist_ok=True)

    month_name = date(year, month, 1).strftime("%B")
    path = month_dir / f"{month:02d}.md"

    lines = [
        "---",
        "type: timeline",
        "view: month",
        f"period: {year}-{month:02d}",
        f"entry_count: {len(entries)}",
        f"generated: {date.today().isoformat()}",
        "---",
        "",
        f"# {month_name} {year}",
        "",
    ]

    current_date = None
    for entry in entries:
        if entry["date"] != current_date:
            current_date = entry["date"]
            lines.append(f"## {current_date}")
            lines.append("")

        tag_str = " ".join(f"#{t}" for t in entry["tags"]) if entry["tags"] else ""
        link = f"[[{entry['source_path']}]]"
        line = f"- **{entry['topic']}**: {entry['summary']} {link}"
        if tag_str:
            line += f" {tag_str}"
        lines.append(line)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def generate_yearly_index(
    timeline_dir: Path,
    year: int,
    months: dict,
) -> None:
    """Generate a yearly index file."""
    year_dir = timeline_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / "_index.md"

    total_entries = sum(len(e) for e in months.values())

    # Collect topic stats
    topic_counts: dict[str, int] = defaultdict(int)
    for month_entries in months.values():
        for entry in month_entries:
            topic_counts[entry["topic"]] += 1

    lines = [
        "---",
        "type: timeline",
        "view: year",
        f"year: {year}",
        f"entry_count: {total_entries}",
        f"generated: {date.today().isoformat()}",
        "---",
        "",
        f"# {year}",
        "",
        "<!-- MANUAL SECTION: curate highlights here -->",
        "",
        "## Highlights",
        "",
        "_No highlights curated yet._",
        "",
        "<!-- END MANUAL SECTION -->",
        "",
        "## Summary by Topic",
        "",
        "| Topic | Entries |",
        "|-------|---------|",
    ]

    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {topic} | {count} |")

    lines.extend(["", "## Months", ""])
    for month_num in sorted(months.keys()):
        month_name = date(year, month_num, 1).strftime("%B")
        count = len(months[month_num])
        lines.append(f"- [{month_name}]({month_num:02d}.md) ({count} entries)")

    # Preserve manual highlights if file exists
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        manual_start = existing.find("<!-- MANUAL SECTION")
        manual_end = existing.find("<!-- END MANUAL SECTION -->")
        if manual_start >= 0 and manual_end >= 0:
            manual_block = existing[manual_start : manual_end + len("<!-- END MANUAL SECTION -->")]
            # Replace the placeholder in new content
            content = "\n".join(lines)
            placeholder_start = content.find("<!-- MANUAL SECTION")
            placeholder_end = content.find("<!-- END MANUAL SECTION -->")
            if placeholder_start >= 0 and placeholder_end >= 0:
                content = (
                    content[:placeholder_start]
                    + manual_block
                    + content[placeholder_end + len("<!-- END MANUAL SECTION -->") :]
                )
                path.write_text(content, encoding="utf-8")
                return

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate all timeline files."""
    if len(sys.argv) < 2:
        print("Usage: python generate_timeline.py <journal_root>")
        sys.exit(1)

    journal_root = Path(sys.argv[1])
    timeline_dir = journal_root / "timeline"

    print(f"Generating timeline from {journal_root}...")
    entries_by_month = collect_entries(journal_root)

    total = 0
    for year, months in sorted(entries_by_month.items()):
        for month, month_entries in sorted(months.items()):
            generate_monthly_file(timeline_dir, year, month, month_entries)
            total += len(month_entries)
        generate_yearly_index(timeline_dir, year, months)

    print(f"Generated timeline: {total} entries across {len(entries_by_month)} years")


if __name__ == "__main__":
    main()
