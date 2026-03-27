# ruff: noqa: T201, N806, E501
"""One-time migration: import existing markdown files into DatabaseStorage.

Run this once against a deployment that still has markdown files as source
of truth. After this, DatabaseStorage is canonical and markdown is generated.

Usage:
    python -m journalctl.scripts.migrate_from_markdown \
        --journal-root /path/to/journal/content \
        --db-path /path/to/journal.db

Or via environment variables (same as production):
    JOURNAL_JOURNAL_ROOT=/path/to/journal/content \
    JOURNAL_DB_PATH=/path/to/journal.db \
    python -m journalctl.scripts.migrate_from_markdown
"""

import argparse
import json
import sys
from pathlib import Path

import frontmatter

# Allow running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parents[3]))


def migrate(journal_root: Path, db_path: Path) -> None:
    from journalctl.models.entry import Message, slugify  # noqa: PLC0415
    from journalctl.storage.database import DatabaseStorage  # noqa: PLC0415
    from journalctl.storage.index import SearchIndex  # noqa: PLC0415

    storage = DatabaseStorage(db_path, journal_root)
    index = SearchIndex(db_path)

    # Force schema init
    _ = storage.conn
    _ = index.conn

    topics_dir = journal_root / "topics"
    conversations_dir = journal_root / "conversations"

    topics_migrated = 0
    entries_migrated = 0
    convs_migrated = 0
    messages_migrated = 0

    # ----------------------------------------------------------------
    # Topics + Entries
    # ----------------------------------------------------------------
    if topics_dir.exists():
        for md_file in sorted(topics_dir.rglob("*.md")):
            rel = md_file.relative_to(topics_dir)
            # topic path: strip .md, convert path separators
            topic_path = rel.with_suffix("").as_posix()

            try:
                post = frontmatter.load(str(md_file))
                meta = post.metadata
                title = meta.get("title", topic_path.split("/")[-1].replace("-", " ").title())
                description = meta.get("description", "")
                tags = meta.get("tags", [])
                created = meta.get("created", "2024-01-01")
                updated = meta.get("updated", created)

                # Create topic
                try:
                    storage.conn.execute(
                        """
                        INSERT OR IGNORE INTO topics (path, title, description, tags, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (topic_path, title, description, json.dumps(tags), created, updated),
                    )
                    storage.conn.commit()
                    topics_migrated += 1
                except Exception as e:
                    print(f"  SKIP topic {topic_path}: {e}")
                    continue

                topic_row = storage.conn.execute(
                    "SELECT id FROM topics WHERE path = ?", (topic_path,)
                ).fetchone()
                topic_id = topic_row["id"]

                # Parse entries from markdown body
                import re  # noqa: PLC0415

                ENTRY_DATE_PATTERN = re.compile(
                    r"^## (\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)",
                    re.MULTILINE,
                )
                INLINE_TAG_PATTERN = re.compile(r"#([a-z0-9-]+)")

                parts = ENTRY_DATE_PATTERN.split(post.content)
                position = 0
                for i in range(1, len(parts), 2):
                    date_str = parts[i].strip().split(" ")[0]  # date only
                    content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                    if not content:
                        continue

                    # Extract inline tags from first line
                    first_line = content.split("\n")[0] if content else ""
                    entry_tags = INLINE_TAG_PATTERN.findall(first_line)
                    if entry_tags and first_line.strip() == " ".join(f"#{t}" for t in entry_tags):
                        content = "\n".join(content.split("\n")[1:]).strip()

                    # Skip conversation-summary entries (these are derived from conversations)
                    if "#conversation-summary" in entry_tags:
                        continue

                    position += 1
                    storage.conn.execute(
                        """
                        INSERT INTO entries (topic_id, date, content, tags, position, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            topic_id,
                            date_str,
                            content,
                            json.dumps(entry_tags),
                            position,
                            date_str,
                            date_str,
                        ),
                    )
                    entries_migrated += 1

                storage.conn.commit()
                print(f"  topic: {topic_path} ({position} entries)")

            except Exception as e:
                print(f"  FAIL topic {md_file}: {e}")
                continue

    # ----------------------------------------------------------------
    # Conversations
    # ----------------------------------------------------------------
    if conversations_dir.exists():
        for md_file in sorted(conversations_dir.rglob("*.md")):
            try:
                post = frontmatter.load(str(md_file))
                meta = post.metadata

                topic = meta.get("topic", "")
                title = meta.get("title", md_file.stem)
                source = meta.get("source", "claude")
                tags = meta.get("tags", [])
                summary = meta.get("summary", "")
                created = meta.get("created", "2024-01-01")
                updated = meta.get("updated", created)
                thread = meta.get("thread")
                thread_seq = meta.get("thread_seq")

                if not topic:
                    continue

                # Ensure topic exists
                storage._get_or_create_topic(topic)
                topic_row = storage.conn.execute(
                    "SELECT id FROM topics WHERE path = ?", (topic,)
                ).fetchone()
                topic_id = topic_row["id"]
                slug = slugify(title)

                # Parse messages from markdown body
                messages = []
                import re as _re  # noqa: PLC0415

                MSG_PATTERN = _re.compile(
                    r"###\s+(User|Assistant)(?:\s+\(([^)]*)\))?\s*\n\n(.*?)(?=\n---\n|$)",
                    _re.DOTALL,
                )
                for m in MSG_PATTERN.finditer(post.content):
                    role = "user" if m.group(1) == "User" else "assistant"
                    timestamp = m.group(2)
                    content = m.group(3).strip()
                    if content:
                        messages.append(Message(role=role, content=content, timestamp=timestamp))

                if not messages:
                    print(f"  SKIP conversation (no messages): {md_file}")
                    continue

                participants = sorted({m.role for m in messages})
                storage.conn.execute(
                    """
                    INSERT OR REPLACE INTO conversations
                        (topic_id, title, slug, source, summary, tags, participants,
                         message_count, thread, thread_seq, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        topic_id,
                        title,
                        slug,
                        source,
                        summary,
                        json.dumps(tags),
                        json.dumps(participants),
                        len(messages),
                        thread,
                        thread_seq,
                        created,
                        updated,
                    ),
                )

                conv_row = storage.conn.execute(
                    "SELECT id FROM conversations WHERE topic_id = ? AND slug = ?",
                    (topic_id, slug),
                ).fetchone()
                conv_id = conv_row["id"]

                storage.conn.executemany(
                    """
                    INSERT INTO messages (conversation_id, role, content, timestamp, position)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [(conv_id, m.role, m.content, m.timestamp, i) for i, m in enumerate(messages)],
                )
                storage.conn.commit()

                # Write JSON archive
                storage._write_conversation_json(
                    topic,
                    slug,
                    __import__(
                        "journalctl.models.entry", fromlist=["ConversationMeta"]
                    ).ConversationMeta(
                        source=source,
                        title=title,
                        topic=topic,
                        tags=tags,
                        created=created,
                        updated=updated,
                        summary=summary,
                        participants=participants,
                        message_count=len(messages),
                    ),
                    messages,
                )

                convs_migrated += 1
                messages_migrated += len(messages)
                print(f"  conversation: {topic}/{title} ({len(messages)} messages)")

            except Exception as e:
                print(f"  FAIL conversation {md_file}: {e}")
                continue

    # ----------------------------------------------------------------
    # Rebuild FTS5 index from migrated data
    # ----------------------------------------------------------------
    print("\nRebuilding FTS5 search index...")
    result = index.rebuild_from_db(storage)

    print("\nMigration complete:")
    print(f"  Topics:        {topics_migrated}")
    print(f"  Entries:       {entries_migrated}")
    print(f"  Conversations: {convs_migrated}")
    print(f"  Messages:      {messages_migrated}")
    print(f"  FTS5 docs:     {result['documents_indexed']}")

    storage.close()
    index.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate markdown journal to SQLite")
    parser.add_argument("--journal-root", type=Path, help="Path to journal content directory")
    parser.add_argument("--db-path", type=Path, help="Path to journal.db")
    args = parser.parse_args()

    if args.journal_root and args.db_path:
        migrate(args.journal_root, args.db_path)
    else:
        # Fall back to environment / settings
        from journalctl.config import get_settings  # noqa: PLC0415

        settings = get_settings()
        migrate(settings.journal_root, settings.db_path)
