"""Markdown file storage layer.

Reads and writes topic files and conversation archives with YAML
frontmatter. Entries are split on ## YYYY-MM-DD date headers, NOT
on --- separators (which are ambiguous with frontmatter delimiters
and normal markdown horizontal rules).
"""

import logging
import re
from datetime import date as date_today_cls
from pathlib import Path

import frontmatter
from filelock import FileLock

from journalctl.models.entry import (
    ConversationMeta,
    Entry,
    Message,
    TopicMeta,
    slugify,
    validate_title,
    validate_topic,
)

logger = logging.getLogger(__name__)

# Pattern to split entries: ## YYYY-MM-DD or ## YYYY-MM-DD HH:MM
ENTRY_DATE_PATTERN = re.compile(
    r"^## (\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)",
    re.MULTILINE,
)

# Pattern to extract inline tags like #decision #milestone
INLINE_TAG_PATTERN = re.compile(r"#([a-z0-9-]+)")


class MarkdownStorage:
    """Read/write markdown files with YAML frontmatter."""

    def __init__(self, journal_root: Path) -> None:
        self.journal_root = journal_root
        self.topics_dir = journal_root / "topics"
        self.conversations_dir = journal_root / "conversations"

    # ------------------------------------------------------------------
    # Topic files
    # ------------------------------------------------------------------

    def topic_path(self, topic: str) -> Path:
        """Get the filesystem path for a topic."""
        validate_topic(topic)
        return self.topics_dir / f"{topic}.md"

    def topic_exists(self, topic: str) -> bool:
        return self.topic_path(topic).exists()

    def read_topic(self, topic: str) -> tuple[TopicMeta, str]:
        """Read a topic file. Returns (metadata, body content)."""
        path = self.topic_path(topic)
        if not path.exists():
            msg = f"Topic '{topic}' not found"
            raise FileNotFoundError(msg)

        post = frontmatter.load(str(path))
        meta = TopicMeta(**post.metadata)
        return meta, post.content

    def parse_entries(self, body: str) -> list[Entry]:
        """Parse a topic file body into individual dated entries.

        Splits on ## YYYY-MM-DD headers. Everything before the first
        date header is treated as the preamble (description) and skipped.
        """
        parts = ENTRY_DATE_PATTERN.split(body)
        # parts = [preamble, date1, content1, date2, content2, ...]
        entries: list[Entry] = []
        idx = 1
        for i in range(1, len(parts), 2):
            date = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""

            # Extract inline tags from first line
            first_line = content.split("\n")[0] if content else ""
            tags = INLINE_TAG_PATTERN.findall(first_line)

            # If first line is only tags, remove it from content
            if tags and first_line.strip() == " ".join(f"#{t}" for t in tags):
                content = "\n".join(content.split("\n")[1:]).strip()

            entries.append(
                Entry(
                    index=idx,
                    date=date,
                    tags=tags,
                    content=content,
                )
            )
            idx += 1

        return entries

    def create_topic(
        self,
        topic: str,
        title: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> Path:
        """Create a new topic file with frontmatter."""
        validate_topic(topic)
        path = self.topic_path(topic)
        if path.exists():
            msg = f"Topic '{topic}' already exists"
            raise FileExistsError(msg)

        path.parent.mkdir(parents=True, exist_ok=True)

        today = date_today_cls.today().isoformat()
        meta = TopicMeta(
            topic=topic,
            title=title,
            description=description,
            tags=tags or [],
            created=today,
            updated=today,
            entry_count=0,
        )

        post = frontmatter.Post(
            content=f"# {title}\n\n{description}",
            **meta.model_dump(),
        )

        with self._lock(path):
            frontmatter.dump(post, str(path))

        return path

    def append_entry(
        self,
        topic: str,
        content: str,
        tags: list[str] | None = None,
        date: str | None = None,
    ) -> tuple[Path, int]:
        """Append a dated entry to a topic file.

        Creates the topic file if it doesn't exist. Returns
        (file_path, new_entry_count).
        """
        validate_topic(topic)
        path = self.topic_path(topic)

        d = date or date_today_cls.today().isoformat()
        tag_line = " ".join(f"#{t}" for t in tags) if tags else ""

        if not path.exists():
            # Auto-create with a title derived from topic
            title = topic.split("/")[-1].replace("-", " ").title()
            self.create_topic(topic, title)

        with self._lock(path):
            post = frontmatter.load(str(path))

            new_section = f"\n\n---\n\n## {d}\n"
            if tag_line:
                new_section += f"\n{tag_line}\n"
            new_section += f"\n{content}"

            post.content += new_section
            post.metadata["updated"] = d.split(" ")[0]  # date part only
            post.metadata["entry_count"] = post.metadata.get("entry_count", 0) + 1

            frontmatter.dump(post, str(path))
            return path, post.metadata["entry_count"]

    def update_entry(
        self,
        topic: str,
        entry_index: int,
        content: str,
        mode: str = "replace",
    ) -> Path:
        """Update an existing entry by 1-based index.

        Args:
            topic: Topic path.
            entry_index: 1-based entry position.
            content: New content.
            mode: 'replace' to overwrite, 'append' to add to entry.
        """
        validate_topic(topic)
        path = self.topic_path(topic)
        if not path.exists():
            msg = f"Topic '{topic}' not found"
            raise FileNotFoundError(msg)

        with self._lock(path):
            post = frontmatter.load(str(path))
            entries = self.parse_entries(post.content)

            if entry_index < 1 or entry_index > len(entries):
                msg = f"Entry index {entry_index} out of range " f"(1-{len(entries)})"
                raise IndexError(msg)

            # Rebuild the body with the updated entry
            # re.split with a capturing group gives:
            # [preamble, date1, content1, date2, content2, ...]
            # where dateN is just "YYYY-MM-DD" (without "## " prefix)
            parts = ENTRY_DATE_PATTERN.split(post.content)
            content_idx = 2 * entry_index

            if mode == "replace":
                parts[content_idx] = f"\n\n{content}\n\n"
            elif mode == "append":
                existing = parts[content_idx].strip()
                parts[content_idx] = f"\n\n{existing}\n\n{content}\n\n"
            else:
                msg = f"Invalid mode '{mode}'. Use 'replace' or 'append'."
                raise ValueError(msg)

            # Rejoin: add back "## " prefix before each date
            rebuilt = parts[0]
            for i in range(1, len(parts), 2):
                rebuilt += f"## {parts[i]}{parts[i + 1]}"
            post.content = rebuilt
            post.metadata["updated"] = date_today_cls.today().isoformat()
            frontmatter.dump(post, str(path))

        return path

    def upsert_conversation_summary(
        self,
        topic: str,
        conv_title: str,
        summary: str,
        conv_date: str,
    ) -> None:
        """Insert or update a #conversation-summary entry in a topic.

        If an entry already links to this conversation, update it.
        Otherwise, append a new entry.
        """
        validate_title(conv_title)
        conv_slug = slugify(conv_title)
        link = f"[[conversations/{topic}/{conv_slug}]]"

        path = self.topic_path(topic)
        if not path.exists():
            title = topic.split("/")[-1].replace("-", " ").title()
            self.create_topic(topic, title)

        with self._lock(path):
            post = frontmatter.load(str(path))

            # Check if a summary entry for this conversation exists
            if link in post.content:
                # Replace the existing summary line
                lines = post.content.split("\n")
                new_lines = []
                for line in lines:
                    if link in line:
                        new_lines.append(f"{summary} {link}")
                    else:
                        new_lines.append(line)
                post.content = "\n".join(new_lines)
            else:
                # Append new conversation-summary entry
                tag_line = "#conversation-summary"
                new_section = f"\n\n---\n\n## {conv_date}\n\n{tag_line}\n\n" f"{summary} {link}"
                post.content += new_section
                post.metadata["entry_count"] = post.metadata.get("entry_count", 0) + 1

            post.metadata["updated"] = conv_date.split(" ")[0]
            frontmatter.dump(post, str(path))

    # ------------------------------------------------------------------
    # Conversation archives
    # ------------------------------------------------------------------

    def conversation_path(self, topic: str, title: str) -> Path:
        """Get filesystem path for a conversation archive."""
        validate_topic(topic)
        slug = slugify(title)
        return self.conversations_dir / topic / f"{slug}.md"

    def save_conversation(
        self,
        topic: str,
        title: str,
        messages: list[Message],
        source: str = "claude",
        tags: list[str] | None = None,
        thread: str | None = None,
        thread_seq: int | None = None,
        summary: str | None = None,
    ) -> tuple[Path, str]:
        """Save a conversation transcript. Idempotent.

        Same topic + title = same file (overwrite). Returns
        (file_path, generated_summary).
        """
        validate_topic(topic)
        validate_title(title)
        path = self.conversation_path(topic, title)
        path.parent.mkdir(parents=True, exist_ok=True)

        today = date_today_cls.today().isoformat()
        auto_summary = summary or self._generate_summary(title, messages)

        meta = ConversationMeta(
            type="conversation",
            source=source,
            title=title,
            topic=topic,
            tags=tags or [],
            created=today,
            updated=today,
            summary=auto_summary,
            participants=sorted({m.role for m in messages}),
            message_count=len(messages),
            thread=thread,
            thread_seq=thread_seq,
        )

        # If file exists, preserve original created date
        if path.exists():
            existing = frontmatter.load(str(path))
            if "created" in existing.metadata:
                meta.created = existing.metadata["created"]

        # Format messages
        body_parts = [f"# {title}\n"]
        for msg in messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            ts = f" ({msg.timestamp})" if msg.timestamp else ""
            body_parts.append(f"---\n\n### {role_label}{ts}\n\n{msg.content}")

        body = "\n\n".join(body_parts)
        post = frontmatter.Post(content=body, **meta.model_dump())

        with self._lock(path):
            frontmatter.dump(post, str(path))

        return path, auto_summary

    def list_conversations(
        self,
        topic_prefix: str | None = None,
    ) -> list[ConversationMeta]:
        """List all conversation archives, optionally filtered by topic."""
        results: list[ConversationMeta] = []

        search_dir = self.conversations_dir
        if topic_prefix:
            validate_topic(topic_prefix)
            search_dir = self.conversations_dir / topic_prefix

        if not search_dir.exists():
            return results

        for md_file in sorted(search_dir.rglob("*.md")):
            try:
                post = frontmatter.load(str(md_file))
                meta = ConversationMeta(**post.metadata)
                results.append(meta)
            except (OSError, ValueError, KeyError) as e:
                logger.warning("Skipping malformed conversation %s: %s", md_file, e)
                continue

        return results

    def read_conversation(self, topic: str, title: str) -> tuple[ConversationMeta, str]:
        """Read a conversation archive."""
        path = self.conversation_path(topic, title)
        if not path.exists():
            msg = f"Conversation '{title}' not found under '{topic}'"
            raise FileNotFoundError(msg)

        post = frontmatter.load(str(path))
        meta = ConversationMeta(**post.metadata)
        return meta, post.content

    # ------------------------------------------------------------------
    # Knowledge files
    # ------------------------------------------------------------------

    def read_knowledge(self, name: str) -> str:
        """Read a knowledge file (e.g. user-profile)."""
        path = self.journal_root / "knowledge" / f"{name}.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Topic listing
    # ------------------------------------------------------------------

    def list_topics(
        self,
        prefix: str | None = None,
    ) -> list[TopicMeta]:
        """List all topics with metadata from frontmatter."""
        results: list[TopicMeta] = []

        search_dir = self.topics_dir
        if prefix:
            validate_topic(prefix)
            search_dir = self.topics_dir / prefix

        if not search_dir.exists():
            return results

        for md_file in sorted(search_dir.rglob("*.md")):
            try:
                post = frontmatter.load(str(md_file))
                meta = TopicMeta(**post.metadata)
                results.append(meta)
            except (OSError, ValueError, KeyError) as e:
                logger.warning("Skipping malformed topic %s: %s", md_file, e)
                continue

        # Sort by updated date, most recent first
        results.sort(key=lambda t: t.updated, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lock(self, path: Path) -> FileLock:
        """Get a file lock for the given path."""
        lock_path = path.parent / f".{path.name}.lock"
        return FileLock(str(lock_path), timeout=10)

    def _generate_summary(
        self,
        title: str,
        messages: list[Message],
    ) -> str:
        """Generate a simple summary from the conversation.

        Takes the title and first user message as the summary.
        A more sophisticated approach could use an LLM, but
        we keep the journal server dependency-free from AI APIs.
        """
        first_user_msg = next(
            (m.content[:200] for m in messages if m.role == "user"),
            "",
        )
        if first_user_msg:
            # Truncate at sentence boundary if possible
            for sep in (". ", "? ", "! "):
                idx = first_user_msg.find(sep)
                if 0 < idx < 150:
                    first_user_msg = first_user_msg[: idx + 1]
                    break
            return f"{title} — {first_user_msg}"
        return title
