# Tools Reference

journalctl exposes 12 MCP tools. The connected LLM calls these automatically during conversation based on context — you don't need to invoke them manually.

## Context tools

### journal_briefing

Context loader. Call at conversation start to give the LLM enough background to be immediately helpful.

**Parameters:** None.

**Returns:** User profile (from `knowledge/user-profile.md`), this week's timeline, top 20 recently-active topics, and document counts.

**When it's used:** Automatically at the start of a new conversation, or when the LLM needs to orient itself ("what have I been working on?").

---

### journal_timeline

View all journal activity within a time period.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `period` | string | yes | Time window. Accepts: `YYYY` (year), `YYYY-MM` (month), `YYYY-WNN` (ISO week), `this-week`, `last-week`, `this-month`, `last-month` |

**Returns:** Chronological list of all entries and conversations updated within the period.

**Examples:**
```
journal_timeline("this-week")      → everything from Mon–Sun of the current week
journal_timeline("2026-03")        → everything from March 2026
journal_timeline("2026-W12")       → ISO week 12 of 2026
journal_timeline("last-month")     → everything from the previous calendar month
```

---

## Topic tools

### journal_create_topic

Create a new topic with metadata. Topics are organized as 1–2 level paths (e.g. `hobbies/running`, `projects/homelab`).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | string | yes | Topic path. Max 2 levels, lowercase alphanumeric with hyphens. |
| `title` | string | yes | Human-readable title |
| `description` | string | no | One-line description |
| `tags` | string[] | no | Initial tags |

**Returns:** Confirmation with the created file path.

**Note:** You don't always need to call this explicitly — `journal_append` auto-creates topics on the fly if the topic file doesn't exist yet.

---

### journal_list_topics

List all topics with metadata. Supports pagination and prefix filtering.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prefix` | string | no | — | Filter to topics under this prefix (e.g. `hobbies`). Do **not** include a trailing slash. |
| `limit` | integer | no | 50 | Max topics to return |
| `offset` | integer | no | 0 | Skip first N topics for pagination |

**Returns:** List of topics with title, description, tags, entry count, created/updated dates. Also returns total count for pagination.

---

## Entry tools

### journal_append

Add a dated entry to a topic. This is the primary write operation — most journal updates go through this tool.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | string | yes | Topic path (e.g. `projects/homelab`) |
| `content` | string | yes | Entry content in markdown |
| `tags` | string[] | no | Inline tags (e.g. `["decision", "milestone"]`) |
| `date` | string | no | Date override as `YYYY-MM-DD`. Defaults to today. |

**Returns:** Confirmation with topic, date, and entry count.

**Behavior:** If the topic file doesn't exist, it's created automatically. If `date` is provided, the entry uses that date instead of today — useful for backdating entries.

---

### journal_read

Read a topic's metadata and content.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | string | yes | Topic path |
| `n` | integer | no | If provided, return only the last N entries as structured data. If omitted, return the full topic content as raw markdown. |

**Returns:**

- Without `n`: `{ metadata, content }` — full markdown string.
- With `n`: `{ metadata, entries[], total_entries, showing }` — structured entry objects with index, date, tags, and content.

**Note:** The return shape changes depending on whether `n` is provided. When using `n`, each entry includes its 1-based index for use with `journal_update`.

---

### journal_update

Edit an existing entry by its 1-based index.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `topic` | string | yes | — | Topic path |
| `entry_index` | integer | yes | — | 1-based position of the entry. Use `journal_read` to see indexes. |
| `content` | string | yes | — | New content |
| `mode` | string | no | `replace` | `replace` to overwrite the entire entry, `append` to add content to the end |

**Returns:** Confirmation with updated entry index and mode.

---

## Search

### journal_search

Full-text search across all journal entries and conversations using FTS5.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | — | Search query. Supports FTS5 syntax: `AND`, `OR`, `NOT`, `"exact phrase"`, `prefix*` |
| `topic_prefix` | string | no | — | Filter to topics under this prefix |
| `date_from` | string | no | — | Filter from this date (`YYYY-MM-DD`). Filters on **document updated date**, not individual entry dates. |
| `date_to` | string | no | — | Filter to this date (`YYYY-MM-DD`). Same caveat as `date_from`. |
| `limit` | integer | no | 10 | Max results |

**Returns:** List of results with file path, topic, title, snippet (with `<mark>` highlighting), relevance score, and date.

**Examples:**
```
journal_search("network setup")                                → find all mentions
journal_search("training plan", topic_prefix="hobbies")        → scoped to hobbies/*
journal_search("budget OR cost", date_from="2026-01-01")       → date-filtered
journal_search("\"half marathon\"")                            → exact phrase match
```

**Known limitation:** The `date_from`/`date_to` filters operate on the document's `updated` metadata field, not on individual entry dates within a file. A backdated entry inside a recently-updated file will match date ranges based on the file's update date, not the entry's date.

---

## Conversation tools

### journal_save_conversation

Save a full chat transcript. Idempotent — re-saving the same topic + title overwrites the file. Git preserves all versions via daily cron. Also auto-generates a summary entry in the parent topic file.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `topic` | string | yes | — | Topic this conversation relates to |
| `title` | string | yes | — | Descriptive title (becomes the filename) |
| `messages` | object[] | yes | — | List of messages. Each should have `role` (`"user"` or `"assistant"`), `content` (string), and optional `timestamp` (string). |
| `source` | string | no | `claude` | Origin: `claude`, `chatgpt`, `manual`, or any string |
| `tags` | string[] | no | — | Tags for the conversation |
| `summary` | string | no | — | Summary override. Auto-generated if not provided. |
| `thread` | string | no | — | Thread ID linking related conversations |
| `thread_seq` | integer | no | — | Sequence number within the thread |

**Returns:** File path, summary text, topic, and title.

---

### journal_list_conversations

List all archived conversations, optionally filtered by topic.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | string | no | Filter by topic prefix. Omit to list all conversations. |

**Returns:** List of conversations with title, topic, tags, created/updated dates, summary, message count, participant list, and thread info.

---

### journal_read_conversation

Read a specific archived conversation's full transcript.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | string | yes | Topic the conversation is under |
| `title` | string | yes | Title of the conversation (matches the filename) |

**Returns:** Full conversation metadata (frontmatter) and transcript content.

---

## Admin tools

### journal_reindex

Rebuild the FTS5 search index from scratch by scanning all markdown files. Use when search results seem wrong, or after manually editing markdown files outside the MCP server.

**Parameters:** None.

**Returns:** Number of documents indexed and duration in seconds.

**When to use:** After manual edits to files on disk, after restoring from git, or if search results seem stale or incomplete.
