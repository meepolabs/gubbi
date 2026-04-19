# Tools Reference

journalctl exposes 13 MCP tools. The connected LLM calls these automatically during conversation based on context — you don't need to invoke them manually.

## Context tools

### journal_briefing

Context loader. Call at conversation start to give the LLM enough background to be immediately helpful.

**Parameters:** None.

**Returns:** User profile, key facts (top semantic memory matches), this week's timeline, top 20 recently-active topics, and document counts.

**When it's used:** Automatically at the start of a new conversation, or when the LLM needs to orient itself ("what have I been working on?").

---

### journal_timeline

View all journal activity within a time period.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `period` | string | yes | Time window. Accepts: `today`, `this-week`, `last-week`, `this-month`, `last-month`, `YYYY` (e.g. `2026`), `YYYY-MM` (e.g. `2026-03`), `YYYY-WNN` (e.g. `2026-W14`) |

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

**Note:** Must be called before writing entries or conversations to a new topic. `journal_append_entry` does **not** auto-create topics.

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

### journal_append_entry

Add a dated entry to a topic. This is the primary write operation — most journal updates go through this tool.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | string | yes | Topic path (e.g. `projects/homelab`) |
| `content` | string | yes | Entry content in markdown |
| `reasoning` | string | no | Optional reasoning/background context (stored separately from content) |
| `tags` | string[] | no | Inline tags (e.g. `["decision", "milestone"]`) |
| `date` | string | no | Date override as `YYYY-MM-DD`. Defaults to today. |

**Returns:** Confirmation with topic, date, entry_id, and entry count.

**Behavior:** The topic must already exist — use `journal_create_topic` first if needed. If `date` is provided, the entry uses that date instead of today — useful for backdating entries.

---

### journal_read_topic

Read a topic's metadata and content.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `topic` | string | yes | — | Topic path |
| `limit` | integer | no | 10 | Max entries to return (max 500). |
| `date_from` | string | no | — | Filter from this date (`YYYY-MM-DD`). |
| `date_to` | string | no | — | Filter to this date (`YYYY-MM-DD`). |
| `offset` | integer | no | 0 | Skip first N entries for pagination. |

**Returns:** `{ metadata, entries[], total, limit, offset }` — structured entry objects with id, date, content, reasoning, and tags. Each entry includes its database ID for use with `journal_update_entry`.

---

### journal_update_entry

Edit an existing entry by its database ID.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `entry_id` | integer | yes | — | Database ID of the entry. Get this from `journal_read_topic` or `journal_search` results. |
| `content` | string | no | — | New entry content. Omit to leave unchanged. |
| `reasoning` | string | no | — | New reasoning/background context. Omit to keep current. |
| `mode` | string | no | `replace` | `replace` to overwrite content, `append` to add to the end. |
| `date` | string | no | — | Override the entry date as `YYYY-MM-DD`. Omit to keep current. |
| `tags` | string[] | no | — | Replace entry tags. Omit to keep current. |

**Returns:** Confirmation with updated entry ID and mode.

---

### journal_delete_entry

Soft-delete an existing entry by its database ID. The entry is marked with a `deleted_at` timestamp and excluded from reads and searches. The linked row in `entry_embeddings` is removed in the same CTE, so semantic search also stops returning it immediately.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entry_id` | integer | yes | Database ID of the entry. Get this from a `journal_read_topic(topic, limit=...)` response, or from `journal_search` results. |

**Returns:** Confirmation with deleted entry ID.

---

## Search

### journal_search

Hybrid search across all journal entries and conversations — `tsvector` keyword matching merged with `pgvector` semantic similarity. Both searches run in one tool call and results are deduplicated before being returned.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | — | Search query. Uses `websearch_to_tsquery` — accepts natural language plus `AND` / `OR` / `-negation` / `"exact phrase"`. Trailing operators are safely handled. |
| `topic_prefix` | string | no | — | Filter to topics under this prefix (no trailing slash) |
| `date_from` | string | no | — | Filter from this date (`YYYY-MM-DD`). Applied to entry date / conversation updated date. |
| `date_to` | string | no | — | Filter to this date (`YYYY-MM-DD`). |
| `limit` | integer | no | 10 | Max results |

**Returns:** List of results with `doc_type`, `topic`, `title`, snippet, relevance score, date, and `entry_id`/`conversation_id` for follow-up calls. Response also includes `semantic_available: bool` — if the ONNX encode step fails, the tool transparently degrades to FTS-only and sets this flag to `false` so the client can tell.

**Examples:**
```
journal_search("network setup")                                → find all mentions
journal_search("training plan", topic_prefix="hobbies")        → scoped to hobbies/*
journal_search("budget OR cost", date_from="2026-01-01")       → date-filtered
journal_search("\"half marathon\"")                            → exact phrase match
```

---

## Conversation tools

### journal_save_conversation

Save a full chat transcript. Idempotent — re-saving the same topic + title overwrites the previous version.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `topic` | string | yes | — | Topic this conversation relates to |
| `title` | string | yes | — | Descriptive title for the conversation |
| `messages` | object[] | yes | — | List of messages. Each should have `role` (`"user"` or `"assistant"`), `content` (string), and optional `timestamp` (string). |
| `summary` | string | yes | — | Concise summary of the conversation (1-3 sentences). |
| `source` | string | no | `claude` | Origin: `claude`, `chatgpt`, `manual`, or any string |
| `tags` | string[] | no | — | Tags for the conversation |
**Returns:** `status` (`"saved"` or `"updated"`), `conversation_id`, `summary`, `topic`, `title`.

---

### journal_list_conversations

List all archived conversations, optionally filtered by topic.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `topic_prefix` | string | no | — | Filter by topic prefix. Omit to list all conversations. |
| `limit` | integer | no | 50 | Max conversations to return (max 200). |
| `offset` | integer | no | 0 | Skip first N conversations for pagination. |

**Returns:** List of conversations with id, title, topic, tags, date, summary, and message count.

---

### journal_read_conversation

Read a specific archived conversation's full transcript.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `conversation_id` | integer | yes | Database ID of the conversation. Get this from `journal_list_conversations` response. |
| `preview` | boolean | no | If true, return only the first and last 3 messages instead of the full transcript. |

**Returns:** Full conversation metadata and transcript content (or preview if requested).

---

## Admin tools

### journal_reindex

Rebuild semantic embeddings (`pgvector`) for every active entry. The `tsvector` full-text index is a `GENERATED ALWAYS STORED` column, so it's always current and is **not** touched by this tool.

**Parameters:** None.

**Returns:** `status` (`ok`, `cooldown`, or `already_running`), `semantic_status` (`ok` / `partial`), `embeddings_generated`, `embeddings_failed`, and `duration_seconds`.

**When to use:** If semantic search results look off, or after swapping the embedding model. Rarely needed in normal operation.

**Safety features:**

- **Cooldown.** If a reindex ran in the last 60 seconds, returns `{status: "cooldown"}` without doing work. Uses `MAX(indexed_at) FROM entries` as a shared cross-worker "last reindex" timestamp.
- **Advisory lock.** A PostgreSQL session-level advisory lock guarantees only one worker runs the reindex at a time across the whole gunicorn pool. If the lock can't be acquired, returns `{status: "already_running"}`.
