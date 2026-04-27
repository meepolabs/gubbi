# Tools Reference

journalctl exposes 12 MCP tools. The connected LLM calls these automatically during conversation based on context -- you don't need to invoke them manually.

All tools that read or write prose run under PostgreSQL row-level security with per-user isolation, and content is encrypted (AES-256-GCM) at rest. Decrypted content is only ever in memory while a tool is processing a request.

## Context tools

### journal_briefing

Context loader. Call at conversation start to give the LLM enough background to be immediately helpful.

**Parameters:** None.

**Returns:** `user_profile`, `key_facts` (top semantic memory matches against a canned key-facts query), `this_week` (entries + conversations updated this week), `topics` (top 20 recently-active), `topic_count`, and `stats`.

**When it's used:** Automatically at the start of a new conversation, or when the LLM needs to orient itself ("what have I been working on?").

---

### journal_timeline

View all journal activity within a time period.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `period` | string | yes | Time window. Accepts: `today`, `this-week`, `last-week`, `this-month`, `last-month`, `YYYY` (e.g. `2026`), `YYYY-MM` (e.g. `2026-03`), `YYYY-WNN` (e.g. `2026-W14`) |

**Returns:** Chronological list of all entries and conversations updated within the period. Period boundaries respect the configured timezone.

**Examples:**
```
journal_timeline("this-week")      -> everything from Mon-Sun of the current week
journal_timeline("2026-03")        -> everything from March 2026
journal_timeline("2026-W12")       -> ISO week 12 of 2026
journal_timeline("last-month")     -> everything from the previous calendar month
```

---

## Topic tools

### journal_create_topic

Create a new topic with metadata. Topics are organized as 1-2 level paths (e.g. `hobbies/running`, `projects/homelab`).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | string | yes | Topic path. Max 2 levels, lowercase alphanumeric with hyphens. |
| `title` | string | yes | Human-readable title |
| `description` | string | no | One-line description |
| `tags` | string[] | no | Initial tags |

**Returns:** Confirmation with the created topic path.

**Note:** Must be called before writing entries or conversations to a new topic. `journal_append_entry` does **not** auto-create topics.

---

### journal_list_topics

List all topics with metadata. Supports pagination and prefix filtering.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prefix` | string | no | -- | Filter to topics under this prefix (e.g. `hobbies`). Do **not** include a trailing slash. |
| `limit` | integer | no | 50 | Max topics to return |
| `offset` | integer | no | 0 | Skip first N topics for pagination |

**Returns:** List of topics with title, description, tags, entry count, created/updated dates. Also returns total count for pagination.

---

## Entry tools

### journal_append_entry

Add a dated entry to a topic. This is the primary write operation -- most journal updates go through this tool.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | string | yes | Topic path (e.g. `projects/homelab`) |
| `content` | string | yes | Entry content in markdown |
| `reasoning` | string | no | Optional reasoning/background context (stored separately from content) |
| `tags` | string[] | no | Inline tags (e.g. `["decision", "milestone"]`) |
| `date` | string | no | Date override as `YYYY-MM-DD`. Defaults to today in the configured timezone. |

**Returns:** Confirmation with topic, date, entry_id, and entry count.

**Behavior:** The topic must already exist -- use `journal_create_topic` first if needed. `content` and `reasoning` are encrypted (AES-256-GCM) before insertion; the `search_vector` (`tsvector`) is computed inline at write time from ephemeral plaintext, so PostgreSQL only ever stores ciphertext + a tokenized index. After commit, the entry is auto-embedded via the local ONNX model and upserted into `entry_embeddings`. Embedding is best-effort -- if it fails, the entry is still saved.

---

### journal_read_topic

Read a topic's metadata and content.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `topic` | string | yes | -- | Topic path |
| `limit` | integer | no | 10 | Max entries to return (max 500). |
| `date_from` | string | no | -- | Filter from this date (`YYYY-MM-DD`). |
| `date_to` | string | no | -- | Filter to this date (`YYYY-MM-DD`). |
| `offset` | integer | no | 0 | Skip first N entries for pagination. |

**Returns:** `{ metadata, entries[], total, limit, offset }` -- structured entry objects with id, date, decrypted content, decrypted reasoning, and tags. Each entry includes its database ID for use with `journal_update_entry`.

---

### journal_update_entry

Edit an existing entry by its database ID.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `entry_id` | integer | yes | -- | Database ID of the entry. Get this from `journal_read_topic` or `journal_search` results. |
| `content` | string | no | -- | New entry content. Omit to leave unchanged. |
| `reasoning` | string | no | -- | New reasoning/background context. Omit to keep current. |
| `mode` | string | no | `replace` | `replace` to overwrite content, `append` to add to the end. |
| `date` | string | no | -- | Override the entry date as `YYYY-MM-DD`. Omit to keep current. |
| `tags` | string[] | no | -- | Replace entry tags. Omit to keep current. |

**Returns:** Confirmation with updated entry ID and mode.

**Behavior:** The repo decrypts the existing row inside the transaction (needed for `mode='append'` and to recompute `search_vector`), re-encrypts the new value, clears `indexed_at`, and re-embeds outside the transaction.

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

Hybrid search across all journal entries and conversations -- `tsvector` keyword matching merged with `pgvector` semantic similarity. Both searches run in one tool call and results are deduplicated before being returned.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | -- | Search query. Uses `websearch_to_tsquery` -- accepts natural language plus `AND` / `OR` / `-negation` / `"exact phrase"`. Trailing operators are safely handled. |
| `topic_prefix` | string | no | -- | Filter to topics under this prefix (no trailing slash) |
| `date_from` | string | no | -- | Filter from this date (`YYYY-MM-DD`). Applied to entry date / conversation updated date. |
| `date_to` | string | no | -- | Filter to this date (`YYYY-MM-DD`). |
| `limit` | integer | no | 10 | Max results |

**Returns:** `{ results, total, query }`. Each result is one of:

- `{ doc_type: "entry", topic, date, entry_id, conversation_id: null, content }` -- full decrypted entry content, truncated at a fixed character cap per result.
- `{ doc_type: "conversation", topic, date, entry_id: null, conversation_id, title, summary }` -- decrypted title and summary, truncated to share the same per-result budget.

The repo SQL no longer uses `ts_headline` (the prose column is encrypted, so server-side snippet extraction is impossible). Tool callers receive full decrypted content instead of an excerpt -- better signal for LLM consumers, and the LLM can summarize or excerpt itself.

If query embedding fails (ONNX path unavailable), the tool transparently degrades to FTS-only -- the response shape is unchanged.

**Examples:**
```
journal_search("network setup")                                -> find all mentions
journal_search("training plan", topic_prefix="hobbies")        -> scoped to hobbies/*
journal_search("budget OR cost", date_from="2026-01-01")       -> date-filtered
journal_search("\"half marathon\"")                            -> exact phrase match
```

---

## Conversation tools

### journal_save_conversation

Save a full chat transcript. Idempotent -- re-saving the same topic + title overwrites the previous version.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `topic` | string | yes | -- | Topic this conversation relates to |
| `title` | string | yes | -- | Descriptive title for the conversation |
| `messages` | object[] | yes | -- | List of messages. Each must have `role` (`"user"` or `"assistant"`), `content` (string), and optional `timestamp` (string). Other roles (system, tool) are filtered out. |
| `summary` | string | yes | -- | Concise summary of the conversation (1-3 sentences). |
| `source` | string | no | `claude` | Origin: `claude`, `chatgpt`, `manual`, or any string |
| `tags` | string[] | no | -- | Tags for the conversation |
| `date` | string | no | today | Conversation date (`YYYY-MM-DD`). Defaults to today in the configured timezone. |

**Returns:** `status` (`"saved"` or `"updated"`), `conversation_id`, `summary`, `topic`, `title`. May include a `note` field describing dropped tags or empty messages.

**Behavior:** The JSON archive is written first to `conversations_json/{uuid}.json`, then a single transaction encrypts and upserts the conversation row, inserts encrypted messages, and upserts a linked entry tagged `['conversation']`. The linked entry is auto-embedded after commit so the saved conversation surfaces in semantic search and the timeline.

---

### journal_list_conversations

List all archived conversations, optionally filtered by topic.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `topic_prefix` | string | no | -- | Filter by topic prefix. Omit to list all conversations. |
| `limit` | integer | no | 50 | Max conversations to return (max 200). |
| `offset` | integer | no | 0 | Skip first N conversations for pagination. |

**Returns:** List of conversations with id, title (decrypted), topic, tags, date, summary (decrypted), and message count, plus total/limit/offset.

---

### journal_read_conversation

Read a specific archived conversation's full transcript.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `conversation_id` | integer | yes | Database ID of the conversation. Get this from `journal_list_conversations` response. |
| `preview` | boolean | no | If true, return only the first and last 3 messages instead of the full transcript. |

**Returns:** `{ metadata, content, preview, messages_shown, messages_total }` -- metadata includes the decrypted title/summary; content is the full markdown-rendered transcript (or preview).

---

## Admin tools

(No public admin tools exposed as MCP tools. Reindex is an internal library function called by future admin APIs, not registered on the MCP server. The `tsvector` index is always current because it's populated inline by the repo on every write -- only semantic embeddings are ever rebuilt.)
