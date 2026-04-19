# Design Philosophy

## The problem

If you use LLMs as a daily companion for life decisions, project planning, hobbies, and work — your conversations vanish between sessions. Every new chat starts from zero.

Some LLMs offer built-in memory, but it's a black box: you don't control what's remembered, can't browse it, can't port it, and can't use it across different providers. Your context is locked inside one vendor's product.

journalctl is the escape hatch. It gives any MCP-compatible LLM persistent, structured memory that you own — stored in a PostgreSQL database on your own server, with conversation transcripts archived as JSON files, and portable to any future client that supports MCP.

## Journal is a ledger, not a brain

The journal is an **append-only historical record**. It doesn't think, compress, forget, or consolidate. It faithfully stores everything you put in it.

This is a deliberate constraint. A ledger has three properties that matter:

1. **Complete history.** Nothing is lost. You can always go back and see exactly what you decided and why.
2. **Chronological structure.** Entries are dated. Time is the natural index for a human life — you remember things by when they happened.
3. **Durable and portable.** All data lives in PostgreSQL on infrastructure you control. Conversation transcripts are archived as JSON files alongside the database. Every entry is a row in a standard SQL schema — export is a single `pg_dump` away. No external service, no vendor lock-in.

If the journal were a "brain" that consolidated and compressed, you'd lose the timeline, the nuance, and the ability to see how your thinking evolved. A brain is lossy. A ledger is not.

## One store, two query modes

The journal answers both **"what happened?"** and **"what is true now?"** from the same set of entries — but through two different query modes:

| Question | What's needed |
|----------|---------------|
| "When did I decide on that approach?" | Keyword match → **`tsvector` FTS** |
| "Show me the conversation where I researched options" | Transcript retrieval → `journal_read_conversation` |
| "What's the current status of the project?" | Meaning-based match → **`pgvector` semantic search** |
| "What are my preferences for X?" | Semantic fuzzy match → **pgvector** |
| "What happened last week?" | Date-range timeline → `journal_timeline` |

`journal_search` runs FTS and semantic search in parallel, merges by source key, and returns both keyword matches and semantically similar entries in a single response. `journal_briefing` surfaces "key facts" by running a canned semantic query against the same embedding index. There is no separate memory service — a single PostgreSQL database holds everything.

![Journal vs Memory](diagrams/journal-vs-memory.svg)

Distilled facts are just entries. Conversations are just entries with a linked JSON archive. The LLM doesn't orchestrate between systems — it just picks the right tool.

## Why not RAG?

RAG (Retrieval-Augmented Generation) is the standard approach to giving LLMs access to external knowledge. It works by chunking documents into vectors, embedding them, and retrieving the nearest chunks at query time.

journalctl is **not** a RAG system, and doesn't need to be:

**RAG solves the wrong problem.** RAG is designed for large, passive document corpora where the LLM needs to "search and find" relevant context. Your journal isn't a corpus to be searched — it's a structured, curated record that the LLM actively writes to. The LLM knows where things are because it put them there.

**No chunking problem.** RAG's biggest headache is chunking: split too small and you lose context, too big and retrieval gets noisy. Journal entries are naturally chunked — each entry is a coherent unit written by a human or by the LLM, dated and topic-tagged.

**No embedding drift alone.** Semantic similarity can be misleading. Two unrelated topics might score as semantically similar because they share abstract concepts. Pairing semantic search with keyword `tsvector` matching and explicit topic/date filters prevents the LLM from drowning in fuzzy false-positives.

**Agentic retrieval beats statistical retrieval.** RAG retrieves the top-K nearest chunks and hopes the answer is in there. An LLM connected to journalctl makes *deliberate, targeted tool calls* — search for a keyword, then read a specific topic, then check a timeline — iteratively narrowing down exactly what it needs. The retrieval is driven by reasoning, not cosine similarity.

**No infrastructure overhead.** journalctl embeds a local ONNX model (`all-MiniLM-L6-v2`, ~24MB quantized) and stores vectors in `pgvector` alongside the rest of the data. No separate vector database, no chunking pipeline, no external embedding API. Just one PostgreSQL database.

The useful parts of RAG — semantic similarity for fuzzy matches when you don't know the exact keywords — are built into `journal_search`, which merges `tsvector` keyword search with `pgvector` semantic results in a single tool call.

## Why PostgreSQL over SQLite

The earliest versions of journalctl were SQLite + FTS5 + `sqlite-vec`. It worked for a single user but hit two walls: `sqlite-vec` required manual hacks to clean up orphan embeddings when entries were soft-deleted, and the multi-tenant path had no clean story for row-level security. PostgreSQL 17 + `pgvector` + `tsvector` generated columns solved both at once:

1. **One database, zero drift.** `tsvector` is a generated stored column — PostgreSQL keeps it in sync with content on every write, no application code needed. `pgvector` lives in its own table linked to `entries` via `ON DELETE CASCADE` — soft-deletes and hard-deletes both clean up correctly.
2. **Native concurrency.** No WAL-mode quirks, no `busy_timeout` retries. Each gunicorn worker gets its own asyncpg pool, MVCC handles the rest.
3. **HNSW semantic search.** `pgvector`'s HNSW index (tuned `m=32, ef_construction=128`) gives sub-millisecond cosine similarity at tens of thousands of entries, with explicit `WHERE topic_id = ANY(...)` pre-filtering applied in SQL.
4. **Multi-tenant ready.** Row-level security, schema-per-tenant isolation, and proper FK cascades are all first-class in PostgreSQL. The self-hosted personal config scales to the hosted multi-tenant product without a schema rewrite.

Semantic search and full-text search both live in the same database as the entries themselves. `journal_reindex` only rebuilds semantic embeddings — `tsvector` is always current.

## Why append-only

Entries are never deleted or modified in place (except through the explicit `journal_update_entry` tool). New information is appended. Old information stays.

This matters because:

- **Decisions have context.** A decision is only useful if you can also see *why* — which means the research, the alternatives considered, and the reasoning all need to persist.
- **Opinions change.** Your March assessment of something might differ from your June assessment. Both are valuable. An append-only log preserves the evolution.
- **Soft deletes preserve history.** Even when entries are deleted, they're marked as deleted in the database (`deleted_at` timestamp), not physically removed. Backups preserve every version.

## Why self-hosted

- **You own every byte.** No third-party service has your journal data.
- **No usage limits.** Search as much as you want, store as much as you want.
- **LLM-portable.** When you switch LLM providers, point the new client at the same server.
- **Auditable.** Every entry is a row in a standard PostgreSQL schema. Connect with `psql` or any BI tool and query directly.
- **Cost-predictable.** A small VM costs $5–15/month. No per-query pricing.

## Why LLM-agnostic

journalctl uses the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP), an open standard for connecting LLMs to external tools. This means:

- Any MCP-compatible client can connect — CLI tools, desktop apps, browser-based chat, mobile apps.
- You're not locked into any specific LLM provider. Switch providers, keep your journal.
- Multiple clients can connect simultaneously — use one from your phone and another from your terminal.
- Authentication is standard (Bearer tokens or OAuth 2.0), not proprietary.

The journal doesn't know or care which LLM is calling it. It exposes tools, receives requests, and returns data. The intelligence stays in the LLM; the persistence stays in the journal.
