# Taxonomy Guide

How to organize your journal topics for a system that grows with your life.

## Topic structure

Topics use 1–2 level paths: `category/subject`. The first level is the **life domain**, the second is the **specific thing**.

```
topics/
├── hobbies/
│   ├── running.md
│   ├── woodworking.md
│   └── photography.md
├── projects/
│   ├── homelab.md
│   ├── side-project.md
│   └── open-source.md
├── health/
│   ├── fitness.md
│   ├── nutrition.md
│   └── medical.md
├── career/
│   └── current-role.md
├── finance/
│   ├── planning.md
│   └── investments.md
├── learning/
│   └── rust-lang.md
├── creative/
│   └── blog.md
└── travel/
    └── trips.md
```

## Naming conventions

**Category names** (first level): broad life domains. Use a generic noun that could contain 3–10 subtopics. Good: `hobbies`, `projects`, `health`. Bad: `running` (too specific for a category), `stuff` (too vague), `things-i-like` (not a domain).

**Subject names** (second level): the specific thing. Use lowercase, hyphenated, and descriptive enough that the name alone tells you what's inside. Good: `woodworking`, `homelab`, `side-project`. Bad: `hobby1` (meaningless), `misc` (catch-all).

**Validation rules:**
- Lowercase alphanumeric with hyphens only
- Max 2 levels (no `hobbies/woodworking/builds` — use `hobbies/woodworking` and separate entries by date)
- No trailing slashes in API calls
- No spaces, underscores, or special characters

## When to create a new topic vs. append to an existing one

**Create a new topic when:**
- The subject is genuinely distinct. Running and woodworking are different hobbies with different timelines.
- You'd want to read the history independently. "Show me my running progress" should not require scrolling past woodworking entries.
- You'd naturally start a different conversation about it.

**Keep entries in the same topic when:**
- It's the same ongoing thing. Different aspects of a single project all go in one topic — they're facets of the same subject.
- The timeline makes more sense together. A fitness journey includes gym sessions, runs, and recovery — they interleave chronologically and splitting them would lose the narrative.

**Don't over-split.** A topic with 100+ entries is fine — that's the ledger doing its job. You don't need separate topics for every sub-aspect. Just use one topic and let dated entries and tags provide granularity.

## Recommended categories

| Category | Use for | Example subtopics |
|----------|---------|-------------------|
| `hobbies/` | Recreational activities and passions | `running`, `woodworking`, `photography`, `gaming` |
| `projects/` | Software, business, and side projects | `homelab`, `side-project`, `open-source` |
| `health/` | Physical and mental wellbeing | `fitness`, `nutrition`, `medical` |
| `career/` | Work, job, professional development | Company name, role, or domain |
| `finance/` | Money, investments, planning | `planning`, `investments`, `taxes` |
| `learning/` | Courses, research, deep dives | `rust-lang`, `ml-fundamentals` |
| `creative/` | Content creation, art, media | `blog`, `photography`, `podcast` |
| `travel/` | Trips, moves, relocation | `europe-2026`, `roadtrips` |
| `meta/` | Journal system itself | `system-test`, `migration-log` |

## Tags vs. topics

**Topics** are structural — they define *where* an entry lives in the filesystem. You choose a topic once when you write the entry.

**Tags** are descriptive — they annotate *what kind* of entry it is. An entry can have multiple tags, and tags work across topics.

Use tags for cross-cutting concerns:

- `#milestone` — a significant achievement or event
- `#decision` — a choice that was made (and why)
- `#research` — investigation and comparison of options
- `#conversation-summary` — auto-generated when saving a conversation
- `#maintenance` — routine upkeep
- `#purchase` — something bought

You can search by tag content using FTS5: `journal_search("#decision")` finds all entries tagged with `#decision` across all topics.

## Migrating from another LLM

If you're coming from another LLM provider with organized conversations, here's the migration approach:

### Phase 1: Active topics (do first)

For each active project or ongoing topic:

1. **Create the journal topic** — `journal_create_topic("projects/homelab", "Home Lab Setup", ...)`
2. **Write a seed entry** — Summarize the current state: where you are, what's been decided, what's next. This gives the LLM enough context to continue immediately.
3. **Archive key conversations** — Save the 2–3 most important full transcripts using `journal_save_conversation`. Not every chat — just the ones with important decisions or detailed research.

### Phase 2: Bulk archive (do later)

For the rest of your conversation history:

1. Export your data from the previous provider (most offer JSON exports).
2. Parse the export and convert to markdown files matching the conversation format.
3. Map original folders/projects to journal topic categories.
4. Write the files to `conversations/` and run `journal_reindex`.

The bulk archive doesn't need careful curation — it's just making old conversations searchable. The important context was captured in Phase 1.

### What goes where

```
Previous LLM Project "Side Project"
    │
    ├── Active decisions, current state   → journal topic entry (projects/side-project)
    ├── Full research conversation         → conversation archive
    ├── Quick one-off questions            → skip (not worth archiving)
    │
    └── Atomic facts: "uses React, deployed on Vercel" → memory service (future)
```

## Growing the taxonomy

Your taxonomy will evolve. New hobbies appear, projects end, interests shift. That's fine — the two-level structure handles this naturally:

- **New subtopic:** Just append to a new file. `journal_append` auto-creates topics.
- **Topic gets huge:** It's still one file. FTS5 searches within it. `journal_read(topic, n=5)` shows the most recent entries. Don't split unless the subtopics are genuinely different areas.
- **Topic goes dormant:** Leave it. It's an append-only ledger. Old topics are historical records.
- **Wrong category:** Move the file in the filesystem, run `journal_reindex`. Git tracks the rename.
- **New category:** Just start using it. Create `travel/europe-2026` and it exists.

The taxonomy is a filesystem. It's as flexible as `mkdir` and `mv`.
